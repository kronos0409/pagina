import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import networkx as nx
import warnings
from graphviz import Digraph
from PIL import Image, ImageDraw, ImageFont
import io
import tempfile
import os
import shutil

warnings.simplefilter(action='ignore', category=FutureWarning)

class CHAIDNode:
    """Nodo para el árbol de decisión CHAID."""
    def __init__(self, data, target, features, parent=None, depth=0):
        self.data = data
        self.target = target
        self.features = features
        self.parent = parent
        self.depth = depth
        self.split_feature = None
        self.split_vals = None
        self.p_value = None
        self.chi_squared = None
        self.f_stat = None
        self.df1 = None
        self.df2 = None
        self.children = {}
        self.prediction = np.mean(data[target]) if len(data) > 0 else 0
        self.std = np.std(data[target]) if len(data) > 1 else 0
        self.node_id = id(self)
        self.percentage = 100  # Para el nodo raíz
        self.n = len(data)

    def _plot_histogram(self, is_large_dataset=False):
        """Genera un histograma para el nodo, ajustando DPI y tamaño según el tamaño del dataset."""
        # Ajustar el tamaño y DPI según si el dataset es grande
        if is_large_dataset:
            figsize = (10, 4)
            dpi = 600  # Reducir DPI para histogramas, ya que el PDF manejará la calidad
            fontsize = 28
        else:
            figsize = (8, 3)
            dpi = 600
            fontsize = 24

        # Crear la figura y el eje principal
        fig, ax1 = plt.subplots(figsize=figsize)

    # Generar el histograma
        counts, bins, _ = ax1.hist(
        self.data[self.target], 
        bins=10, 
        color='royalblue', 
        edgecolor='black', 
        alpha=0.7
        )

        # Ajustar el eje Y principal (frecuencia absoluta)
        ax1.set_xlabel(self.target, fontsize=fontsize)
        ax1.set_ylabel('Frecuencia', fontsize=fontsize, color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue', labelsize=fontsize-4)

        # Reducir la cantidad de ticks en el eje Y para evitar que se vean apretados
        max_count = int(max(counts))
        if max_count > 0:
            step = max(1, max_count // 4)  # Mostrar aproximadamente 4-5 ticks
            ax1.set_yticks(range(0, max_count + 1, step))
        ax1.tick_params(axis='x', labelsize=fontsize-4)
        ax1.grid(True, alpha=0.3)

        # Calcular la frecuencia relativa acumulada
        total_count = sum(counts)
        if total_count > 0:
            relative_freq = counts / total_count  # Frecuencia relativa
            cumulative_relative_freq = np.cumsum(relative_freq)  # Frecuencia relativa acumulada

            # Crear el segundo eje Y para la frecuencia relativa acumulada
            ax2 = ax1.twinx()
            ax2.plot(
                bins[:-1] + np.diff(bins) / 2,  # Centros de los bins
                cumulative_relative_freq, 
                color='red', 
                linestyle='-', 
                marker='o', 
                markersize=5, 
                label='Frecuencia Relativa Acumulada'
            )
            ax2.set_ylabel('Frecuencia Relativa Acumulada', fontsize=10, color='red')
            ax2.tick_params(axis='y', labelcolor='red', labelsize=fontsize-4)
            ax2.set_ylim(0, 1)  # La frecuencia relativa acumulada va de 0 a 1

            # Ajustar los ticks del segundo eje Y (frecuencia relativa acumulada)
            ax2.set_yticks(np.linspace(0, 1, 6))  # Mostrar 6 ticks (0, 0.2, 0.4, ..., 1.0)

        # Ajustar el layout para evitar superposiciones
        plt.tight_layout()

        # Guardar la figura en un buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=dpi)
        plt.close()
        buf.seek(0)
        return buf

    def __repr__(self):
        if self.split_feature:
            return f"Node(feature={self.split_feature}, n={len(self.data)}, mean={self.prediction:.4f})"
        return f"Leaf(n={len(self.data)}, mean={self.prediction:.4f})"

class CHAID(BaseEstimator):
    """
    Implementación mejorada de árboles de decisión usando metodología CHAID.
    Similar a la implementación en SPSS pero para variable objetivo numérica.
    Visualización adaptada para generar PDF para bases de datos grandes.
    """
    def __init__(self, alpha_merge=0.05, alpha_split=0.05, max_depth=10, min_sample_node=30,
                 min_sample_split=50, min_children=2, max_children=None, max_iterations=10000, 
                 bonferroni_adjustment=True, large_dataset_threshold=10000):
        self.alpha_merge = alpha_merge
        self.alpha_split = alpha_split
        self.max_depth = max_depth
        self.min_sample_node = min_sample_node
        self.min_sample_split = min_sample_split
        self.min_children = min_children
        self.max_children = max_children
        self.max_iterations = max_iterations
        self.bonferroni_adjustment = bonferroni_adjustment
        self.large_dataset_threshold = large_dataset_threshold
        self.root = None
        self.total_samples = 0
        self.orphan_groups = []
        self.node_counter = 0
        self.is_large_dataset = False
    def get_terminal_nodes_data(self, original_df):
        """Recolecta los datos de los nodos terminales y huérfanos con sus etiquetas."""
        if not self.root:
            raise ValueError("El modelo debe ser entrenado antes de obtener los nodos terminales.")

        terminal_data = []
        visited_indices = set()

        def traverse_node(node, df_subset):
            if not node.children:  # Es un nodo terminal (hoja)
                df_subset = df_subset.copy()
                df_subset['NODO'] = f"node_{node.node_id}"
                terminal_data.append(df_subset)
                visited_indices.update(df_subset.index)
                return
            
            if node.split_feature and node.split_vals:
                df_subset['_temp_group'] = df_subset[node.split_feature].map(node.split_vals)
                for group, child in node.children.items():
                    child_data = df_subset[df_subset['_temp_group'] == group].drop('_temp_group', axis=1)
                    traverse_node(child, child_data)
                if '_temp_group' in df_subset.columns:
                    df_subset.drop('_temp_group', axis=1, inplace=True)

        # Recorrer el árbol desde la raíz
        traverse_node(self.root, original_df)

        # Manejar los grupos huérfanos
        for orphan in self.orphan_groups:
            orphan_data = orphan['data'].copy()
            orphan_data['NODO'] = 'huérfanos'
            terminal_data.append(orphan_data)
            visited_indices.update(orphan_data.index)

        # Combinar todos los datos en un solo DataFrame
        if terminal_data:
            result_df = pd.concat(terminal_data, axis=0, ignore_index=False)
            # Asegurarse de que todos los índices del DF original estén presentes
            missing_indices = original_df.index.difference(visited_indices)
            if not missing_indices.empty:
                missing_data = original_df.loc[missing_indices].copy()
                missing_data['NODO'] = 'sin_clasificar'  # Por si hay datos no clasificados
                result_df = pd.concat([result_df, missing_data], axis=0, ignore_index=False)
            return result_df.sort_index()
        else:
            result_df = original_df.copy()
            result_df['NODO'] = 'sin_nodos_terminales'
            return result_df
    def _merge_categories(self, data, feature, target):
        """Fusiona categorías similares basándose en la variable objetivo (de mejor_arbol)."""
        categories = data[feature].unique()

        if len(categories) <= 1:
            return {cat: 0 for cat in categories}, 0, 0, 1.0, 0, 0

        category_stats = {}
        for cat in categories:
            cat_values = data[data[feature] == cat][target].values
            if len(cat_values) > 0:
                category_stats[cat] = {
                    'mean': np.mean(cat_values),
                    'std': np.std(cat_values) if len(cat_values) > 1 else 0,
                    'count': len(cat_values),
                    'values': cat_values
                }

        sorted_cats = sorted(categories, key=lambda x: (
            category_stats.get(x, {'mean': 0})['mean'] if x in category_stats else 0,
            category_stats.get(x, {'std': 0})['std'] if x in category_stats else 0,
            category_stats.get(x, {'count': 0})['count'] if x in category_stats else 0
        ))

        groups = {cat: i for i, cat in enumerate(sorted_cats)}
        n_groups = len(sorted_cats)

        iteration = 0
        changes_made = True

        while changes_made and n_groups > 1 and iteration < self.max_iterations:
            changes_made = False
            iteration += 1

            p_values = {}
            group_means = {}
            group_stds = {}
            group_counts = {}

            for i in range(n_groups - 1):
                group1 = i
                group2 = i + 1

                cats_group1 = [cat for cat, g in groups.items() if g == group1]
                cats_group2 = [cat for cat, g in groups.items() if g == group2]

                if not cats_group1 or not cats_group2:
                    continue

                vals_group1 = np.concatenate([category_stats[cat]['values'] for cat in cats_group1 if cat in category_stats])
                vals_group2 = np.concatenate([category_stats[cat]['values'] for cat in cats_group2 if cat in category_stats])

                if len(vals_group1) > 0 and len(vals_group2) > 0:
                    from statsmodels.formula.api import ols
                    from statsmodels.stats.anova import anova_lm

                    temp_df = pd.DataFrame({
                        'y': np.concatenate([vals_group1, vals_group2]),
                        'group': np.concatenate([np.ones(len(vals_group1)), np.ones(len(vals_group2)) * 2])
                    })
                    model = ols('y ~ C(group)', data=temp_df).fit()
                    anova_result = anova_lm(model)
                    f_stat = anova_result['F'][0]
                    p_value = anova_result['PR(>F)'][0]

                    if np.isnan(p_value):
                        p_value = 0

                    p_values[(group1, group2)] = p_value
                    group_means[group1] = np.mean(vals_group1)
                    group_means[group2] = np.mean(vals_group2)
                    group_stds[group1] = np.std(vals_group1) if len(vals_group1) > 1 else 0
                    group_stds[group2] = np.std(vals_group2) if len(vals_group2) > 1 else 0
                    group_counts[group1] = len(vals_group1)
                    group_counts[group2] = len(vals_group2)

            if p_values:
                def fusion_key(pair_item):
                    (group1, group2), p_val = pair_item
                    mean_diff = abs(group_means.get(group1, 0) - group_means.get(group2, 0))
                    total_count = group_counts.get(group1, 0) + group_counts.get(group2, 0)
                    return (p_val, -mean_diff, total_count)

                best_pair = max(p_values.items(), key=fusion_key)
                best_p_value = best_pair[1]

                group1, group2 = best_pair[0]
                count1 = group_counts.get(group1, 0)
                count2 = group_counts.get(group2, 0)
                min_count = min(count1, count2)
                #if min_count <= self.min_sample_node * .1:
                    #effective_alpha = self.alpha_merge / 5
                #elif min_count <= self.min_sample_node * .5:
                    #effective_alpha = self.alpha_merge / 10
                #else:
                effective_alpha = self.alpha_merge

                if best_p_value > effective_alpha or np.isclose(best_p_value, effective_alpha, rtol=1e-4, atol=1e-7):
                    for cat in list(groups.keys()):
                        if groups[cat] == group2:
                            groups[cat] = group1
                        elif groups[cat] > group2:
                            groups[cat] -= 1
                    n_groups -= 1
                    changes_made = True

        group_means = {}
        for group in range(n_groups):
            cats_in_group = [cat for cat, g in groups.items() if g == group]
            vals = np.concatenate([category_stats[cat]['values'] for cat in cats_in_group if cat in category_stats])
            if len(vals) > 0:
                group_means[group] = np.mean(vals)

        sorted_groups = sorted(group_means.keys(), key=lambda g: group_means.get(g, 0))
        group_mapping = {old_g: new_g for new_g, old_g in enumerate(sorted_groups)}
        merged_groups = {cat: group_mapping[g] for cat, g in groups.items()}

        group_data = {}
        for group in range(n_groups):
            cats_in_group = [cat for cat, g in merged_groups.items() if g == group]
            vals = [category_stats[cat]['values'] for cat in cats_in_group if cat in category_stats]
            if vals:
                group_data[group] = np.concatenate(vals)

        if len(group_data) > 1:
            anova_data = [vals for vals in group_data.values() if len(vals) > 0]
            if len(anova_data) > 1:
                from statsmodels.formula.api import ols
                from statsmodels.stats.anova import anova_lm
                temp_df = pd.DataFrame({
                    'y': np.concatenate(anova_data),
                    'group': np.concatenate([np.ones(len(vals)) * i for i, vals in enumerate(anova_data)])
                })
                model = ols('y ~ C(group)', data=temp_df).fit()
                anova_result = anova_lm(model)
                f_stat = anova_result['F'][0]
                p_value = anova_result['PR(>F)'][0]
                df1 = len(anova_data) - 1
                df2 = sum(len(g) for g in anova_data) - len(anova_data)
                chi_squared = f_stat * df1
                if self.bonferroni_adjustment:
                    num_tests = len(categories) - 1
                    p_value = min(p_value * num_tests, 1.0)
            else:
                f_stat, chi_squared, p_value = 0, 0, 1.0
                df1, df2 = 0, 0
        else:
            f_stat, chi_squared, p_value = 0, 0, 1.0
            df1, df2 = 0, 0

        return merged_groups, f_stat, chi_squared, p_value, df1, df2

    def _adjust_p_value(self, p_value, n_tests):
        """Ajusta el valor p usando el método de Bonferroni."""
        if self.bonferroni_adjustment:
            return min(p_value * n_tests, 1.0)
        return p_value

    def _find_best_split(self, node):
        """Encuentra la mejor característica para dividir el nodo (de mejor_arbol)."""
        if len(node.data) < self.min_sample_split or node.depth >= self.max_depth:
            return None

        best_feature = None
        best_mapping = None
        best_p_value = 1.0
        best_f_stat = 0
        best_chi_squared = 0
        best_df1 = 0
        best_df2 = 0

        n_tests = len(node.features)

        for feature in node.features:
            unique_values = node.data[feature].unique()
            if len(unique_values) <= 1:
                continue

            mapping, f_stat, chi_squared, p_value, df1, df2 = self._merge_categories(node.data, feature, node.target)
            adjusted_p_value = self._adjust_p_value(p_value, n_tests)

            if adjusted_p_value < self.alpha_split:
                if best_feature is None or adjusted_p_value < best_p_value:
                    best_feature = feature
                    best_mapping = mapping
                    best_p_value = adjusted_p_value
                    best_f_stat = f_stat
                    best_chi_squared = chi_squared
                    best_df1 = df1
                    best_df2 = df2
                elif adjusted_p_value == best_p_value and f_stat > best_f_stat:
                    best_feature = feature
                    best_mapping = mapping
                    best_p_value = adjusted_p_value
                    best_f_stat = f_stat
                    best_chi_squared = chi_squared
                    best_df1 = df1
                    best_df2 = df2

        if best_feature:
            return {
                'feature': best_feature,
                'mapping': best_mapping,
                'chi_squared': best_chi_squared,
                'p_value': best_p_value,
                'f_stat': best_f_stat,
                'df1': best_df1,
                'df2': best_df2
            }
        return None

    def _split_node(self, node):
        """Divide un nodo basado en la mejor característica encontrada (de mejor_arbol)."""
        split_info = self._find_best_split(node)

        if not split_info:
            return

        node.split_feature = split_info['feature']
        node.split_vals = split_info['mapping']
        node.chi_squared = split_info['chi_squared']
        node.p_value = split_info['p_value']
        node.f_stat = split_info['f_stat']
        node.df1 = split_info['df1']
        node.df2 = split_info['df2']

        node.data['_temp_group'] = node.data[node.split_feature].map(node.split_vals)

        valid_children_count = 0
        potential_children = {}
        orphan_candidates = {}

        for group in sorted(node.data['_temp_group'].unique()):
            group_data = node.data[node.data['_temp_group'] == group].drop('_temp_group', axis=1)
            if len(group_data) >= self.min_sample_node:
                valid_children_count += 1
                potential_children[group] = group_data
            else:
                orphan_candidates[group] = group_data

        if valid_children_count < self.min_children:
            node.split_feature = None
            node.split_vals = None
            node.chi_squared = None
            node.p_value = None
            node.f_stat = None
            node.df1 = None
            node.df2 = None
            if '_temp_group' in node.data.columns:
                node.data = node.data.drop('_temp_group', axis=1)
            return

        if self.max_children is not None and valid_children_count > self.max_children:
            group_stats = {}
            for group, group_data in potential_children.items():
                group_stats[group] = {
                    'size': len(group_data),
                    'mean': np.mean(group_data[node.target]),
                    'std': np.std(group_data[node.target]) if len(group_data) > 1 else 0,
                    'data': group_data
                }
            
            groups_to_keep = sorted(group_stats.keys(), 
                                  key=lambda g: (group_stats[g]['size'], -abs(group_stats[g]['mean'] - node.prediction)),
                                  reverse=True)[:self.max_children]
            
            groups_to_merge = [g for g in potential_children.keys() if g not in groups_to_keep]
            
            if groups_to_merge:
                for merge_group in groups_to_merge:
                    merge_mean = group_stats[merge_group]['mean']
                    best_target_group = groups_to_keep[0]
                    min_diff = float('inf')
                    
                    for keep_group in groups_to_keep:
                        diff = abs(merge_mean - group_stats[keep_group]['mean'])
                        if diff < min_diff:
                            min_diff = diff
                            best_target_group = keep_group
                    
                    for cat, group in node.split_vals.items():
                        if group == merge_group:
                            node.split_vals[cat] = best_target_group
                
                node.data['_temp_group'] = node.data[node.split_feature].map(node.split_vals)
                potential_children = {}
                for group in groups_to_keep:
                    group_data = node.data[node.data['_temp_group'] == group].drop('_temp_group', axis=1)
                    potential_children[group] = group_data

        final_children = {}
        for group, group_data in potential_children.items():
            if len(group_data) >= self.min_sample_node:
                final_children[group] = group_data
            else:
                orphan_candidates[group] = group_data

        if orphan_candidates:
            inverse_map = {}
            for cat, group in node.split_vals.items():
                if group not in inverse_map:
                    inverse_map[group] = []
                inverse_map[group].append(cat)
            
            for group, orphan_data in orphan_candidates.items():
                if len(orphan_data) > 0:
                    categories = inverse_map.get(group, [])
                    self.orphan_groups.append({
                        'parent_feature': node.split_feature,
                        'parent_n': len(node.data),
                        'categories': categories,
                        'data': orphan_data,
                        'mean': np.mean(orphan_data[node.target]),
                        'std': np.std(orphan_data[node.target]) if len(orphan_data) > 1 else 0,
                        'count': len(orphan_data)
                    })

        total_in_children = 0
        for group, group_data in final_children.items():
            child_node = CHAIDNode(
                data=group_data,
                target=node.target,
                features=node.features,
                parent=node,
                depth=node.depth + 1
            )
            self.node_counter += 1
            child_node.node_id = self.node_counter
            child_node.percentage = (len(group_data) / self.total_samples) * 100
            total_in_children += len(group_data)
            node.children[group] = child_node
            self._split_node(child_node)

        if '_temp_group' in node.data.columns:
            node.data = node.data.drop('_temp_group', axis=1)

    def fit(self, X, y=None):
        """Ajusta el modelo a los datos (de mejor_arbol)."""
        if isinstance(X, pd.DataFrame) and y is None:
            data = X.copy()
            target_col = self.target
        else:
            data = X.copy()
            data[self.target] = y
            target_col = self.target

        self.feature_names = [col for col in data.columns if col != target_col]
        self.total_samples = len(data)

        # Determinar si el dataset es grande
        self.is_large_dataset = self.total_samples > self.large_dataset_threshold
        print(f"Dataset size: {self.total_samples}, Is large dataset: {self.is_large_dataset}")

        self.root = CHAIDNode(
            data=data,
            target=target_col,
            features=self.feature_names,
            depth=0
        )
        self.node_counter = 0
        self.root.node_id = self.node_counter

        self._split_node(self.root)
        return self

    def predict(self, X):
        """Predice usando el árbol entrenado (de mejor_arbol)."""
        if not self.root:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")

        predictions = np.zeros(len(X))

        for i, row in X.iterrows():
            node = self.root
            while node.split_feature and node.children:
                feature_val = row[node.split_feature]
                if feature_val not in node.split_vals:
                    break
                group = node.split_vals[feature_val]
                if group in node.children:
                    node = node.children[group]
                else:
                    break
            predictions[i - X.index[0]] = node.prediction

        return predictions

    def _create_node_image(self, node):
        """Crea una imagen para el nodo con histograma y estadísticas, ajustando según el tamaño del dataset."""
        bg_color = (240, 248, 255)
        border_color = (70, 130, 180)

        # Ajustar tamaños según si el dataset es grande
        if self.is_large_dataset:
            text_height = 300 
            hist_width, hist_height = 600, 400
            total_width = 600
            font_size = 36
            dpi = 600  # Reducir DPI, ya que el PDF manejará la calidad
        else:
            text_height = 250 
            hist_width, hist_height = 500, 350
            total_width = 500
            font_size = 32
            dpi = 600

        stats_text = (
            f"Nodo {node.node_id}\n"
            f"Media: {node.prediction:.2f}\n"
            f"Desv. Est.: {node.std:.2f}\n"
            f"n: {node.n}\n"
            f"%: {node.percentage:.1f}%\n"
            f"Previsto: {node.prediction:.2f}"
        )

        text_img = Image.new('RGB', (total_width, text_height), color=bg_color)
        draw = ImageDraw.Draw(text_img)
        draw.rectangle([(0, 0), (total_width - 1, text_height - 1)], outline=border_color, width=2)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default(font_size)

        draw.text((10, 10), stats_text, fill='black', font=font)

        hist_buf = node._plot_histogram(is_large_dataset=self.is_large_dataset)
        hist_img = Image.open(hist_buf)
        hist_img = hist_img.resize((hist_width, hist_height), Image.Resampling.LANCZOS)

        total_height = text_height + hist_height
        combined_img = Image.new('RGB', (total_width, total_height), color=bg_color)
        combined_img.paste(text_img, (0, 0))
        combined_img.paste(hist_img, (0, text_height))

        draw = ImageDraw.Draw(combined_img)
        draw.rectangle([(0, 0), (total_width - 1, total_height - 1)], outline=border_color, width=2)

        result_buf = io.BytesIO()
        combined_img.save(result_buf, format='PNG', dpi=(dpi, dpi))
        result_buf.seek(0)
        return result_buf

    def visualize(self, output_format='pdf'):
        """
        Genera una visualización del árbol CHAID.
        Para bases de datos grandes, genera un PDF optimizado para Adobe Acrobat.
        """
        dot = Digraph(comment='CHAID Tree')
        dot.attr(rankdir='TB')
        dot.attr('graph', splines='line', nodesep='0.8', ranksep='1.2')  # Cambiar 'ortho' por 'line' para mejor compatibilidad
        dot.attr('node', shape='none', margin='0', fontname='Helvetica', fontsize='12')
        dot.attr('edge', fontname='Helvetica', fontsize='12')

        # Ajustar DPI para PDF (reducido para evitar problemas de tamaño)
        dot.attr(dpi='300')  # Reducir DPI para mantener el archivo manejable

        tmpdirname = tempfile.mkdtemp()
        node_images = {}

        def add_node(node, parent_id=None):
            if node is None:
                return

            node_id = f"node_{node.node_id}"
            node_img_buf = self._create_node_image(node)
            node_img_path = os.path.join(tmpdirname, f"{node_id}.png")
            with open(node_img_path, 'wb') as f:
                f.write(node_img_buf.getvalue())
            node_images[node_id] = node_img_path

            # Ajustar tamaños de nodos según si es raíz o hijo
            if self.is_large_dataset:
                root_width, root_height = '4', '4.5'
                child_width, child_height = '3.5', '4'
                var_fontsize, div_fontsize = '16', '14'
            else:
                root_width, root_height = '3.5', '4'
                child_width, child_height = '3', '3.5'
                var_fontsize, div_fontsize = '14', '12'

            if node.node_id == 0:
                dot.node(node_id, label='', image=node_img_path, imagescale='true', fixedsize='true', width=root_width, height=root_height)
            else:
                dot.node(node_id, label='', image=node_img_path, imagescale='true', fixedsize='true', width=child_width, height=child_height)

            if node.children and node.split_feature:
                intermediate_var_id = f"int_var_{node.node_id}"
                dot.node(intermediate_var_id, label=node.split_feature, shape='ellipse', style='filled', fillcolor='lightblue', fontsize=var_fontsize, width='1.5', height='0.5')
                stats_label = f"P={node.p_value:.3f}, Chi2={node.chi_squared:.1f}, df={node.df1}" if node.p_value is not None and node.chi_squared is not None else ""
                dot.edge(node_id, intermediate_var_id, label=stats_label, fontsize='12', penwidth='1.2')

                inverse_map = {}
                for cat, g in node.split_vals.items():
                    if g not in inverse_map:
                        inverse_map[g] = []
                    inverse_map[g].append(cat)

                for group, child in node.children.items():
                    categories = inverse_map.get(group, [])
                    categories_str = ";".join([str(c) for c in categories])
                    if len(categories_str) > 25:
                        categories_str = categories_str[:22] + "..."

                    intermediate_div_id = f"int_div_{node.node_id}_{group}"
                    dot.node(intermediate_div_id, label=categories_str, shape='ellipse', style='filled', fillcolor='lightcyan', fontsize=div_fontsize, width='1', height='0.3')
                    dot.edge(intermediate_var_id, intermediate_div_id, fontsize='12', penwidth='1')

                    child_id = f"node_{child.node_id}"
                    child_img_buf = self._create_node_image(child)
                    child_img_path = os.path.join(tmpdirname, f"{child_id}.png")
                    with open(child_img_path, 'wb') as f:
                        f.write(child_img_buf.getvalue())
                    node_images[child_id] = child_img_path
                    dot.node(child_id, label='', image=child_img_path, imagescale='true', fixedsize='true', width=child_width, height=child_height)
                    dot.edge(intermediate_div_id, child_id, fontsize='12', penwidth='1')

                    add_node(child, node_id)

        try:
            if self.root:
                add_node(self.root)

                if self.orphan_groups:
                    orphan_cluster = Digraph(name='cluster_orphans')
                    orphan_cluster.attr(label='Grupos Huérfanos', style='dashed', color='gray', fontsize='24', labelloc='t', nodesep='0.5', ranksep='0.5')
                    for i, orphan in enumerate(self.orphan_groups):
                        categories_str = ", ".join([str(c) for c in orphan['categories']])
                        if len(categories_str) > 25:
                            categories_str = categories_str[:22] + "..."
                        orphan_id = f"orphan_{i}"
                        stats_text = (
                            f"Feature: {orphan['parent_feature']}\n"
                            f"n padre: {orphan['parent_n']}\n"
                            f"Categori as: {categories_str}\n"
                            f"Media: {orphan['mean']:.2f}\n"
                            f"Desv. Est.: {orphan['std']:.2f}\n"
                            f"n: {orphan['count']}"
                        )
                        text_img = Image.new('RGB', (700, 500), color=(240, 248, 255))
                        draw = ImageDraw.Draw(text_img)
                        draw.rectangle([(0, 0), (799, 599)], outline=(70, 130, 180), width=2)
                        try:
                            font = ImageFont.truetype("DejaVuSans.ttf", 50)
                        except:
                            font = ImageFont.load_default(50)
                        draw.text((15, 15), stats_text, fill='black', font=font)
                        orphan_img_path = os.path.join(tmpdirname, f"{orphan_id}.png")
                        text_img.save(orphan_img_path, format='PNG')
                        orphan_cluster.node(orphan_id, label='', image=orphan_img_path, shape='none')
                    dot.subgraph(orphan_cluster)

                dot.engine = 'dot'
                output_path = os.path.join(tmpdirname, "chaid_tree")

                # Generar el PDF optimizado
                dot.format = 'pdf'
                dot.attr(size='8.5,11')  # Tamaño de página estándar (Carta: 8.5x11 pulgadas)
                dot.attr(ratio='compress')  # Comprimir para ajustar contenido
                dot.render(filename=output_path, cleanup=True, quiet=True)
                pdf_path = f"{output_path}.pdf"
                with open(pdf_path, 'rb') as f:
                    pdf_data = io.BytesIO(f.read())
                    pdf_data.seek(0)

                # Generar vista previa en PNG (opcional)
                dot.format = 'png'
                dot.attr(dpi='100')  # DPI bajo para vista previa
                dot.render(filename=output_path, cleanup=True, quiet=True)
                png_path = f"{output_path}.png"
                with Image.open(png_path) as img:
                    max_size = (1280, 720)  # Tamaño razonable para vista previa
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    preview_buf = io.BytesIO()
                    img.save(preview_buf, format='PNG')
                    preview_buf.seek(0)
                    png_data = preview_buf

                return pdf_data, png_data

            else:
                raise ValueError("El árbol no ha sido construido. Llama a fit() primero.")
        finally:
            shutil.rmtree(tmpdirname, ignore_errors=True)

def chaid_tree(df, target, features=None, alpha_merge=0.05, alpha_split=0.05, max_depth=10,
               min_sample_node=30, min_sample_split=50, min_children=2, max_children=None, 
               max_iterations=10000, bonferroni_adjustment=True, large_dataset_threshold=10000):
    """Función de ayuda para crear y ajustar un árbol CHAID."""
    if features is None:
        features = [col for col in df.columns if col != target]

    subset_df = df[features + [target]].copy()

    model = CHAID(
        alpha_merge=alpha_merge,
        alpha_split=alpha_split,
        max_depth=max_depth,
        min_sample_node=min_sample_node,
        min_sample_split=min_sample_split,
        min_children=min_children,
        max_children=max_children,
        max_iterations=max_iterations,
        bonferroni_adjustment=bonferroni_adjustment,
        large_dataset_threshold=large_dataset_threshold
    )
    print(max_depth)
    model.target = target
    model.fit(subset_df)

    return model