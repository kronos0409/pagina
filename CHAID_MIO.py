import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import networkx as nx
import warnings
from PIL import Image, ImageDraw, ImageFont
import io
import tempfile
import os
import shutil
import logging
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import levene
from joblib import Parallel, delayed
# Configurar el logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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
        """Genera un histograma para el nodo."""
        if is_large_dataset:
            figsize = (14, 6)
            dpi = 600
            fontsize = 36
        else:
            figsize = (12, 4)
            dpi = 600
            fontsize = 32

        fig, ax1 = plt.subplots(figsize=figsize)

        counts, bins, _ = ax1.hist(
            self.data[self.target], 
            bins=10, 
            color='royalblue', 
            edgecolor='black', 
            alpha=0.7
        )

        ax1.set_xlabel(self.target, fontsize=fontsize)
        ax1.set_ylabel('Frecuencia', fontsize=fontsize, color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue', labelsize=fontsize-4)

        max_count = int(max(counts))
        if max_count > 0:
            step = max(1, max_count // 4)
            ax1.set_yticks(range(0, max_count + 1, step))
        ax1.tick_params(axis='x', labelsize=fontsize-4)
        ax1.grid(True, alpha=0.3)

        total_count = sum(counts)
        if total_count > 0:
            relative_freq = counts / total_count
            cumulative_relative_freq = np.cumsum(relative_freq)

            ax2 = ax1.twinx()
            ax2.plot(
                bins[:-1] + np.diff(bins) / 2,
                cumulative_relative_freq, 
                color='red', 
                linestyle='-', 
                marker='o', 
                markersize=5, 
                label='Frecuencia Relativa Acumulada'
            )
            ax2.set_ylabel('Frecuencia Relativa Acumulada', fontsize=10, color='red')
            ax2.tick_params(axis='y', labelcolor='red', labelsize=fontsize-4)
            ax2.set_ylim(0, 1)
            ax2.set_yticks(np.linspace(0, 1, 6))

        plt.tight_layout()

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
    def __init__(self, alpha_merge=0.05, alpha_split=0.05, max_depth=10, min_sample_node=30,
                 min_sample_split=50, min_children=2, max_children=None, max_iterations=1000,  # Reducido de 10000 a 1000
                 bonferroni_adjustment=True, large_dataset_threshold=10000, porcentaje=1.00):
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
        self.porcentaje = porcentaje

    def get_terminal_nodes_data(self, original_df):
        """Recolecta los datos de los nodos terminales y huérfanos con sus etiquetas."""
        if not self.root:
            raise ValueError("El modelo debe ser entrenado antes de obtener los nodos terminales.")

        terminal_data = []
        visited_indices = set()

        def traverse_node(node, df_subset):
            if not node.children:
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

        traverse_node(self.root, original_df)

        for orphan in self.orphan_groups:
            orphan_data = orphan['data'].copy()
            orphan_data['NODO'] = 'huérfanos'
            terminal_data.append(orphan_data)
            visited_indices.update(orphan_data.index)

        if terminal_data:
            result_df = pd.concat(terminal_data, axis=0, ignore_index=False)
            missing_indices = original_df.index.difference(visited_indices)
            if not missing_indices.empty:
                missing_data = original_df.loc[missing_indices].copy()
                missing_data['NODO'] = 'sin_clasificar'
                result_df = pd.concat([result_df, missing_data], axis=0, ignore_index=False)
            return result_df.sort_index()
        else:
            result_df = original_df.copy()
            result_df['NODO'] = 'sin_nodos_terminales'
            return result_df

    def _merge_categories(self, data, feature, target, features):
        categories = data[feature].unique()
        if len(categories) <= 1:
            return {cat: 0 for cat in categories}, 0, 0, 0, 1.0, 1.0, 0, 0, True

        grouped = data.groupby(feature)[target]
        category_stats = {cat: {'values': grouped.get_group(cat).values} for cat in categories}

        initial_p_value = 1.0
        initial_f_stat = 0.0
        formula = f"{target} ~ " + " + ".join([f"C({f})" for f in features])
        try:
            model = ols(formula, data=data).fit()
            anova_result = anova_lm(model, typ=3)
            initial_f_stat = anova_result.loc[f"C({feature})", "F"]
            initial_p_value = anova_result.loc[f"C({feature})", "PR(>F)"]
        except Exception as e:
            logger.error(f"Error al calcular ANOVA inicial para '{feature}': {str(e)}")

        for cat in categories:
            vals = category_stats[cat]['values']
            category_stats[cat]['mean'] = np.mean(vals)
            category_stats[cat]['std'] = np.std(vals) if len(vals) > 1 else 0
            category_stats[cat]['count'] = len(vals)

        sorted_cats = sorted(categories, key=lambda x: (
            category_stats[x]['mean'], category_stats[x]['std'], category_stats[x]['count']
        ))
        groups = {cat: i for i, cat in enumerate(sorted_cats)}
        n_groups = len(sorted_cats)

        iteration = 0
        changes_made = True
        group_data = {i: np.concatenate([category_stats[cat]['values'] for cat in sorted_cats if groups[cat] == i]) 
                     for i in range(n_groups)}
        min_groups = 4

        while changes_made and n_groups > min_groups and iteration < self.max_iterations:
            changes_made = False
            iteration += 1
            p_values = {}
            for i in range(n_groups - 1):
                vals_group1 = group_data[i]
                vals_group2 = group_data[i + 1]
                if len(vals_group1) > 0 and len(vals_group2) > 0:
                    f_stat, p_value = stats.f_oneway(vals_group1, vals_group2)
                    p_values[(i, i + 1)] = p_value if not np.isnan(p_value) else 0

            if p_values:
                best_pair, best_p_value = max(p_values.items(), key=lambda x: x[1])
                if best_p_value > self.alpha_merge:
                    group1, group2 = best_pair
                    group_data[group1] = np.concatenate([group_data[group1], group_data[group2]])
                    del group_data[group2]
                    for cat in groups:
                        if groups[cat] == group2:
                            groups[cat] = group1
                        elif groups[cat] > group2:
                            groups[cat] -= 1
                    new_group_data = {}
                    for new_idx, old_idx in enumerate(sorted(group_data.keys())):
                        new_group_data[new_idx] = group_data[old_idx]
                        for cat in groups:
                            if groups[cat] == old_idx:
                                groups[cat] = new_idx
                    group_data = new_group_data
                    n_groups -= 1
                    changes_made = True

        if n_groups < min_groups and len(categories) >= min_groups:
            groups = {cat: i % min_groups for i, cat in enumerate(categories)}
            n_groups = min_groups
            group_data = {i: np.concatenate([category_stats[cat]['values'] for cat in categories if groups[cat] == i]) 
                         for i in range(n_groups)}

        group_means = {g: np.mean(group_data[g]) for g in group_data}
        sorted_groups = sorted(group_means.keys(), key=lambda g: group_means[g])
        group_mapping = {old_g: new_g for new_g, old_g in enumerate(sorted_groups)}
        merged_groups = {cat: group_mapping[g] for cat, g in groups.items()}

        if n_groups > 1 and (anova_data := [group_data[g] for g in group_data if len(group_data[g]) > 0]):
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
                p_value = min(p_value * (len(categories) - 1), 1.0)
        else:
            f_stat, chi_squared, p_value = 0, 0, 1.0
            df1, df2 = 0, 0

        all_merged = n_groups <= 1
        return merged_groups, initial_f_stat, f_stat, chi_squared, p_value, initial_p_value, df1, df2, all_merged

    def _adjust_p_value(self, p_value, n_tests):
        """Ajusta el valor p usando el método de Bonferroni."""
        if self.bonferroni_adjustment:
            return min(p_value * n_tests, 1.0)
        return p_value

    

    def _find_best_split(self, node):
        if len(node.data) < self.min_sample_split or node.depth >= self.max_depth:
            return None

        n_tests = len(node.features)
        feature_grouped = {feat: node.data.groupby(feat)[node.target] for feat in node.features}

        def evaluate_feature(feature):
            if len(feature_grouped[feature].groups) <= 1:
                return None
            mapping, initial_f_stat, f_stat, chi_squared, p_value, initial_p_value, df1, df2, all_merged = self._merge_categories(
                node.data, feature, node.target, node.features
            )
            if all_merged:
                return None
            adjusted_p_value = self._adjust_p_value(p_value, n_tests)
            if adjusted_p_value < self.alpha_split:
                return {
                    'feature': feature,
                    'mapping': mapping,
                    'chi_squared': chi_squared,
                    'p_value': adjusted_p_value,
                    'f_stat': f_stat,
                    'initial_f_stat': initial_f_stat,
                    'initial_p_value': initial_p_value,
                    'df1': df1,
                    'df2': df2
                }
            return None

        candidates = Parallel(n_jobs=-1, backend='loky')(
            delayed(evaluate_feature)(feature) for feature in node.features
        )
        candidates = [c for c in candidates if c is not None]

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x['initial_p_value'], -x['f_stat']))
        for candidate in candidates:
            feature = candidate['feature']
            mapping = candidate['mapping']
            node.data['_temp_group'] = node.data[feature].map(mapping)
            groups = sorted(node.data['_temp_group'].unique())
            valid_groups = sum(1 for group in groups if len(node.data[node.data['_temp_group'] == group]) >= self.min_sample_node)
            if valid_groups < self.min_children:
                continue
            total_size = len(node.data)
            max_group_size = max(len(node.data[node.data['_temp_group'] == group]) for group in groups)
            if (max_group_size / total_size) > self.porcentaje:
                continue
            node.data.drop('_temp_group', axis=1, inplace=True)
            return {k: candidate[k] for k in ['feature', 'mapping', 'chi_squared', 'p_value', 'f_stat', 'initial_f_stat', 'df1', 'df2']}

        if '_temp_group' in node.data.columns:
            node.data.drop('_temp_group', axis=1, inplace=True)
        return None
    
    def _split_node(self, node):
        """Divide un nodo basado en la mejor característica encontrada."""
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
        """Ajusta el modelo a los datos."""
        if isinstance(X, pd.DataFrame) and y is None:
            data = X.copy()
            target_col = self.target
        else:
            data = X.copy()
            data[self.target] = y
            target_col = self.target

        self.feature_names = [col for col in data.columns if col != target_col]
        self.total_samples = len(data)

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
        """Predice usando el árbol entrenado."""
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
        """Crea una imagen para el nodo con histograma y estadísticas."""
        bg_color = (240, 248, 255)
        border_color = (70, 130, 180)

        if self.is_large_dataset:
            total_width = 1680
            total_height = 1200
            text_height = int(total_height * 0.5)   # Aumentado de 500 a 550 para más espacio
            hist_height = total_height - text_height  # Ajustado para mantener total_height
            hist_width = total_width
            font_size = 67
            dpi = 650
        else:
            total_width = 1440
            total_height = 1000
            text_height = int(total_height * 0.5)  # Aumentado de 400 a 450 para más espacio
            hist_height = total_height - text_height  # Ajustado para mantener total_height
            hist_width = total_width
            font_size = 67
            dpi = 650

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
        draw.rectangle([(0, 0), (total_width-1, text_height-1)], outline=(70, 130, 180), width=2)

        try:
            font = ImageFont.truetype("ARLRDBD.TTF", font_size)
        except:
            print("No se encontró ariblk.ttf, usando fuente predeterminada.")
            font = ImageFont.load_default(font_size)

        draw.text((25, 25), stats_text, fill='black', font=font)

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
        combined_img.close()
        result_buf.seek(0)
        return result_buf


    def visualize(self, output_format='pdf'):
        """Genera una visualización del árbol CHAID usando NetworkX y matplotlib."""
        # Crear un grafo dirigido
        G = nx.DiGraph()

        # Diccionario para almacenar las imágenes de los nodos
        node_images = {}
        stats_labels = {}  # Para estadísticas (P, Chi2, df)
        edge_labels = {}   # Para etiquetas personalizadas en las aristas (como "capa 2")
        category_num_lines = {}  # Diccionario para almacenar el número de líneas de cada nodo de tipo 'category'

        # Función para agregar nodos y aristas al grafo
        def add_node_to_graph(node, parent_id=None, level=0):
            if node is None:
                return

            node_id = f"node_{node.node_id}"
            G.add_node(node_id, type='data', level=level)

            # Generar la imagen del nodo (con histograma)
            node_img_buf = self._create_node_image(node)
            node_images[node_id] = Image.open(node_img_buf)

            if parent_id:
                G.add_edge(parent_id, node_id)

            if node.children and node.split_feature:
                # Nodo para la variable de división (como LITH o USP)
                var_id = f"int_var_{node.node_id}"
                G.add_node(var_id, type='variable', label=node.split_feature, level=level + 1)
                G.add_edge(node_id, var_id)
                # Almacenar estadísticas para la arista
                stats_label = f"P={node.p_value:.3e}, df={node.df1}" if node.p_value is not None and node.chi_squared is not None else ""
                stats_labels[(node_id, var_id)] = stats_label

                # Mapa inverso para las categorías
                inverse_map = {}
                for cat, g in node.split_vals.items():
                    if g not in inverse_map:
                        inverse_map[g] = []
                    inverse_map[g].append(cat)

                # Agregar nodos para las categorías (LC, OX, SULF, etc.)
                for group, child in node.children.items():
                    categories = inverse_map.get(group, [])
                    # Dividir las categorías en varias líneas
                    categories_list = [str(c) for c in categories]
                    max_chars_per_line = 20  # Límite de caracteres por línea
                    lines = []
                    current_line = []
                    current_length = 0

                    for cat in categories_list:
                        cat_length = len(cat) + 1  # +1 por el ";"
                        if current_length + cat_length > max_chars_per_line and current_line:
                            lines.append(";".join(current_line))
                            current_line = [cat]
                            current_length = cat_length
                        else:
                            current_line.append(cat)
                            current_length += cat_length
                    if current_line:
                        lines.append(";".join(current_line))

                    # Depuración para verificar las líneas generadas
                    logger.debug(f"Categorías para grupo {group}: {categories_list}")
                    logger.debug(f"Líneas generadas: {lines}")
                    categories_str = "\n".join(lines)  # Unir las líneas con saltos de línea
                    logger.debug(f"categories_str: {categories_str!r}")  # Mostrar la representación exacta

                    # Calcular el número de líneas y almacenarlo
                    num_lines = len(lines)  # Número de líneas es el número de elementos en 'lines'
                    logger.debug(f"num_lines calculado: {num_lines}")

                    div_id = f"int_div_{node.node_id}_{group}"
                    G.add_node(div_id, type='category', label=categories_str, level=level + 2)
                    G.add_edge(var_id, div_id)
                    # Almacenar el número de líneas en el diccionario
                    category_num_lines[div_id] = num_lines
                    # Agregar una etiqueta personalizada a la arista (como "capa 2")
                    edge_labels[(var_id, div_id)] = f"capa {group}"

                    # Agregar el nodo hijo
                    add_node_to_graph(child, div_id, level + 3)

        # Construir el grafo
        if not self.root:
            raise ValueError("El árbol no ha sido construido. Llama a fit() primero.")
        add_node_to_graph(self.root)

        # Calcular el número de hojas (nodos terminales) por subárbol
        def count_leaves(node_id):
            children = list(G.successors(node_id))
            if not children:
                return 1  # Es un nodo hoja
            total_leaves = 0
            for child in children:
                if G.nodes[child]['type'] == 'data':
                    total_leaves += count_leaves(child)
                else:
                    total_leaves += count_leaves(child)
            # Asegurarse de que total_leaves sea al menos 1 para evitar divisiones por cero
            return max(total_leaves, 1)

        # Asignar el número de hojas a cada nodo
        for node in G.nodes:
            G.nodes[node]['leaves'] = count_leaves(node)

        # Calcular posiciones dinámicamente
        pos = {}
        level_heights = {}  # Para rastrear los nodos en cada nivel
        base_node_width = 6.0  # Ancho base por hoja (ajustable)
        base_level_spacing = 3.0  # Espaciado vertical base entre niveles

        # Usar un espaciado uniforme para todos los niveles
        level_spacing = base_level_spacing

        def assign_positions(node_id, x=0, level=0):
            if level not in level_heights:
                level_heights[level] = []
            level_heights[level].append((node_id, x))

            # Asignar posición vertical basada en el nivel con espaciado uniforme
            y_offset = 1.5 if level == 0 else 0
            y = -level * level_spacing + y_offset
            pos[node_id] = (x, y)

            # Procesar hijos
            children = list(G.successors(node_id))
            if children:
                # Calcular el ancho total de los hijos basado en el número de hojas
                total_leaves = sum(G.nodes[child]['leaves'] for child in children)
                if total_leaves == 0:
                    total_leaves = len(children)  # Fallback si no hay hojas
                total_width = total_leaves * base_node_width
                start_x = x - (total_width / 2)  # Centrar el subárbol respecto al padre

                # Asignar posiciones a los hijos
                current_x = start_x
                for child in children:
                    child_leaves = G.nodes[child]['leaves']
                    if child_leaves == 0:
                        child_leaves = 1  # Fallback para evitar división por cero
                    # El ancho asignado a este hijo es proporcional a su número de hojas
                    child_width = (child_leaves / total_leaves) * total_width
                    child_x = current_x + (child_width / 2)  # Posicionar el hijo en el centro de su espacio asignado
                    assign_positions(child, child_x, G.nodes[child]['level'])
                    current_x += child_width

        # Asignar posiciones iniciales
        root_id = f"node_{self.root.node_id}"
        assign_positions(root_id)

        # Crear la figura con tamaño dinámico
        max_leaves_in_level = 0
        for level, nodes in level_heights.items():
            level_leaves = sum(G.nodes[node_id]['leaves'] for node_id, _ in nodes if G.nodes[node_id]['type'] == 'data')
            max_leaves_in_level = max(max_leaves_in_level, level_leaves)
        fig_width = max(30, max_leaves_in_level * 4.0)
        fig_height = max(15, len(level_heights) * level_spacing + 2)  # Añadir un margen adicional
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_axis_off()

        # Dibujar las imágenes de los nodos y etiquetas
        plt.title("Árbol CHAID", fontsize=16, pad=20)
        for node_id, (x, y) in pos.items():
            if G.nodes[node_id]['type'] == 'data':
                img = node_images[node_id]
                img_array = np.array(img)
                # Ajustar el tamaño de la imagen
                extent = [x - 2, x + 3, y - 2.0, y + 2.0]
                ax.imshow(img_array, extent=extent, aspect='auto')
            else:
                # Dibujar etiquetas para variables y categorías
                label = G.nodes[node_id].get('label', '')
                if G.nodes[node_id]['type'] == 'variable':
                    ax.text(x, y, label, ha='center', va='center', fontsize=16, 
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='square', linestyle='dashed', pad=0.5))
                else:  # Categoría
                    ax.text(x, y, label, ha='center', va='center', fontsize=14, 
                            bbox=dict(facecolor='none', edgecolor='black', boxstyle='square', linestyle='dashed', pad=0.5))

        # Dibujar las aristas sin flechas y ajustando dinámicamente las líneas verticales
        for edge in G.edges():
            start_id, end_id = edge
            x1, y1 = pos[start_id]
            x2, y2 = pos[end_id]

            # Determinar el tipo de conexión
            if G.nodes[end_id]['type'] == 'variable':
                y1_adjusted = y1 - 2.0  # Ajuste para conectar con el nodo de variable
                label_height = 0.6
                y2_adjusted = y2 + label_height
                ax.plot([x1, x1], [y1_adjusted, y2_adjusted - 0.25], 'k-', linewidth=1.5)
            elif G.nodes[end_id]['type'] == 'category':
            # Usar valores fijos para asegurar simetría
                base_label_height = 0.6
                base_vertical_adjustment = 0.1  # Reducido para evitar superposiciones

                # Obtener el número de líneas del nodo de categoría (end_id)
                num_lines = category_num_lines.get(end_id, 1)
                logger.debug(f"num_lines para end_id {end_id} (category): {num_lines}")
                print(f"num_lines para end_id {end_id} (category): {num_lines}")
                # Calcular la altura del cuadro de la categoría
                text_height_per_line = 0.2  # Aproximación de la altura por línea
                pad_height = 0.5  # pad=0.5 en el bbox
                category_box_height = (num_lines * text_height_per_line) + (2 * pad_height)
                y2_adjusted = y2 + (category_box_height / 2)  # Borde superior del cuadro

                # Ajustar label_height y vertical_adjustment según el número de líneas
                if num_lines >= 4:
                    label_height = base_label_height + 0.4 * (num_lines - 1)
                    vertical_adjustment = base_vertical_adjustment + 0.03 * (num_lines - 1)
                elif num_lines == 3:
                    label_height = base_label_height + 0.4 * (num_lines - 1)
                    vertical_adjustment = base_vertical_adjustment + 0.07 * (num_lines - 1)  # Reducido
                    multiplicador = 1.55
                elif num_lines == 2:
                    label_height = base_label_height + 0.2 * (num_lines - 1)
                    vertical_adjustment = base_vertical_adjustment + 0.2 * (num_lines - 1)
                    if num_lines>= 2:
                        multiplicador = 1.55
                else:
                    label_height = base_label_height + 0.2 * (num_lines - 1)
                    vertical_adjustment = base_vertical_adjustment + .23  # Reducido
                    multiplicador = 0.9
                y1_adjusted = y1 - label_height
                # Calcular branch_y de manera simétrica
                branch_y = (y1 + y2) / 2

                # Dibujar las líneas
                ax.plot([x1, x1], [y1_adjusted + vertical_adjustment*multiplicador, branch_y], 'k-', linewidth=1.5)  # Vertical (desde el nodo de variable)
                ax.plot([x1, x2], [branch_y, branch_y], 'k-', linewidth=1.5)  # Horizontal
                ax.plot([x2, x2], [branch_y, y2_adjusted-vertical_adjustment], 'k-', linewidth=1.5)  # Vertical hacia el nodo de categoría
            else:  # Conexión entre categoría y nodo de datos
                # Verificar que start_id es el nodo de tipo 'category'
                if G.nodes[start_id]['type'] != 'category':
                    logger.error(f"Error: start_id {start_id} no es de tipo 'category', es {G.nodes[start_id]['type']}")
                    continue
                if G.nodes[end_id]['type'] != 'data':
                    logger.error(f"Error: end_id {end_id} no es de tipo 'data', es {G.nodes[end_id]['type']}")
                    continue

                # Obtener el número de líneas del diccionario
                num_lines = category_num_lines.get(start_id, 1)
                # Ajustar dinámicamente label_height y vertical_adjustment según el número de líneas
                base_label_height = 0.6
                base_vertical_adjustment = 0.1  # Reducido para evitar superposiciones
                if num_lines >= 4:
                    label_height = base_label_height + 0.4 * (num_lines - 1)
                    vertical_adjustment_up = base_vertical_adjustment + 0.085 * (num_lines - 1)  # Reducido
                    vertical_adjustment_down = base_vertical_adjustment + 0.09  * (num_lines - 1)
                elif num_lines == 3:
                    label_height = base_label_height + 0.4 * (num_lines - 1)
                    vertical_adjustment_up = base_vertical_adjustment + 0.13  * (num_lines - 1)
                    vertical_adjustment_down = base_vertical_adjustment + .08  * (num_lines - 1)  # Reducido
                elif num_lines == 2:
                    label_height = base_label_height + 0.2 * (num_lines - 1)
                    vertical_adjustment_up = base_vertical_adjustment + 0.068 * (num_lines - 1)  # Reducido
                    vertical_adjustment_down = base_vertical_adjustment + 0.13  * (num_lines - 1)
                else:
                    label_height = base_label_height
                    vertical_adjustment_up = base_vertical_adjustment + 0.03  # Reducido
                    vertical_adjustment_down = base_vertical_adjustment + 0.15 # Reducido
                y1_adjusted = y1 - label_height
                # Calcular branch_y de manera simétrica
                branch_y = (y1 + y2) / 2
                # Ajustar y2_adjusted para que la línea se detenga exactamente en el borde superior del cuadro del nodo de datos
                y2_adjusted = y2 + 2.0  # Borde superior del cuadro del nodo de datos

                # Dibujar las líneas
                ax.plot([x1, x1], [y1_adjusted + 2.3*vertical_adjustment_up, branch_y+vertical_adjustment_down*2], 'k-', linewidth=1.5)
                ax.plot([x1, x2], [branch_y, branch_y], 'k-', linewidth=1.5)
                #ax.plot([x2, x2], [branch_y, y2_adjusted+vertical_adjustment], 'k-', linewidth=1.5)  # Vertical hacia el nodo de datos

            # Dibujar etiquetas de estadísticas
            if (start_id, end_id) in stats_labels:
                label = stats_labels[(start_id, end_id)]
                mid_x = x1
                mid_y = (y1_adjusted + y2_adjusted) / 2
                ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8, color='blue')
    
        # Dibujar los grupos huérfanos
        if self.orphan_groups:
            max_x = max(x for x, y in pos.values()) + 6
            y_start = max(y for x, y in pos.values()) + 1

            # Nuevas dimensiones para los nodos huérfanos
            orphan_width = 1700  # Aumentado de 700 a 900
            orphan_height = 1000  # Aumentado de 500 a 650
            font_size = 92     # Aumentado de 50 a 60 para mantener proporción

            # Calcular el nuevo extent para mantener proporciones
            # Proporción original de la imagen: 700/500 = 1.4
            # Nuevo extent: queremos que el ancho sea 1.5 unidades (en lugar de 1.0) para que se vea más grande
            extent_width = 2.88  # Aumentado de 1.0 (0.5 + 0.5)
            extent_height = 3.2  # Mantener proporción (650/900 ≈ 0.722)
            # Nuevo espaciado vertical para evitar superposiciones
            image_ratio = orphan_height / orphan_width  # 1000 / 1200 ≈ 0.833
            extent_width = extent_height / image_ratio
            vertical_spacing = extent_height + 0.5  # Aumentado de 2.0 para reflejar la nueva altura

            for i, orphan in enumerate(self.orphan_groups):
                categories_str = ", ".join([str(c) for c in orphan['categories']])
                if len(categories_str) > 25:
                    categories_str = categories_str[:22] + "..."
                stats_text = (
                    f"Feature: {orphan['parent_feature']}\n"
                    f"n padre: {orphan['parent_n']}\n"
                    f"Categorías: {categories_str}\n"
                    f"Media: {orphan['mean']:.2f}\n"
                    f"Desv. Est.: {orphan['std']:.2f}\n"
                    f"n: {orphan['count']}"
                )
                text_img = Image.new('RGB', (orphan_width, orphan_height), color=(240, 248, 255))
                draw = ImageDraw.Draw(text_img)
                draw.rectangle([(0, 0), (orphan_width-1, orphan_height-1)], outline=(70, 130, 180), width=2)
                try:
                    font = ImageFont.truetype("ARLRDBD.TTF", font_size)
                except:
                    font = ImageFont.load_default(font_size)
                draw.text((15, 15), stats_text, fill='black', font=font)
                img_array = np.array(text_img)
                y_pos = y_start - i * vertical_spacing
                # Ajustar el extent para el nuevo tamaño
                extent = [max_x - extent_width/2, max_x + extent_width/2, y_pos - extent_height/2, y_pos + extent_height/2]
                ax.imshow(img_array, extent=extent, aspect='auto')
                if i == 0:
                    # Ajustar la posición del título "Grupos Huérfanos"
                    ax.text(max_x, y_pos + extent_height/2 + 0.5, "Grupos Huérfanos", ha='center', va='bottom', fontsize=12)

        # Ajustar los límites de la figura
        all_x = [x for x, y in pos.values()]
        all_y = [y for x, y in pos.values()]
        if self.orphan_groups:
            all_x.append(max_x + extent_width/2 + 1)
            all_y.append(y_start - len(self.orphan_groups) * vertical_spacing - extent_height/2 - 1)
        ax.set_xlim(min(all_x) - 2, max(all_x) + 2)
        ax.set_ylim(min(all_y) - 2, max(all_y) + 4)
    
        # Guardar la figura como PDF
        tmpdirname = tempfile.mkdtemp()
        output_path = os.path.join(tmpdirname, "chaid_tree")

        plt.savefig(f"{output_path}.pdf", format='pdf', bbox_inches='tight', dpi=300)
        with open(f"{output_path}.pdf", 'rb') as f:
            pdf_data = io.BytesIO(f.read())
            pdf_data.seek(0)

        plt.close()

        # Limpiar archivos temporales
        shutil.rmtree(tmpdirname, ignore_errors=True)

        return pdf_data
    
def chaid_tree(df, target, features=None, alpha_merge=0.05, alpha_split=0.05, max_depth=10,
               min_sample_node=30, min_sample_split=50, min_children=2, max_children=None, 
               max_iterations=10000, bonferroni_adjustment=True, large_dataset_threshold=10000, porcentaje=1.0):
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
        large_dataset_threshold=large_dataset_threshold,
        porcentaje=porcentaje
    )
    model.target = target
    model.fit(subset_df)

    return model
