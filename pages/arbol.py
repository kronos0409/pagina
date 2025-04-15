import streamlit as st
import pandas as pd
import CHAID_MIO as CH
import io

def Cambiar_tipo(variable, tipo, DF):
    if tipo == "Continua":
        DF[variable] = DF[variable].astype(float) 
    elif tipo == "Discreta":
        DF[variable] = DF[variable].astype(int) 
    else:
        DF[variable] = DF[variable].astype(str) 
    return DF

# Inicializar variables en session_state si no existen
if "independiente" not in st.session_state:
    st.session_state.independiente = []
    st.session_state.tipos_independiente = {}
    st.session_state.largo = 0
    st.session_state.tree_results = None
    st.session_state.terminal_df = None
    st.session_state.boton = False
    st.session_state.Excel = pd.DataFrame()
    st.session_state.Archivo = None
    st.session_state.Archivo_previo = None
    st.session_state.Hoja_seleccionada = None
    st.session_state.Parametros = None
    st.session_state.min_thresholds = None
st.header("Bienvenido a la pagina de funciones visuales")

# Subir archivo
archivo = st.file_uploader("Suba sus archivos aqui", key="file_uploader")

# Verificar si el archivo cambió o se eliminó
if archivo != st.session_state.Archivo_previo:
    st.session_state.Archivo = archivo
    st.session_state.Archivo_previo = archivo
    st.session_state.Excel = pd.DataFrame()  # Reiniciar el DataFrame si el archivo cambió
    st.session_state.Hoja_seleccionada = None

# Leer el archivo solo si hay uno nuevo o cambió la hoja seleccionada
if st.session_state.Archivo is not None:
    if st.session_state.Archivo.name.endswith(".csv"):
        if st.session_state.Excel.empty:
            DF = pd.read_csv(st.session_state.Archivo, delimiter=";")
            DF.fillna('N/A', inplace=True)
            st.session_state.Excel = DF
    elif st.session_state.Archivo.name.endswith(".xlsx"):
        excel_file = pd.ExcelFile(st.session_state.Archivo)
        hojas = excel_file.sheet_names
        hoja_seleccionada = st.selectbox(
            "Selecciona la hoja del archivo Excel que deseas usar:",
            hojas,
            index=hojas.index(st.session_state.Hoja_seleccionada) if st.session_state.Hoja_seleccionada in hojas else 0
        )
        if hoja_seleccionada != st.session_state.Hoja_seleccionada or st.session_state.Excel.empty:
            DF = pd.read_excel(st.session_state.Archivo, sheet_name=hoja_seleccionada)
            DF.fillna('N/A', inplace=True)
            st.session_state.Excel = DF
            st.session_state.Hoja_seleccionada = hoja_seleccionada

# Mostrar vista previa si hay datos en el DataFrame
if not st.session_state.Excel.empty:
    if st.session_state.Archivo.name.endswith(".csv"):
        st.write("Vista previa de los datos:")
    else:
        st.write(f"Vista previa de los datos de la hoja '{st.session_state.Hoja_seleccionada}':")
    st.dataframe(st.session_state.Excel.head())

# Si no hay archivo, reiniciar el DataFrame y la hoja seleccionada
if st.session_state.Archivo is None:
    st.session_state.Excel = pd.DataFrame()
    st.session_state.Hoja_seleccionada = None

# Selección de opciones
eleccion = st.radio("Eliga una opcion", options=["Arboles de decision","Proximamente"], index=None)

try:
    if eleccion == "Arboles de decision" and not st.session_state.Excel.empty:
        DF = st.session_state.Excel
        c1, c2,c3,c4 = st.columns(4)
        with c1:
            variable_dep = st.selectbox("Variable Dependiente", options=DF.columns)
            tipo_variable_dep = st.selectbox("Tipo Variable Dependiente", options=["Continua", "Discreta", "Categorica"])
            variable_indep = st.multiselect("Variables Independientes", options=DF.columns)
            if st.session_state.largo != len(variable_indep):
                st.session_state.independiente = []
                st.session_state.tipos_independiente = {}
            st.session_state.largo = len(variable_indep)
            st.session_state.independiente = variable_indep

        with c2:
            for variable in st.session_state.independiente:
                tipo_variable_indep = st.selectbox(f"Tipo de la variable {variable}", options=["Continua", "Discreta", "Categorica"], key=variable, index=None)
                st.session_state.tipos_independiente[variable] = tipo_variable_indep

        with c3:
            alfa = st.slider("Alfa Split", min_value=float(0), max_value=float(1), step=0.01, value=.05)
            fusiones = st.slider("Alfa Merge", min_value=float(0), max_value=float(1), step=0.01, value=.05)
            porcentaje = st.slider("Porcentaje de Muestra", min_value=0.01, max_value=1.0, step=0.01, value=1.0)
            bonferoni = st.checkbox("Corrección de Bonferoni")
        with c4:
            min_leaf = st.text_input("Minimo De Hoja Hijo", value="20")
            min_node = st.text_input("Minimo De Hoja Padre", value="30")
            min_divisiones = st.number_input("Minimo De Divisiones", min_value=1, max_value=10, value=2)
            max_divisiones = st.number_input("Maximo De Divisiones", min_value=min_divisiones, max_value=11, value=min_divisiones)
        profundidad = st.slider("Nivel De Profundidad", min_value=1, max_value=10, value=3)

        c1, c2 = st.columns(2)
        with c1:
            arbol = st.button("Generar arbol")
        with c2:
            reiniciar = st.button("Reiniciar")

        if reiniciar:
            st.session_state.boton = False
            st.session_state.tree_results = None
            st.session_state.terminal_df = None

        if arbol or st.session_state.boton:
            if not st.session_state.boton:
                for i in range(len(st.session_state.independiente) + 1):
                    if i == 0:
                        variable = variable_dep
                        tipo = tipo_variable_dep
                    else:
                        variable = st.session_state.independiente[i-1]
                        tipo = st.session_state.tipos_independiente.get(variable)
                    DF = Cambiar_tipo(variable, tipo, DF)
                status = st.status("Procesando el árbol...", expanded=True)
                with status:
                    st.write("Paso 1: Aplicando Algoritmo CHAID/ANOVA...")
                    tree = CH.chaid_tree(
                    df=DF,
                    target=variable_dep,
                    features=variable_indep,
                    alpha_merge=float(fusiones),
                    alpha_split=float(alfa),
                    max_depth=int(profundidad),
                    min_sample_node=int(min_leaf),
                    min_sample_split=int(min_node),
                    max_children=int(max_divisiones),
                    min_children=int(min_divisiones),
                    max_iterations=20000,
                    bonferroni_adjustment=bonferoni,
                    large_dataset_threshold=5000,
                    porcentaje=float(porcentaje),
                )
            
                with status:
                    st.write("Paso 2: Creando imagen del árbol...")
                    status.update(label="Creando imagen del árbol", state="running")
                    pdf_data = tree.visualize()
                    st.write("Generando datos de nodos terminales...")
                    status.update(label="Paso 3: Generando datos de nodos terminales", state="running")
                    terminal_df = tree.get_terminal_nodes_data(DF)
                    st.session_state.tree_results = {'pdf_data': pdf_data}
                    st.session_state.terminal_df = terminal_df
                    status.update(label="Tareas completadas", state="complete")

            if st.session_state.tree_results:
                st.session_state.boton = True
                pdf_data = st.session_state.tree_results['pdf_data']
                if pdf_data:
                    st.download_button(
                        label="Descargar Árbol CHAID en PDF",
                        data=pdf_data,
                        file_name="chaid_tree.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )
                else:
                    st.error("No se pudo generar el árbol")

            if st.session_state.terminal_df is not None:
                st.write("Vista previa de los datos con nodos terminales:")
                st.dataframe(st.session_state.terminal_df.head())
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    st.session_state.terminal_df.to_excel(writer, sheet_name='Nodos_Terminales', index=False)
                excel_buffer.seek(0)
                st.download_button(
                    label="Descargar Excel con Nodos Terminales",
                    data=excel_buffer,
                    file_name="nodos_terminales.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
    elif eleccion == "Random Forest" and not st.session_state.Excel.empty:
        DF = st.session_state.Excel
        c1, c2,c3,c4 = st.columns(4)
        st.subheader("Configuración del Árbol de Decisión")
        with c1:
            variable_dep = st.selectbox("Variable Dependiente", options=DF.columns)
            tipo_variable_dep = st.selectbox("Tipo Variable Dependiente", options=["Continua", "Discreta", "Categorica"])
            variable_indep = st.multiselect("Variables Independientes", options=DF.columns)
            if st.session_state.largo != len(variable_indep):
                st.session_state.independiente = []
                st.session_state.tipos_independiente = {}
            st.session_state.largo = len(variable_indep)
            st.session_state.independiente = variable_indep

        with c2:
            for variable in st.session_state.independiente:
                tipo_variable_indep = st.selectbox(f"Tipo de la variable {variable}", options=["Continua", "Discreta", "Categorica"], key=variable, index=None)
                st.session_state.tipos_independiente[variable] = tipo_variable_indep

        with c3:
            alfa = st.slider("Alfa Split", min_value=float(0), max_value=float(1), step=0.01, value=.05)
            fusiones = st.slider("Alfa Merge", min_value=float(0), max_value=float(1), step=0.01, value=.05)
            porcentaje = st.slider("Porcentaje de Muestra", min_value=0.01, max_value=1.0, step=0.01, value=1.0)
            bonferoni = st.checkbox("Corrección de Bonferoni")
        with c4:
            min_leaf = st.text_input("Minimo De Hoja Hijo", value="20")
            min_node = st.text_input("Minimo De Hoja Padre", value="30")
            min_divisiones = st.number_input("Minimo De Divisiones", min_value=1, max_value=10, value=2)
            max_divisiones = st.number_input("Maximo De Divisiones", min_value=min_divisiones, max_value=11, value=min_divisiones)
        profundidad = st.slider("Nivel De Profundidad", min_value=1, max_value=10, value=3)
        
        st.subheader("Configuración del Random Forest")
        with c1:
            n_trees = st.text_input("Número de Árboles", value="1000")
        with c2:
            priority_params = st.multiselect("Parámetros a Priorizar", options=["Homogeneidad", "Profundidad", "Tamaño de nodo", "Significancia"], default=["homogeneidad"])
            st.session_state.priority_params = priority_params
        #with c3:
            #for param in priority_params:
                
        c1, c2 = st.columns(2)
        with c1:
            arbol = st.button("Generar arbol")
        with c2:
            reiniciar = st.button("Reiniciar")

        if reiniciar:
            st.session_state.boton = False
            st.session_state.tree_results = None
            st.session_state.terminal_df = None

        if arbol or st.session_state.boton:
            if not st.session_state.boton:
                for i in range(len(st.session_state.independiente) + 1):
                    if i == 0:
                        variable = variable_dep
                        tipo = tipo_variable_dep
                    else:
                        variable = st.session_state.independiente[i-1]
                        tipo = st.session_state.tipos_independiente.get(variable)
                    DF = Cambiar_tipo(variable, tipo, DF)
                status = st.status("Procesando el árbol...", expanded=True)
                with status:
                    st.write("Paso 1: Aplicando Algoritmo CHAID/ANOVA...")
                    tree = CH.chaid_tree(
                    df=DF,
                    target=variable_dep,
                    features=variable_indep,
                    alpha_merge=float(fusiones),
                    alpha_split=float(alfa),
                    max_depth=int(profundidad),
                    min_sample_node=int(min_leaf),
                    min_sample_split=int(min_node),
                    max_children=int(max_divisiones),
                    min_children=int(min_divisiones),
                    max_iterations=20000,
                    bonferroni_adjustment=bonferoni,
                    large_dataset_threshold=5000,
                    porcentaje=float(porcentaje),
                )
            
                with status:
                    st.write("Paso 2: Creando imagen del árbol...")
                    status.update(label="Creando imagen del árbol", state="running")
                    pdf_data = tree.visualize()
                    st.write("Generando datos de nodos terminales...")
                    status.update(label="Paso 3: Generando datos de nodos terminales", state="running")
                    terminal_df = tree.get_terminal_nodes_data(DF)
                    st.session_state.tree_results = {'pdf_data': pdf_data}
                    st.session_state.terminal_df = terminal_df
                    status.update(label="Tareas completadas", state="complete")

            if st.session_state.tree_results:
                st.session_state.boton = True
                pdf_data = st.session_state.tree_results['pdf_data']
                if pdf_data:
                    st.download_button(
                        label="Descargar Árbol CHAID en PDF",
                        data=pdf_data,
                        file_name="chaid_tree.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )
                else:
                    st.error("No se pudo generar el árbol")

            if st.session_state.terminal_df is not None:
                st.write("Vista previa de los datos con nodos terminales:")
                st.dataframe(st.session_state.terminal_df.head())
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    st.session_state.terminal_df.to_excel(writer, sheet_name='Nodos_Terminales', index=False)
                excel_buffer.seek(0)
                st.download_button(
                    label="Descargar Excel con Nodos Terminales",
                    data=excel_buffer,
                    file_name="nodos_terminales.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
    else:
        st.warning("Suba un archivo por favor")
        st.session_state.Excel = pd.DataFrame()
except NameError as e:
    st.warning("Suba un archivo por favor")
    print(e)