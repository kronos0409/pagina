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
eleccion = st.radio("Eliga una opcion", options=["Arboles de decision", "Otra"], index=None)

try:
    if eleccion == "Arboles de decision" and not st.session_state.Excel.empty:
        DF = st.session_state.Excel
        c1, c2 = st.columns(2)
        with c1:
            variable_dep = st.selectbox("Escoga la variable dependiente", options=DF.columns)
            tipo_variable_dep = st.selectbox("Escoga el tipo de la variable dependiente", options=["Continua", "Discreta", "Categorica"])
            variable_indep = st.multiselect("Escoga las variables independientes", options=DF.columns)
            if st.session_state.largo != len(variable_indep):
                st.session_state.independiente = []
                st.session_state.tipos_independiente = {}
            st.session_state.largo = len(variable_indep)
            st.session_state.independiente = variable_indep

        with c2:
            for variable in st.session_state.independiente:
                tipo_variable_indep = st.selectbox(f"Escoga el tipo de la variable {variable}", options=["Continua", "Discreta", "Categorica"], key=variable, index=None)
                st.session_state.tipos_independiente[variable] = tipo_variable_indep

        with c1:
            alfa = st.slider("Eliga el valor de alfa", min_value=float(0), max_value=float(1), step=0.01, value=.05)
            min_leaf = st.text_input("Eliga el minimo de cada hoja", value="20")
        with c2:
            fusiones = st.slider("Eliga el valor para las fusiones", min_value=float(0), max_value=float(1), step=0.01, value=.05)
            min_node = st.text_input("Eliga el minimo de cada hoja padre", value="30")
        profundidad = st.slider("Eliga el nivel de profundidad", min_value=1, max_value=10, value=3)

        c1, c2 = st.columns(2)
        with c1:
            arbol = st.button("Empezar arbol")
        with c2:
            reiniciar = st.button("reiniciar")

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

                tree = CH.chaid_tree(
                    df=DF,
                    target=variable_dep,
                    features=variable_indep,
                    alpha_merge=float(fusiones),
                    alpha_split=float(alfa),
                    max_depth=int(profundidad),
                    min_sample_node=int(min_leaf),
                    min_sample_split=int(min_node),
                    max_children=6,
                    min_children=2,
                    max_iterations=20000,
                    bonferroni_adjustment=True,
                    large_dataset_threshold=5000
                )

                with st.spinner("Construyendo el árbol..."):
                    pdf_data, tree_image = tree.visualize()
                    terminal_df = tree.get_terminal_nodes_data(DF)
                    st.session_state.tree_results = {'pdf_data': pdf_data, 'tree_image': tree_image}
                    st.session_state.terminal_df = terminal_df

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
    elif st.session_state.Excel.empty:
        st.warning("Suba un archivo por favor")

except NameError as e:
    st.warning("Suba un archivo por favor")