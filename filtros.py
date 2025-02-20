import pandas as pd
import streamlit as st
import os
import time
import warnings
import io
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

st.sidebar.title("Funciones de filtro")
st.sidebar.header("Información antes de empezar:")
st.sidebar.markdown("1-Se requiere archivo Excel")
st.sidebar.markdown("2-Esta pagina esta pensada para facilitar el trabajo de filtrado, optimizacion y creacion de tramos a partir de prinicpalmente ['HOLEID', 'From' y 'To']")
st.header("Empezando pruebas")
if "accion" not in st.session_state:
    st.session_state.accion = None
if "texto" not in st.session_state:
    st.session_state.texto = ''
def barra_progreso(lista_excel,BD_excel):
    total_filas = 0
    for i in range(0,2):
        if i == 0:
            archivo = lista_excel   
        else:
            archivo = BD_excel
        for hoja in archivo.sheet_names:
            df = pd.read_excel(archivo, sheet_name=hoja)
            filas_hoja = df.shape[0]
            total_filas += filas_hoja
    patron = 1/((total_filas)*len(BD_excel.sheet_names))   
    print("__________")
    print(total_filas  )
    print(patron)
    return patron
def procesar_hojas(archivo,tramo):
    xls = pd.read_excel(archivo,sheet_name=None)
    output = io.BytesIO()
    if tramo != '':
        hojas_procesadas = [dividir_en_tramos(df, tramo) for df in xls.values()]
        df_consolidado = hojas_procesadas[0]
        progreso = st.progress(0)
        patron = len(hojas_procesadas)
        patron = int(np.round(100/(patron)))
        n = 0
        if len(hojas_procesadas) !=1:
           for hoja in hojas_procesadas[1:]:
               n+=1
               df_consolidado = pd.merge(df_consolidado, hoja, on=['HOLEID', 'From', 'To'], how='outer')
               with pd.ExcelWriter(output, engine="openpyxl") as writer:
                   df_consolidado.to_excel(writer, index=False)
                   output.seek(0)
                   progress =n*patron
                   progreso.progress(progress)
        else:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                   df_consolidado.to_excel(writer, index=False)
                   output.seek(0)
                   progress =n*patron
                   progreso.progress(progress)
        progreso.progress(100)
        return output
def dividir_en_tramos(df, tramo_size):
    filas_tramos = []
    for holeid in df['HOLEID'].unique():
        sub_df = df[df['HOLEID'] == holeid]
        max_to = sub_df['To'].max()  # Usamos el valor máximo de 'To' solo al final de cada HOLEID
        current_from = 0  # Iniciar 'From' en 0 para cada HOLEID
        
        while current_from < max_to:
            current_to = current_from + tramo_size
            mensaje = ''  # Reiniciar mensaje en cada tramo
            
            # Filtrar filas dentro del rango actual
            mask = (sub_df['From'] < current_to) & (sub_df['To'] > current_from)
            matching_rows = sub_df[mask]
            
            if not matching_rows.empty:
                row_principal,mensaje_ERROR = obtener_holeid_principal(matching_rows, current_from, current_to)
                
                # Agregar el mensaje si es el último tramo que supera el max_to
                to_actual = matching_rows.iloc[0]['To']
                if current_to>to_actual:
                    mensaje = mensaje_ERROR
                if current_to > max_to:
                    current_to = max_to  # Limitar current_to solo si ya es el último tramo del HOLEID
                    mensaje= mensaje_ERROR
                nueva_fila = [holeid, current_from, current_to] + row_principal.iloc[3:].to_list()[:len(df.columns) - 3] + [mensaje]
            else:
                nueva_fila = [holeid, current_from, current_to] + [None] * (len(df.columns) - 3) + [mensaje]
            
            filas_tramos.append(nueva_fila)
            current_from = current_to  # Avanzar al siguiente tramo
    
    columnas = ['HOLEID', 'From', 'To'] + df.columns[3:].to_list()[:len(df.columns)-3] + ['mensaje']
    return pd.DataFrame(filas_tramos, columns=columnas)
def obtener_holeid_principal(matching_rows, current_from, current_to):
    matching_rows['covered_range'] = matching_rows.apply(lambda row: min(row['To'], current_to) - max(row['From'], current_from), axis=1)
    max_covered_range = matching_rows['covered_range'].max()
    top_rows = matching_rows[matching_rows['covered_range'] == max_covered_range]
    if len(top_rows) > 1:
        combined_row = top_rows.iloc[0].copy()
        for col in top_rows.columns[3:]:  
            combined_values = "/".join(top_rows[col].astype(str).unique())
            combined_row[col] = combined_values
        return combined_row, "ERROR 50%"
    if len(matching_rows) == 1:
        return top_rows.iloc[0], ""
    return top_rows.iloc[0], "2 o mas"
def Menu(archivos):
    if st.toggle("Asignacion?"):
        st.subheader("las columnas deben ser 'HOLEID', 'From' y 'To'")
        nombre=[]
        for i in range(0,len(archivos)):
            nombre.append(archivos[i].name)
        BD = st.selectbox("Cual es la base de datos?",options=nombre)
        lista = st.selectbox("Cual es la lista?",options=nombre)
        funcion = 4
    else:
        st.subheader("El codigo esta diseñado pensando en que tengas una hoja extra sin informacion que actua como una clase de leyenda, si no tienes esa hoja, agrega una vacia porfavor y asegurate que esta este al ultimo")
        st.write("Tambien si tienes la columna 'length' o una columna que actue como sustituto, porfavor eliminala")
        st.write("Cuando tengas la optimizacion hecha, subelo a la pagina")
        funcion=st.radio("Selecciona lo que desees hacer",index=None ,options = ["1-Filtrado", "2-Optimizacion", "3-Tramos (no ejecutar sin antes optimizar)"])
        if funcion == '1-Filtrado' :
            funcion = 1
        elif funcion == "2-Optimizacion":
            funcion = 2
        elif funcion == "3-Tramos (no ejecutar sin antes optimizar)":
            funcion=3
        else:
            funcion = 0
        BD = 0
        lista = 0
    aplicar = st.button("Aplicar función")
    return aplicar,funcion, BD, lista
def Conseguir_archivo():
    tramo = st.toggle("Tienes la optimizacion hecha?")
    return st.file_uploader("introduce tu archivo", accept_multiple_files=tramo),tramo
def Filtrado(archivo):
    data = {}
    xlsx = pd.ExcelFile(archivo)
    progreso = st.progress(0)
    patron = len(xlsx.sheet_names[:-1])
    patron = int(np.round(100/((patron*2)+1)))
    n = 0
    for sheet in xlsx.sheet_names[:-1]:
        n+=1
        start = time.time()
        df = pd.read_excel(xlsx, sheet_name=sheet)
        df['traslapo'] = (df['To'] > df['From'].shift(-1)) & (df['HOLEID'] == df['HOLEID'].shift(-1))
        df['traslapo'] = df['traslapo'].apply(lambda x: 'T' if x else 'F')
        df['vacio'] = (df['To'] < df['From'].shift(-1)) & (df['HOLEID'] == df['HOLEID'].shift(-1))
        df['vacio'] = df['vacio'].apply(lambda x: 'T' if x else 'F')
        df['cuadro vacio'] = df.isnull().any(axis=1).apply(lambda x: 'T' if x else 'F')
        df['revisar'] = "F"
        for i in range(len(df) - 1):
            if df.loc[i, 'traslapo']=='T' or df.loc[i,'vacio']== 'T':
                df.loc[i, 'revisar'] = 'T'
                df.loc[i + 1, 'revisar'] = 'T'
        data[sheet] = df
        progress =n*patron
        progreso.progress(progress)
    start_escritura = time.time()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet, df in data.items():
            n+=1
            start = time.time()      
            df.to_excel(writer, sheet_name=sheet, index=False)
            output.seek(0)
            fin = time.time()
            progress =n*patron
            progreso.progress(progress)
    progreso.progress(100)
    return output
def Optimizacion(archivo):
    data={}
    xlsx = pd.ExcelFile(archivo)
    progreso = st.progress(0)
    patron = len(xlsx.sheet_names[:-1])
    patron = int(np.round(100/((patron*2)+1)))
    n = 0  
    for sheet in xlsx.sheet_names[:-1]:
        n+=1
        start = time.time()
        df = pd.read_excel(xlsx, sheet_name=sheet)
        result = []
        fila_inicial= df.iloc[0].copy()
        for i in range(1,len(df)):
            fila_siguiente=df.iloc[i]
            if (fila_inicial['HOLEID'] == fila_siguiente['HOLEID']) and fila_inicial[3:].equals(fila_siguiente[3:]):
                fila_inicial['From']= min(fila_inicial['From'], fila_siguiente['From'])
                fila_inicial['To'] = max(fila_inicial['To'], fila_siguiente['To'])
            else:
                result.append(fila_inicial)
                fila_inicial= fila_siguiente.copy()
        result.append(fila_inicial)
        optimizado = pd.DataFrame(result)
        data[sheet] = optimizado
        progress =n*patron
        progreso.progress(progress)
    output = io.BytesIO()
    start_escritura = time.time()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet, df in data.items():
            n +=1
            start = time.time()      
            df.to_excel(writer, sheet_name=sheet, index=False)
            output.seek(0)
            progress =n*patron
            progreso.progress(progress)
    progreso.progress(100)
    return output 
def Asignacion_final(BD, lista, archivos,progress):
    for archivo in archivos:
        if archivo.name == BD:
            df_BD_dict = pd.read_excel(archivo, sheet_name=None)  # Diccionario de hojas
            excel = pd.ExcelFile(archivo)
        elif archivo.name == lista:
            df_lista = pd.read_excel(archivo)
            
    # Convertir las columnas de `df_lista` a los tipos requeridos
    df_lista['HOLEID'] = df_lista['HOLEID'].astype(str).str.strip()
    df_lista['From'] = df_lista['From'].astype(float)
    df_lista['To'] = df_lista['To'].astype(float)
    progreso =  st.progress(0)
    resultados = []
    patron = progress
    # Procesar cada hoja de `df_BD`
    hojas = excel.sheet_names
    i=0
    conteo = 0
    for hoja, df_BD in df_BD_dict.items():
        
        if 'HOLEID' not in df_BD.columns:
            continue  # Saltar hojas sin la columna 'HOLEID'

        # Convertir las columnas de `df_BD` a los tipos requeridos
        df_BD['HOLEID'] = df_BD['HOLEID'].astype(str).str.strip()
        df_BD['From'] = df_BD['From'].astype(float)
        df_BD['To'] = df_BD['To'].astype(float)

        # Recorrer cada fila de `df_lista`
        for _, fila_lista in df_lista.iterrows():
            # Filtrar coincidencias en `df_BD` por HOLEID y rango From-To
            filtro = (
                (df_BD['HOLEID'] == fila_lista['HOLEID']) &
                (df_BD['From'] < fila_lista['To']) &
                (df_BD['To'] > fila_lista['From'])
            )
            coincidencias = df_BD[filtro]

            nueva_fila = fila_lista.to_dict()

            if not coincidencias.empty:
                # Calcular el porcentaje de intersección
                coincidencias = coincidencias.copy()
                coincidencias['Intersección'] = (
                    coincidencias[['To', 'From']].min(axis=1) - 
                    coincidencias[['To', 'From']].max(axis=1)
                ).clip(lower=0)
                coincidencias['Porcentaje'] = (
                    coincidencias['Intersección'] / (fila_lista['To'] - fila_lista['From'])
                ) * 100

                # Determinar las filas con el porcentaje máximo
                max_porcentaje = coincidencias['Porcentaje'].max()
                filas_maximas = coincidencias[coincidencias['Porcentaje'] == max_porcentaje]

                if len(filas_maximas) > 1:
                    # Combinar información en caso de múltiples coincidencias
                    for columna in df_BD.columns:
                        if columna not in df_lista.columns:
                            nueva_fila[columna] = '/'.join(map(str, filas_maximas[columna].unique()))
                    nueva_fila[f'comentario_{hojas[i]}'] = 'ERROR: Más de una caracteristica en el rango'
                else:
                    # Asignar información de la única coincidencia
                    for columna in df_BD.columns:
                        if columna not in df_lista.columns:
                            nueva_fila[columna] = filas_maximas.iloc[0][columna]
                    nueva_fila[f'comentario_{hojas[i]}'] = ''
            else:
                # Sin coincidencias, rellenar con datos vacíos
                for columna in df_BD.columns:
                    if columna not in df_lista.columns:
                        nueva_fila[columna] = ''
                nueva_fila[f'comentario_{hojas[i]}'] = 'Sin coincidencias'
            progreso.progress(patron)
            
            patron+=progress
            # Agregar la nueva fila al resultado
            resultados.append(nueva_fila)
            conteo+=1
            #if conteo == 100:
            ##    break
        #if conteo == 100:
        #    break
        i+=1
    print(conteo)
    df_resultado = pd.DataFrame(resultados).drop_duplicates()
    return df_resultado,patron,progreso

def Asignacion_inicial(BD, lista, archivos):
    for archivo in archivos:
        if archivo.name == BD:
            df_BD_dict = pd.read_excel(archivo, sheet_name=None)
            BD_excel = pd.ExcelFile(archivo)
        elif archivo.name == lista:
            df_lista = pd.read_excel(archivo, sheet_name=None)
            lista_excel = pd.ExcelFile(archivo)
    output = io.BytesIO()
    progress = barra_progreso(lista_excel,BD_excel  )
    hojas_procesadas,progreso_final,progreso =Asignacion_final(BD, lista, archivos,progress)
    hojas_procesadas = hojas_procesadas.fillna('')
    hojas_procesadas = hojas_procesadas.groupby(['HOLEID', 'From', 'To'], as_index=False).agg(
    
    lambda x: ' / '.join(
        sorted(set(str(val) for val in x if pd.notna(val) and val != ''))
    )
)   
    patron = (1-progreso_final)/2
    progreso_final+=patron
    progreso.progress(progreso_final)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        hojas_procesadas.to_excel(writer, index=False)

    output.seek(0)
    progreso.progress(100)
    return output
archivo,opti = Conseguir_archivo()
if type(archivo) == list:
    for i in range(0,len(archivo)):
        if archivo[i].name == "Optimizado.xlsx":
            posicion_opti = i
            if i == 0:
                posicion = 1
            else:
                posicion = 0
if archivo != [] and archivo != None:
    seguir,funcion,BD,lista = Menu(archivo)
    if seguir == True:
        if funcion == 1:
            st.session_state.accion = ""
            if type(archivo) == list and len(archivo)!=1:
                archivo_filtrado = Filtrado(archivo[posicion])
                if len(archivo) == 1:
                    st.warning("Solo has ingresado 1 archivo, desactiva 'Tienes la optimizacion hecha?'")
                    archivo_filtrado=''
            else:
                archivo_filtrado = Filtrado(archivo)
            if archivo_filtrado!='':
                st.download_button("Descarga tus archivos aca",file_name="Filtrado.xlsx" ,data=archivo_filtrado,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        elif funcion == 2:
            st.session_state.accion = ""
            if type(archivo) == list and len(archivo)!=1:
                archivo_optimizado = Optimizacion(archivo[posicion])
                if len(archivo) == 1:
                    st.warning("Solo has ingresado 1 archivo, desactiva 'Tienes la optimizacion hecha?'")
                    archivo_optimizado=''
            else:
                archivo_optimizado = Optimizacion(archivo)
            if archivo_optimizado!='': 
                st.download_button("Descarga tus archivos aca",file_name="Optimizado.xlsx" ,data=archivo_optimizado,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        elif funcion == 3:
            st.session_state.accion = "tramos"
        else:
            archivo_asignado = Asignacion_inicial(BD,lista,archivo)
            if archivo_asignado!='':
                st.download_button("Descarga tus archivos aca",file_name="archivo_asignado.xlsx",data=archivo_asignado,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if st.session_state.accion == "tramos" and funcion==3 and opti == True:
        st.session_state.texto = st.text_input("dime la cantidad que seran los tramos:",value=st.session_state.texto)
        tramo = st.session_state.texto
        if st.session_state.texto:
            if type(archivo) == list and tramo !='':
                with st.spinner("Dividiendo Tramos"):
                    archivo_tramos=procesar_hojas(archivo[posicion_opti],float(tramo))
                st.download_button("Descarga tus archivos aca",file_name="Tramos.xlsx" ,data=archivo_tramos,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                st.session_state.accion = ""
    else:
        if st.session_state.accion == "tramos" and funcion==3:
            if archivo.name == "Optimizado.xlsx":
                st.warning("Activa 'Tienes la optimizacion hecha?'")
            else:
                st.error("Asegurate de que hayas ingresado el archivo 'Optimizacion.xlsx'")
