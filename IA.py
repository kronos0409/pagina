from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import pandas as pd
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import graphviz
# Inicializar el historial del chat en session_state
#decodificar
#______________
def decodificar(df, encoders):
    for col, encoder in encoders.items():
        df[col] = df[col].apply(lambda x: encoder.inverse_transform([int(x)])[0] if pd.notna(x) else x)
    return df
def filtrar_descodificar_y_graficar(df, condiciones, columna_histograma, encoders, numero):
    # Crear un filtro que seleccione todos los datos al inicio
    filtro = pd.Series(True, index=df.index)
    df_resultado= df
    # Aplicar condiciones dinámicamente
    for condicion in condiciones:
        #print(condicion)
        variable, operador, valor = condicion.split()
        #print("funcion")
        #print(variable)
        #print(operador)
        #print(valor)
        valor = float(valor)  # Convertir a número
        #print(df_resultado.head(5))
        if operador == "<=":
            df_resultado = df_resultado[df_resultado[variable] <= valor]
        elif operador == ">=":
            df_resultado = df_resultado[df_resultado[variable] >= valor]
        elif operador == "<":
            df_resultado = df_resultado[df_resultado[variable] < valor]
        elif operador == ">":
            df_resultado= df_resultado[df_resultado[variable] > valor]

    df_resultado = df_resultado.reset_index(drop=True)
    df_resultado = decodificar(df_resultado, encoders)
    #print(encoders)
    #print("df_resultado")
    #print(df_resultado.head(10))
# Mostrar la lista de filtros decodificados
    # Graficar histograma
    fig, ax = plt.subplots()
    ax.hist(df_resultado[columna_histograma], bins=15, edgecolor='black')
    ax.set_xlabel(columna_histograma)
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Histograma de {columna_histograma} en Nodo Terminal {numero}')

# Mostrar en Streamlit
    st.pyplot(fig)
    st.session_state.mensajes.append({"role": "ai", "fig_mat": fig}) 
    return df_resultado
def get_paths_to_terminal_nodes(tree, feature_names):
    """
    Obtiene los caminos desde la raíz hasta cada nodo terminal en un árbol de decisión.
    
    Parámetros:
        tree: objeto DecisionTreeClassifier o DecisionTreeRegressor entrenado.
        feature_names: lista de nombres de las características (columnas).
        
    Retorna:
        Lista de caminos hacia los nodos terminales. Cada camino es una lista de condiciones.
    """
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold

    def traverse(node, path):
        """
        Función recursiva para recorrer el árbol y encontrar los caminos a nodos terminales.
        """
        # Si es un nodo terminal
        if children_left[node] == -1 and children_right[node] == -1:
            paths.append(path)
            return

        # Si no es terminal, añade la condición correspondiente y recorre
        if children_left[node] != -1:
            traverse(
                children_left[node],
                path + [f"{feature_names[feature[node]]} <= {threshold[node]:.4f}"]
            )
        if children_right[node] != -1:
            traverse(
                children_right[node],
                path + [f"{feature_names[feature[node]]} > {threshold[node]:.4f}"]
            )

    # Lista para almacenar los caminos
    paths = []
    traverse(0, [])  # Comienza desde el nodo raíz
    return paths
#____________
#fase_1
#______________________________________________________
def fase_1(DF, solicitud):
    prompt = PromptTemplate(
    input_variables=["input_usuario"],
    template=f"""Eres un sistema de validación de solicitudes. Tu única tarea es clasificar la petición del usuario la cual es "{solicitud}" en **una de las cuatro categorías** y dar una breve explicación de tu decisión.  

### 📌 **Categorías de Respuesta:**  
1. **"petición aceptada"** → La solicitud es clara, razonable y específica en las variables necesarias.  
2. **"petición denegada"** → La solicitud es inválida, imposible de cumplir o no tiene sentido.  
3. **"petición ambigua"** → La solicitud tiene múltiples interpretaciones y no está claro qué quiere el usuario.  
   - ❌ Ejemplo: "Quiero un reporte." (No se especifica qué tipo de reporte).  
   - ❌ Ejemplo: "Dame acceso." (No se sabe a qué recurso).  
4. **"hace falta más información"** → La solicitud es clara, pero incompleta. No se puede procesar sin información adicional.  
   - ❌ Ejemplo: "Dame los datos del cliente Juan." (Falta el apellido o un ID).  
   - ❌ Ejemplo: "Genera un informe de ventas de este mes." (Falta país, ciudad o categoría).  

---

### ⚠️ **Reglas de Clasificación:**  
✔ **NO es ambigua si especifica las variables necesarias, aunque no se tenga el dataset.**  
✔ **NO es ambigua si se da una orden clara pero solo hacen falta las variables.**
✔ **Si hay varias interpretaciones, responde "petición ambigua" y sugiere cómo hacerla más clara.**  
✔ **Si falta información clave, responde "hace falta más información" y menciona qué falta.**  
✔ **Ten mucho cuidado con la diferencia entre "ambigua" y "hace falta más información".**  
✔ **Siempre responde con el formato:**
✔ **No agregues el resumen de los datos**
### 📂 **Contexto:**  
- Tienes acceso a un resumen del archivo de datos.  
- Evalúa si la solicitud es posible basándote en este resumen.  
- Considera solicitudes anteriores para mejorar la respuesta.  

Ejemplo de uso:
    solo debe ser asi, debes de seguir este formato para tu respuesta
    Usuario: {solicitud}
    respuesta: 
**Datos:**  
- **Resumen del archivo:** {DF.head(5)}  
- **Solicitud actual:** {solicitud}, esta es la solicitud que se debe resolver
- **Solicitudes anteriores:** {respuesta_usuario[-20:]}  
            """)
    print(solicitud)
    chain = LLMChain(llm=llm, prompt=prompt)
    respuesta = evaluar_peticion(solicitud, chain)
    with st.chat_message("ai"):
        st.write(respuesta)
    st.session_state.mensajes.append({"role": "ai", "content": respuesta})
            
    if "hace falta más información" in respuesta.lower() or "petición aceptada" in respuesta.lower():
        fase_2(respuesta, solicitud)
#______________________________________________________

#Fase 2
#______________________________________________________
def fase_2(respuesta, solicitud):
    if "hace falta  mas informacion" in respuesta.lower() or "hace falta más información" in respuesta.lower():
        if  not st.session_state.seguir:
            st.session_state.seguir = True
            prompt = PromptTemplate(
            input_variables=["input_usuario"],
            template=f""" 
                Tu trabajo es analizar los siguientes mensajes: '{respuesta}' y {solicitud}, 
                y generar un código en Python que contenga SOLO una lista con máximo 3 preguntas 
                para conseguir la información que falta. SOLO DEBES DEVOLVER LA LISTA EN ESTE FORMATO:

                preguntas = ["pregunta 1", "pregunta 2", "pregunta 3"]

                REGLAS:
                    - SOLO PIDE VARIABLES (ejemplo: '¿Cuál es la variable X?')
                    - NO INCLUYAS EXPLICACIONES, SOLO LA LISTA
                    - SI SE PIDE UN ÁRBOL DE DECISIÓN, PIDE SOLO VARIABLES Y CANTIDAD DE NODOS
                    - piensa bien las preguntas maximo 3 pero intenta que sean 2
                    """
                    )
            preguntas_chain = LLMChain(llm=llm, prompt=prompt)
            preguntas_codigo = preguntas_chain.run(input_usuario=solicitud).strip()
    
        # Limpiar y ejecutar el código generado
            preguntas_codigo = preguntas_codigo.replace("```python", "").replace("```", "").strip()
            preguntas_dict = {}
            try:
                exec(preguntas_codigo, {}, preguntas_dict)
                st.session_state.preguntas_pendientes = preguntas_dict.get("preguntas", [])
            except Exception as e:
                st.session_state.preguntas_pendientes = []
                st.error(f"Error generando preguntas: {e}")

            # Inicializar respuestas en session_state para cada pregunta pendiente
            if st.session_state.preguntas_pendientes:
                for pregunta in st.session_state.preguntas_pendientes:
                    if pregunta not in st.session_state.respuestas_preguntas:
                        st.session_state.respuestas_preguntas[pregunta] = ""
            if st.session_state.preguntas_pendientes:
                with st.form("preguntas_form"):
                    respuestas_usuario = {}
                    for pregunta in st.session_state.preguntas_pendientes:
                        respuestas_usuario[pregunta] = st.text_input(
                        pregunta, 
                        value=st.session_state.respuestas_preguntas.get(pregunta, ""),
                        key=f"input_{pregunta}"
                        )
                    submit_button = st.form_submit_button("Confirmar respuestas")
        
        else:
            if st.session_state.preguntas_pendientes:
                with st.form("preguntas_form"):
                    respuestas_usuario = {}
                    for pregunta in st.session_state.preguntas_pendientes:
                        respuestas_usuario[pregunta] = st.text_input(
                        pregunta, 
                        value=st.session_state.respuestas_preguntas.get(pregunta, ""),
                        key=f"input_{pregunta}"
                            )
                    submit_button = st.form_submit_button("Confirmar respuestas")
            if submit_button:
                respuestas_total = []
                preguntas_total= []
                if all(respuestas_usuario[p].strip() for p in respuestas_usuario):
                    st.session_state.respuestas_preguntas.update(respuestas_usuario)
                    with st.chat_message("ai"):
                        st.success("¡Gracias por responder todas las preguntas!")
                        for pregunta, respuesta in respuestas_usuario.items():
                            st.write(f"**{pregunta}**: {respuesta}")
                            preguntas_total.append(pregunta)
                            respuestas_total.append(respuesta)
                            st.session_state.mensajes.append({"role": "ai", "content": f"**{pregunta}**: {respuesta}"})
                    st.session_state.mensajes.append({"role": "ai", "content": "¡Gracias por responder todas las preguntas!"})
                    st.session_state.preguntas_pendientes = []
                    st.session_state.seguir = False
                    print(respuestas_total)
                    fase_3(respuestas_total, solicitud, respuesta_usuario[i],preguntas_total)
                else:
                    st.warning("Por favor, responde todas las preguntas antes de continuar.")
    else:
        print("peticion aceptada")
        respuestas_total = None
        preguntas_total= None
        try:
            fase_3(respuestas_total, solicitud, respuesta,preguntas_total)     
        except Exception as e:
            with st.chat_message('ai'):
                st.write("Lo lamento, hubo un error en mi codigo, intentalo de nuevo porfavor")
                st.write(e)
            print(e)
#______________________________________________________

#Fase 3
def fase_3(respuestas,solicitud,respuesta_usuario, preguntas):
    print("FASE 3")
    print("____________________")
    if respuestas!= None:
        prompt = PromptTemplate(
            input_variables=["input_usuario"],
            template=f"""
Instrucciones para generar código Python basado en solicitudes del usuario:
- Analizar: '{respuesta_usuario}', {solicitud}
- Generar código Python funcional según lo pedido.

### Herramientas Permitidas
- Bibliotecas: Pandas, Matplotlib, Plotly, Seaborn, Sklearn, LabelEncoder, Graphviz
- Importar explícitamente todas las dependencias usadas.

### Reglas Generales
- DataFrame: Usar 'DF' (existente en el sistema, no crear otro).
- Variables: Usar 'X' e 'y' (no 'x' ni 'y').
- Figuras: Siempre asignar a 'fig', no usar plt.show().
- Sin comentarios en el código generado.
- Información solicitada (ej. promedio, mediana): Asignar a 'mensaje' dinámico.

### Gráficos y Modelos
- Árboles de Decisión:
  - Generar solo si el usuario lo solicita explícitamente.
  - Usar Graphviz y 'tree' de Sklearn.
  - Variable del modelo: 'model'.
- Random Forest:
  - Iteraciones predeterminadas: 1000 (salvo indicación contraria).

### Codificación de Variables Categóricas
- Usar LabelEncoder únicamente para árboles de decisión:
  - Aplicar solo si se solicita un árbol de decisión y hay variables categóricas.
  - Ejemplo: encoder_categoria = LabelEncoder()
            X['categoria'] = encoder_categoria.fit_transform(X['categoria'])
- Generar diccionarios (solo para árboles de decisión):
  - 'diccionario': variable: código: categoría for código, categoría in enumerate(encoder.classes_)
  - 'encoders': variable: encoder_objeto
  - Ejemplo con LITO y MNZ:
    - diccionario = 'LITO': código: categoría, 'MNZ': código: categoría
    - encoders = 'LITO': encoder_lito, 'MNZ': encoder_mnz
- Codificación al final del código, sin usar .update().

### Datos
- Base de datos: 'DF'
  - Columnas: {DF.columns}
  - Resumen: {DF.head(5)}
- No leer ni inventar datos, usar únicamente 'DF'.

###Archivos
-si debes hcaer un archivo o df nuevo, guardalo como DF_Nuevo
"""
                    )
    else:
        prompt = PromptTemplate(
            input_variables=["input_usuario"],
            template=f"""
Instrucciones para generar código Python basado en solicitudes del usuario:
- Analizar: '{respuesta_usuario}', {solicitud}
- Generar código Python funcional según lo pedido.

### Herramientas Permitidas
- Bibliotecas: Pandas, Matplotlib, Plotly, Seaborn, Sklearn, LabelEncoder, Graphviz
- Importar explícitamente todas las dependencias usadas.

### Reglas Generales
- DataFrame: Usar 'DF' (existente en el sistema, no crear otro).
- Variables: Usar 'X' e 'y' (no 'x' ni 'y').
- Figuras: Siempre asignar a 'fig', no usar plt.show().
- Sin comentarios en el código generado.
- Información solicitada (ej. promedio, mediana): Asignar a 'mensaje' dinámico.

### Gráficos y Modelos
- Árboles de Decisión:
  - Generar solo si el usuario lo solicita explícitamente.
  - Usar Graphviz y 'tree' de Sklearn.
  - Variable del modelo: 'model'.
- Random Forest:
  - Iteraciones predeterminadas: 1000 (salvo indicación contraria).

### Codificación de Variables Categóricas
- Usar LabelEncoder únicamente para árboles de decisión:
  - Aplicar solo si se solicita un árbol de decisión y hay variables categóricas.
  - Ejemplo: encoder_categoria = LabelEncoder()
            X['categoria'] = encoder_categoria.fit_transform(X['categoria'])
- Generar diccionarios (solo para árboles de decisión):
  - 'diccionario': variable: código: categoría for código, categoría in enumerate(encoder.classes_)
  - 'encoders': variable: encoder_objeto
  - Ejemplo con LITO y MNZ:
    - diccionario = 'LITO': código: categoría, 'MNZ': código: categoría
    - encoders = 'LITO': encoder_lito, 'MNZ': encoder_mnz
- Codificación al final del código, sin usar .update().

### Datos
- Base de datos: 'DF'
  - Columnas: {DF.columns}
  - Resumen: {DF.head(5)}
- No leer ni inventar datos, usar únicamente 'DF'.

###Archivos
-si debes hcaer un archivo o df nuevo, guardalo como DF_Nuevo
"""
                    ) 
    chain = LLMChain(llm=llm, prompt=prompt)
    codigo = chain.run(input_usuario=solicitud).strip()
    codigo = codigo.replace("```python", "").replace("```", "").strip()
    mensaje = ''
    model = ''
    encoders = {}
    cluster = pd.DataFrame()
    X= pd.DataFrame()
    X_porsiaca = pd.DataFrame()
    Y = pd.Series()
    DF_Nuevo = pd.DataFrame()
    diccionario = {}
    with st.chat_message("ai"):
        st.code(codigo)
        st.session_state.mensajes.append({"role": "ai", "code": codigo})
    entorno = {"plt": plt, "DF": DF, "mensaje": mensaje, "DF_Nuevo" : DF_Nuevo, "diccionario": diccionario, "X": X, "model": model, "encoders" : encoders, "y" : Y, "cluster": cluster, "x" : X_porsiaca}  # Asegúrate de definir DF antes
    if "X" in entorno:
        X = entorno["X"]
        if X.empty:
            X = entorno["x"]
    # Ejecutar el código
    exec(codigo, entorno)

    # Verificar si `fig` existe en el entorno
    if "fig" in entorno:
        fig = entorno["fig"]
    # Si es una figura de Matplotlib
        try:
            if isinstance(fig, plt.Figure):
                st.pyplot(fig)
                st.session_state.mensajes.append({"role": "ai", "fig_mat": fig}) 
            # Si es una figura de Plotly
            elif isinstance(fig, go.Figure):
                st.plotly_chart(fig)
                st.session_state.mensajes.append({"role": "ai", "fig_plot": fig}) 
            else:
                st.graphviz_chart(fig)
                st.session_state.mensajes.append({"role": "ai", "arbol": fig}) 
        except:
            st.write("Lo lamento, hubo un error en el codigo")
    if 'mensaje' in entorno:
        with st.chat_message('ai'):
            st.write(entorno["mensaje"])
            st.session_state.mensajes.append({"role": "ai", "content": entorno["mensaje"]})
    if "DF_Nuevo" in entorno and not entorno["DF_Nuevo"].empty:
        with st.chat_message('ai'):
            st.write(entorno["DF_Nuevo"])
            st.session_state.mensajes.append({"role": "ai", "DF": entorno["DF_Nuevo"]})
        output = io.BytesIO()
        DF_Nuevo = entorno["DF_Nuevo"]
        #print(DF_Nuevo.head())
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            DF_Nuevo.to_excel(writer, index=False)
            output.seek(0)
        st.session_state.archivo = output
        if st.session_state.archivo:
            st.download_button("Descarga tus archivos aca",file_name="archivo.xlsx",data=st.session_state.archivo,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")  
    if "cluster " in entorno and not entorno["cluster"].empty:
        st.dataframe(entorno["cluster"])
    if "diccionario" in entorno and len(entorno["diccionario"]) !=0:
        
        X = entorno["X"]
        
        Y =  entorno['y']
        #print("X")
        #print(X.head(5))
        #print("Y")
        #print(Y.head(5))
        #print("holi")
        df_histo = pd.concat([X, Y], axis=1)
        #print(df_histo.head(5))
        model = entorno["model"]
        #print(model)
        diccionario = entorno["diccionario"]
        diccionario_df=pd.DataFrame(entorno["diccionario"])
        st.dataframe(diccionario_df)
        st.session_state.mensajes.append({"role": "ai", "DF": diccionario})
        #print("hola")
        paths = get_paths_to_terminal_nodes(model,X.columns)
        nodos  = []
        for i, path in enumerate(paths):
            camino=[]
            mensaje=(f"nodo terminal {i + 1}:")
            for step in path:
                camino.append(step)
            nodos.append(camino)
        n = 1
        hojas = []
        encoders = entorno["encoders"]
        for caminos in nodos:
            hoja = filtrar_descodificar_y_graficar(df_histo,caminos, Y.name, encoders, n)
            n+=1
            claves = []
            hoja["filtro"] = ''
            for condicion in caminos:
                variable, operador, valor = condicion.split()
                #print("condicion")
                #print(variable)
                #print(valor)
                #print(operador)
                #print(diccionario)
                valor = float(valor)  # Convertir a número
                dict = diccionario[variable]
                #print(dict)
                #for k,v in dict.items():
                #    print(f"k: {k}\nV: {v}")
                if operador == "<=":     
                    clave = [v for k, v in dict.items() if k <= valor]
            
                elif operador == ">=":
                    clave = [v for k, v in dict.items() if k >= valor]
                elif operador == "<":
                    clave = [v for k, v in dict.items() if k < valor]
                elif operador == ">":
                    clave = [v for k, v in dict.items() if k > valor]
                if clave:
                    clave.insert(0, f"{variable} : ")
                    claves.append(clave)
            i = 1
    
            for clave in claves:
                #print(claves)
                #print(clave)
                cadena = clave[0] + " " + ", ".join(clave[1:])
                #print(clave)
                hoja.at[i,"filtro"] = cadena    
                i+=1
            hojas.append(hoja)
        nombre_archivo = "nodos_terminales.xlsx"
        columnas = list(df_histo.columns)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for i, df in enumerate(hojas, start=1):  # Comenzamos desde 1
                print("DFS")
                print(df.head(10))
                df = df.drop_duplicates()
                df_merge= DF.merge(df, on=columnas, how= 'inner')   
                df_merge = df_merge.drop_duplicates()
                #df_merge = df_merge.drop_duplicates(subset=['filtro'])
                df_merge.loc[df_merge.duplicated(subset=['filtro']), 'filtro'] = ''
                df_merge.to_excel(writer, sheet_name=f"Nodo terminal {i}", index=False)
                output.seek(0)
            print(f"Archivo {nombre_archivo} guardado con éxito.")
        st.session_state.archivo = output
        if st.session_state.archivo:
            st.download_button("Descarga tus archivos aca",file_name=nombre_archivo,data=st.session_state.archivo,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")  
    
#______________________________________________________

#IA
#___________________________
def evaluar_peticion(peticion, chain):
    
    return chain.run(input_usuario=peticion).strip()
#___________________________
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
    st.session_state.seguir = False
    st.session_state.i = 0
if "archivo" not in st.session_state:
    st.session_state.archivo = pd.DataFrame()
if "preguntas_pendientes" not in st.session_state:
    st.session_state.preguntas_pendientes = []
if "respuestas_preguntas" not in st.session_state:
    st.session_state.respuestas_preguntas = {}

# Mostrar mensajes previos (si existen)
# Clave de OpenAI
openai_api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else st.text_input("Ingresa tu clave de OpenAI", type="password")
if not openai_api_key:
    st.warning("Por favor, ingresa tu clave de OpenAI para continuar.")
    st.stop()

# Inicializar modelo
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=openai_api_key)

respuesta_usuario = []
respuesta_ia = []

st.title("Asistente Inteligente para Data Science")
archivo = st.file_uploader("Sube tu archivo Excel o CSV", type=["xlsx", "csv"])

# Variable para almacenar el DataFrame
DF = None

if archivo:
    # Si es un archivo CSV
    if archivo.name.endswith(".csv"):
        DF = pd.read_csv(archivo, delimiter=";")
        DF.fillna('N/A', inplace=True)
        st.write("Vista previa de los datos:")
        st.dataframe(DF.head())
    
    # Si es un archivo Excel
    elif archivo.name.endswith(".xlsx"):
        # Leer el archivo Excel para obtener las hojas disponibles
        excel_file = pd.ExcelFile(archivo)
        hojas = excel_file.sheet_names  # Lista de nombres de hojas
        
        # Permitir al usuario seleccionar una hoja
        hoja_seleccionada = st.selectbox("Selecciona la hoja del archivo Excel que deseas usar:", hojas)
        
        # Cargar el DataFrame de la hoja seleccionada
        if hoja_seleccionada:
            DF = pd.read_excel(archivo, sheet_name=hoja_seleccionada)
            DF.fillna('N/A', inplace=True)
            st.write(f"Vista previa de los datos de la hoja '{hoja_seleccionada}':")
            st.dataframe(DF.head())
    
    # Continuar solo si DF está definido
    if DF is not None:
        for mensaje in st.session_state.mensajes:
            with st.chat_message(mensaje["role"]):
                try:
                    if mensaje['role'] == 'user':
                        respuesta_usuario.append(mensaje['content'])
                    else:
                        respuesta_ia.append(mensaje['content'])
                    if isinstance(mensaje["content"], str):
                        st.write(mensaje["content"])    
                except:
                    try:
                        if "code" in mensaje:
                            st.code(mensaje["code"])
                        elif "fig_mat" in mensaje:
                            st.pyplot(mensaje["fig_mat"])
                        elif "fig_plot" in mensaje:
                            st.plotly_chart(mensaje["fig_plot"])
                        elif "arbol" in mensaje:
                            st.graphviz_chart(mensaje["arbol"])
                        else:
                            st.dataframe(mensaje["DF"])
                    except Exception as e:
                        st.error(f"Error al mostrar mensaje: {e}")
        
        solicitud = st.chat_input("Escribe tu consulta sobre los datos...")
        if solicitud:
            st.session_state.mensajes.append({"role": "user", "content": solicitud})
            with st.chat_message("user"):
                st.write(solicitud)
            fase_1(DF, solicitud)
        
        try:
            if st.session_state.seguir:
                i = st.session_state.i 
                fase_2(respuesta_ia[0], respuesta_usuario[i])
                st.session_state.i += 1
        except Exception as e:
            print(e)
else:
    st.session_state.mensajes = []