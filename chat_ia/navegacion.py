import streamlit as st

st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Ir a:", ["Inicio", "IA"])

if pagina == "IA":
    st.switch_page("pages/IA")
else:
    st.title("Bienvenido a la app principal")