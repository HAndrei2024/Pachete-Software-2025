import streamlit as st

def home_screen():
    st.header("Introducere")
    st.markdown(r"""
        ### Care este scopul proiectului?
                
        Vrem sa facem proiectul la Pachete Software!
        - **Tema 1:** Sa facem Home Page-ul!
        - **Tema 2:** Sa analizam un set de date!
     """)
    
    if st.button("Mergi la set-ul de date!"):
        st.write("Butonul a fost apăsat!")