import streamlit as st

def dataset_screen():
    st.header("Apartments Dataset")
    st.markdown(r"""
        Ne-am ales un dataset cu informatii despre locuinte de vanzare/inchiriat din Statele Unite ale Americii. Dataset-ul este in format csv si este potrivit pentru analiza preturilor per total, pentru evidentierea anumitor tipare cand vine vorba de zona, etc. Inainte de analiza propriu-zisa, preprocesarea datelor este necesara intrucat nu toate elementele au o forma standard. 
     """)

    st.markdown(r"""
        ## 1. Informatii despre dataset
     """)
    if st.button("Afiseaza informatii"):
        st.write("Butonul a fost apăsat!")
        #TODO: Afiseaza basic info
    

    st.markdown(r"""
        ## 2. Descriptive Statistics
     """)
    if st.button("Afiseaza statistici"):
        st.write("Butonul a fost apăsat!")
        #TODO: Afiseaza descriptive statistici
    

    #TODO: Orice analiza despre set-ul de date
