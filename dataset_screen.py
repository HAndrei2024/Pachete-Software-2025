import streamlit as st
import pandas as pd
from io import StringIO
import analysis_functions as anf



def dataset_screen(df):
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
        info = anf.custom_info()
        st.write(info)
    

    st.markdown(r"""
        ## 2. Descriptive Statistics
     """)
    if st.button("Afiseaza statistici"):
        st.write("Butonul a fost apăsat!")
        #TODO: Afiseaza descriptive statistici
        st.write(anf.descriptive_statistics())
    

    #TODO: Orice analiza despre set-ul de date

    st.markdown(r"""
        ## 3. Vizualizare Dataset
     """)
    if st.button("Afiseaza dataset"):
        st.write("Butonul a fost apăsat!")

        st.table(df.head())
        st.line_chart(df["PRICE"])


    st.markdown(r"""
        ## 3. Vizualizare Dataset dupa Prelucrare
     """)
    if st.button("Afiseaza dataset dupa prelucrare"):
        st.write("Butonul a fost apăsat!")
        st.write("Tipul coloanei BATH modificat la: int64")
        st.write("Eliminare outliers")
        df = anf.prelucreaza_dataset(df)
        st.table(df.head())
        st.line_chart(df["PRICE"])


    st.markdown(r"""
        ## 4. Filtrare Dataset
     """)
    max_price = df["PRICE"].max()
    valoare_min = st.slider("Selectează prețul minim:", min_value=50, max_value=max_price, value=100)
    valoare_max = st.slider("Selectează prețul maxim:", min_value=50, max_value=max_price, value=250)

    col_names = list(df.columns)
    col_names.remove("PRICE")
    col_names.append("None")
    selected_col = st.selectbox("Selectează o coloană:", col_names)

    sublocalities = list(df["SUBLOCALITY"].unique())
    sublocalities.append("Toate")
    selected_sublocality = st.selectbox("Selecteaza o localitate", sublocalities)

    data_filtrata = df[(df["PRICE"] >= valoare_min) & (df["PRICE"] <= valoare_max)]

    if selected_col != "None":
        data_filtrata = data_filtrata[["PRICE", selected_col]]
    else:
        # data_filtrata = df
        pass

    if selected_sublocality != "Toate" and selected_col == "None":
        data_filtrata = data_filtrata[data_filtrata["SUBLOCALITY"] == selected_sublocality]

    st.write("Produsele filtrate:", data_filtrata)


    st.markdown(r"""
        ## 5. Analiza GroupBy
     """)
    
    if st.button("Analiza GroupBy"):
        response_string = anf.groupby_analysis_chat()
        st.write(response_string)



    st.title("Data Analysis on Properties")

    # Button for Locality Summary
    if st.button('Show Locality Summary'):
        anf_locality_summary = anf.get_locality_summary(df)
        st.write(anf_locality_summary)

    # Button for New York properties sorted by price
    if st.button('Show New York Properties Sorted by Price'):
        anf_ny_properties = anf.get_ny_properties_sorted_by_price(df)
        st.write(anf_ny_properties)

    # Button for Maximum number of beds for House for Sale
    if st.button('Show Max Beds for House for Sale'):
        anf_max_beds_house = anf.get_max_beds_house_for_sale(df)
        st.write(anf_max_beds_house)

    # Button for Property Counts per Locality
    if st.button('Show Property Counts per Locality'):
        anf_property_counts = anf.get_property_counts_per_locality(df)
        st.write(anf_property_counts)

    # Button for Average Beds per Property Type
    if st.button('Show Average Beds per Property Type'):
        anf_avg_beds = anf.get_avg_beds_per_property_type(df)
        st.write(anf_avg_beds)

    # Button for Minimum Size per Property Type
    if st.button('Show Minimum Size per Property Type'):
        anf_min_size = anf.get_min_size_per_property_type(df)
        st.write(anf_min_size)

    # Button for Average Price per Property Type
    if st.button('Show Average Price per Property Type'):
        anf_avg_price = anf.get_avg_price_per_property_type(df)
        st.write(anf_avg_price)

    # Button for Maximum Price per Property Type
    if st.button('Show Maximum Price per Property Type'):
        anf_max_price = anf.get_max_price_per_property_type(df)
        st.write(anf_max_price)
