import streamlit as st
import numpy as np
import pandas as pd
import home_screen
import dataset_screen

df = pd.read_csv("Dataset.csv")

st.title("Proiect Pachete Software")
st.markdown("Andrei Harja & Stefan Grigoras")

section = st.sidebar.radio("Navigați la:",
                           ["Home Screen", "Dataset",])

if section == "Home Screen":
    home_screen.home_screen()

elif section == "Dataset":
    dataset_screen.dataset_screen(df)
