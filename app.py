import streamlit as st
import numpy as np
import pandas as pd

# Set title of the app
st.title("Interactive Streamlit Example")

# Input widgets
name = st.text_input("Enter your name:")
age = st.slider("Select your age", 0, 100, 25)

# Display user inputs
st.write(f"Hello, {name}!")
st.write(f"You are {age} years old.")

# Display a plot based on age
data = pd.DataFrame(
    np.random.randn(100, 2), columns=["x", "y"]
)
data["y"] = data["y"] * age  # Scale y by age
st.line_chart(data)
