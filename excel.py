import streamlit as st
import pandas as pd

# Upload a single Excel file
excel_file = st.file_uploader("Upload Excel workbook with Sales and Prices", type="xlsx")

def load_data():
    # Read both sheets into separate DataFrames
    sales_df = pd.read_excel(excel_file, sheet_name=0)   # First sheet
    prices_df = pd.read_excel(excel_file, sheet_name=1)  # Second sheet

    st.write("Sales Data")
    st.dataframe(sales_df)

    st.write("Prices Data")
    st.dataframe(prices_df)

# Load button
load_data_button = st.button("Load data", on_click=load_data)