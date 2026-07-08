import streamlit as st

from streamlit_app import data_validation_tab


st.set_page_config(page_title="Data Validation", layout="wide")
data_validation_tab()
