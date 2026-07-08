import streamlit as st

from streamlit_app import statistical_analysis_tab


st.set_page_config(page_title="Statistical Analysis", layout="wide")
statistical_analysis_tab()
