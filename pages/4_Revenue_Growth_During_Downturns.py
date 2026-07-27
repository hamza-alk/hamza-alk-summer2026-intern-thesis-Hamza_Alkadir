import streamlit as st

from streamlit_app import revenue_growth_dataset_tab


st.set_page_config(page_title="Revenue Growth During QQQ Downturns", layout="wide")
revenue_growth_dataset_tab()
