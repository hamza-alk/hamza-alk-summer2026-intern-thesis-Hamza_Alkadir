import streamlit as st

from streamlit_app import qqq_corrections_dataset_tab


st.set_page_config(page_title="QQQ Corrections ETF Dataset", layout="wide")
qqq_corrections_dataset_tab()
