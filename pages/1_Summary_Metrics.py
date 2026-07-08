import streamlit as st

from streamlit_app import summary_metrics_tab


st.set_page_config(page_title="Summary Metrics", layout="wide")
summary_metrics_tab()
