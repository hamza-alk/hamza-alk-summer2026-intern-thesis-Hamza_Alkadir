import streamlit as st

from streamlit_app import thesis_verdict_tab


st.set_page_config(page_title="Thesis Verdict", layout="wide")
thesis_verdict_tab()
