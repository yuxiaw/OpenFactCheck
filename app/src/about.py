import streamlit as st

from api.service import FactCheckAPI

def about(factcheck_api: FactCheckAPI):
    """
    This function creates a Streamlit app to display information about the dashboard.
    """
    st.write("This is where you can find information about the dashboard.")