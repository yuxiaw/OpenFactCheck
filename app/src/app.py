import streamlit as st
from streamlit_option_menu import option_menu
from components.openai import openai_apikey
from components.footer import footer

from cmn.config import Config
from api.service import FactCheckAPI

from src.evaluate_response import evaluate_response
from src.evaluate_llm import evaluate_llm
from src.evaluate_factchecker import evaluate_factchecker
from src.leaderboards import leaderboards
from src.about import about

class App:
    def __init__(self):
        pass

    def run(self, config_path: str):
        # Get the configuration
        config = Config(config_path)

        # Create a FactCheckAPI object
        factcheck_api = FactCheckAPI(config.get("api-url"))

        # Set up Dashboard
        st.set_page_config(page_title="OpenFactCheck Dashboard", 
                        page_icon=":bar_chart:", 
                        layout="wide")

        # Title
        st.markdown("<h1 style='text-align: center;'>OpenFactCheck Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center;'>An Open-source Factuality Evaluation Demo for LLMs</h3>", unsafe_allow_html=True)

        # OpenAI API Key setup only if not set
        default_key = config.get("openai-apikey")
        openai_apikey(default_key)

        # Selection Menu
        selected = option_menu(None, ["Evaluate LLM Response", "Evaluate LLM", "Evaluate FactChecker", "Leaderboards", "About"], 
            icons=['card-checklist', 'check-square', "check2-all", "trophy", "info-circle"],
            menu_icon="cast", 
            default_index=0, 
            orientation="horizontal"
        )

        # Check if backend is running
        try:
            factcheck_api.health()
        except:
            st.error("The backend service is not running. Please start the backend service and refresh the page.")
            return
        
        # Check if OpenAI API Key is set
        if "openai_apikey" not in st.session_state:
            st.error("Please set the OpenAI API Key.")
            return

        # Load the selected page
        if selected == "Evaluate LLM Response":
            evaluate_response(factcheck_api)
        elif selected == "Evaluate LLM":
            evaluate_llm(factcheck_api)
        elif selected == "Evaluate FactChecker":
            evaluate_factchecker(factcheck_api)
        elif selected == "Leaderboards":
            leaderboards(factcheck_api)
        else:
            about(factcheck_api)

        # # Footer
        # footer([
        # "Copyright © 2024 Mohamed Bin Zayed University of Artificial Intelligence",
        # ])