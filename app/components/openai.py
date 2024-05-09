import os
import streamlit as st

from cmn.openai import validate_apikey

def openai_apikey(default_key):
    """
    This function sets the OpenAI API Key for the demo.
    """

    if 'openai_apikey' not in st.session_state:
        st.session_state.openai_apikey = default_key

    # Check if the key is already validated
    if not st.session_state.get('openai_apikey_valid', False):
        # Divider
        st.markdown("---")

        # Set OpenAI API Key
        st.markdown("OpenAI API Key is required for this demo. Please enter your OpenAI API Key below.", unsafe_allow_html=True)

        # Create two columns with equal width
        col1, col2 = st.columns([7, 1])

        # Place the text input in the first column
        with col1:
            new_key = st.text_input("Enter OpenAI API Key", type="password", label_visibility="collapsed", value=st.session_state.openai_apikey or '')

        # Place the button in the second column
        with col2:
            if st.button("Set OpenAI API Key", use_container_width=True):
                if validate_apikey(new_key):
                    st.success("OpenAI API Key is set successfully.")
                    st.session_state.openai_apikey = new_key
                    st.session_state.openai_apikey_valid = True
                    os.environ["OPENAI_API_KEY"] = new_key
                else:
                    st.error("Incorrect API key provided. Please check and try again.")

        if st.session_state.openai_apikey and validate_apikey(st.session_state.openai_apikey):
            st.session_state.openai_apikey_valid = True
            st.success(f"OpenAI API Key is currently set. API Key is valid.")
        else:
            st.info("Please enter your OpenAI API Key. You can find your API key at https://platform.openai.com/account/api-keys.")

        # Divider
        st.markdown("---")
    else:
        # Display a success message if the key is already validated
        st.success("OpenAI API Key is already set and valid.")
