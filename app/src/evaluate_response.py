import pandas as pd
import streamlit as st

from api.service import FactCheckAPI

# Create a function to check a LLM response
def evaluate_response(factcheck_api: FactCheckAPI):
    """
    This function creates a Streamlit app to evaluate the factuality of a LLM response.
    """
    if 'response' not in st.session_state:
        st.session_state.response = None

    # Initialize the solvers
    claim_processors = factcheck_api.list_claim_processors()
    retrievers = factcheck_api.list_retrievers()
    verifiers = factcheck_api.list_verifiers()

    st.write("This is where you can check factuality of a LLM response.")

    # Customize FactChecker
    st.write("Customize FactChecker")

    # Dropdown in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        claim_processor = st.selectbox("Select Claim Processor", list(claim_processors))
    with col2:
        retriever = st.selectbox("Select Retriever", list(retrievers))
    with col3:
        verifier = st.selectbox("Select Verifier", list(verifiers))

    # Input
    input_text = {"text": st.text_area("Enter LLM response here", "This is a sample LLM response.")}

    # Button to check factuality
    if st.button("Check Factuality"):
        with st.spinner("Preparing pipeline and checking factuality..."):
            # Set api key and configure pipeline
            factcheck_api.pipeline_configure_global({"openai_key": {"value": st.session_state.openai_apikey, "env_name": "OPENAI_API_KEY"}})

            pipeline = factcheck_api.pipeline_configure_solvers({claim_processor: claim_processors[claim_processor]}, {retriever: retrievers[retriever]}, {verifier: verifiers[verifier]})

            # Display pipeline configuration
            pipeline_str = " -> ".join([claim_processor, retriever, verifier])
            st.write("Pipeline:", pipeline_str)

            # Evaluate the response
            response = factcheck_api.evaluate_response(input_text)
            st.session_state.response = response

        # Display the response
        if response and 'result' in response:
            if isinstance(response['result'], bool):
                st.write("The factuality of the LLM response is:", response['result'])
            else:
                st.write("The factuality of the LLM response is:", pd.DataFrame(response['result']))
            st.write("Log:", pd.DataFrame(response['intermediate_result']))

    # Button to reset
    if st.session_state.response:
        if st.button("Reset"):
            st.session_state.response = None
            st.rerun()