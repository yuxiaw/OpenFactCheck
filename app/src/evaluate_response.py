import pandas as pd
import streamlit as st

from api.service import FactCheckAPI

# Create a function to check a LLM response
def evaluate_response(factcheck_api: FactCheckAPI):
    """
    This function creates a Streamlit app to evaluate the factuality of a LLM response.
    """
    # Initialize the solvers
    claim_processors = factcheck_api.list_claim_processors()
    retrievers = factcheck_api.list_retrievers()
    verifiers = factcheck_api.list_verifiers()

    st.write("This is where you can check factuality of a LLM response.")

    # Customize FactChecker
    st.write("Customize FactChecker")

    # Dropdown in four columns
    col1, col2, col3 = st.columns(3)
    with col1:
        claimprocessors_labels = claim_processors
        claimprocessor = st.selectbox("Select Claim Processer", claimprocessors_labels)
    with col2:
        retrivers_labels = retrievers
        retriever = st.selectbox("Select Retriever", retrivers_labels)
    with col3:
        verifiers_labels = verifiers
        verifier = st.selectbox("Select Verifier", verifiers_labels)

    # Input
    input_text = {"text": st.text_area("Enter LLM response here", "This is a sample LLM response.")}
    
    # Button
    if st.button("Check Factuality"):

        # Prepare Pipeline
        with st.spinner("Preparing pipeline..."):
            print("Preparing pipeline...")
            claimprocessor_dict = {claimprocessor: claim_processors[claimprocessor]}
            retriever_dict = {retriever: retrievers[retriever]}
            verifier_dict = {verifier: verifiers[verifier]}
            pipeline = factcheck_api.pipeline_configure(claimprocessor_dict, retriever_dict, verifier_dict)

            # Create a pipeline like this:
            # Pipeline: ClaimProcessor -> Retriever -> Verifier
            pipeline_str = " -> ".join([claimprocessor, retriever, verifier])
            st.write("Pipeline:", pipeline_str)
            
        # Display a waiting message
        with st.spinner("Checking factuality of the LLM response..."):
            print("Checking factuality of the LLM response...")
            print("Input Text:", input_text)
            response = factcheck_api.evaluate_response(input_text)

        # Display the response
        if type(response["result"]) is not bool:
            st.write("The factuality of the LLM response is:", pd.DataFrame(response["result"]))
        else:
            st.write("The factuality of the LLM response is:", response["result"])

        # Display the log
        st.write("Log:", pd.DataFrame(response["intermediate_result"]))