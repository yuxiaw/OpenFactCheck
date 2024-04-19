import os
import uuid
import json
import zipfile
import pandas as pd
import streamlit as st

from api.service import FactCheckAPI

from src.leaderboards import save_leaderboard

def evaluate_factchecker(factcheck_api: FactCheckAPI):
    """
    This function creates a Streamlit app to evaluate a Factchecker.
    """
    st.write("This is where you can evaluate the factuality of a FactChecker.")

    # Display the instructions
    st.write("Download the benchmark evaluate the factuality of a FactChecker.")

    # File path to the benchmark
    file_claims_path = 'assets/templates/factchecker/claims.jsonl'
    file_documents_path = 'assets/templates/factchecker/documents.jsonl'

    # Check if the file exists
    if os.path.exists(file_claims_path) and os.path.exists(file_documents_path):
        # Create a ZIP file in memory
        from io import BytesIO
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Define the name of the file within the ZIP archive
            zip_path = os.path.basename(file_claims_path) # 'claims.jsonl'
            # Add file to the ZIP file
            zf.write(file_claims_path, arcname=zip_path)

            zip_path = os.path.basename(file_documents_path) # 'documents.jsonl'
            # Add file to the ZIP file
            zf.write(file_documents_path, arcname=zip_path)
        
        # Reset pointer to start of the memory file
        memory_file.seek(0)

        # Create a download button and the file will be downloaded when clicked
        btn = st.download_button(
            label="Download",
            data=memory_file,
            file_name="benchmark.zip",
            mime="application/zip"
        )
    else:
        st.error("File not found.")

    # Display the instructions
    st.write("Upload the FactChecker responses as a JSON file below to evaluate the factuality.")

    # Upload the model output
    uploaded_file = st.file_uploader("Upload", type=["csv"], label_visibility="collapsed")

    # Check if the file is uploaded
    if uploaded_file is None:
        st.info("Please upload a CSV file.")
        return
    
    # Check if the file is a CSV file
    if uploaded_file.type != "text/csv":
        st.error("Invalid file format. Please upload a CSV file.")
        return
    
    # Read the CSV file
    uploaded_data = pd.read_csv(uploaded_file)

    def update_first_name():
        st.session_state.first_name = st.session_state.input_first_name

    def update_last_name():
        st.session_state.last_name = st.session_state.input_last_name

    def update_email():
        st.session_state.email = st.session_state.input_email

    def update_organization():
        st.session_state.organization = st.session_state.input_organization

    def update_factchecker():
        st.session_state.factchecker = st.session_state.input_factchecker

    def update_include_in_leaderboard():
        st.session_state.include_in_leaderboard = st.session_state.input_include_in_leaderboard

    # Initialize the session state
    if "disabled" not in st.session_state:
        st.session_state.disabled = False

    # Display instructions
    st.write("Please provide the following information to be included in the leaderboard.")

    # Create text inputs to enter the user information
    st.session_state.id = uuid.uuid4().hex
    st.text_input("First Name", key="input_first_name", on_change=update_first_name)
    st.text_input("Last Name", key="input_last_name", on_change=update_last_name)
    st.text_input("Email", key="input_email", on_change=update_email)
    st.text_input("FactChecker Name", key="input_factchecker", on_change=update_factchecker)
    st.text_input("Organization (Optional)", key="input_organization", on_change=update_organization)

    st.checkbox("Please check this box if you want your LLM to be included in the leaderboard.", 
                key="input_include_in_leaderboard", 
                on_change=update_include_in_leaderboard)

    # Create a submit button
    submit_button = st.button(
        "Submit",
        on_click=lambda: st.session_state.update({"disabled": True}),
        disabled=not (st.session_state.get('input_first_name', '') and
                    st.session_state.get('input_last_name', '') and
                    st.session_state.get('input_email', '') and
                    st.session_state.get('input_factchecker', '')) or st.session_state.disabled
    )

    if submit_button:
        # Create a directory to store the uploaded data
        os.makedirs(f"../eval_results/factchecker/{st.session_state.id}/input/")

        # Save the uploaded data locally
        with open(f"../eval_results/factchecker/{st.session_state.id}/input/response.csv", "w") as file:
            uploaded_data.to_csv(file, index=False)

        # Display a success message
        st.success("User information saved successfully.")

        # Evaluate the FactChecker
        _ = factcheck_api.evaluate_factchecker(st.session_state.id)

        # Create a path to the result file
        report_path = f"../eval_results/factchecker/{st.session_state.id}/results.json"

        # Display a waiting message
        with st.spinner("Checking factuality of the FactCheker..."):
            print("Checking factuality of the FactCheker...")

            # Keep checking if the file exists
            while not os.path.exists(report_path):
                pass

        # Display the evaluation report
        st.write("Evaluation report:")

        # Load the evaluation report
        if os.path.exists(report_path):
            with open(report_path, "rb") as file:
                report = json.load(file)
                st.write(report)
        else:
            st.error("File not found.")

        # Save information to a JSON file
        if st.session_state.disabled:
            if st.session_state.include_in_leaderboard:
                data = {
                    "ID": st.session_state.id,
                    "Name": f"{st.session_state.first_name} {st.session_state.last_name}",
                    "Email": st.session_state.email,
                    "Organization": st.session_state.organization,
                    "FactChecker": st.session_state.factchecker,
                    "Knowledge Source": "FactChecker",
                    "F1": 0.0,
                    "Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0
                }

                # Append the data to the CSV file
                save_leaderboard(data, "factchecker")
    