import os
import uuid
import json
import zipfile
import pandas as pd
import streamlit as st

from api.service import FactCheckAPI
from src.leaderboards import save_leaderboard
from src.utils import pad_image

def evaluate_llm(factcheck_api: FactCheckAPI):
    """
    This function creates a Streamlit app to evaluate the factuality of a LLM.
    """
    st.write("This is where you can evaluate the factuality of a LLM.")

    # Display the instructions
    st.write("Download the questions and instructions to evaluate the factuality of a LLM.")

    # Assuming 'question_set.csv' is in the 'templates' folder within the 'assets' directory
    file_path = 'assets/templates/llm/questions.csv'

    # Check if the file exists
    if os.path.exists(file_path):
        # Create a ZIP file in memory
        from io import BytesIO
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # Define the name of the file within the ZIP archive
            zip_path = os.path.basename(file_path)  # 'questions.csv'
            # Add file to the ZIP file
            zf.write(file_path, arcname=zip_path)
        
        # Reset pointer to start of the memory file
        memory_file.seek(0)

        # Create a download button and the file will be downloaded when clicked
        btn = st.download_button(
            label="Download",
            data=memory_file,
            file_name="questions.csv.zip",
            mime="application/zip"
        )
    else:
        st.error("File not found.")

    # Display the instructions
    st.write("Upload the model responses as a JSON file below to evaluate the factuality.")

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

    def update_llm_model():
        st.session_state.llm_model = st.session_state.input_llm_model

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
    st.text_input("LLM Model Name", key="input_llm_model", on_change=update_llm_model)
    st.text_input("Organization (Optional)", key="input_organization", on_change=update_organization)

    # Create a checkbox to include the user in the leaderboard
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
                    st.session_state.get('input_llm_model', '')) or st.session_state.disabled
    )

    if submit_button:
        # Create a directory to store the uploaded data
        os.makedirs(f"../eval_results/llm/{st.session_state.id}/input/")

        # Save the uploaded data locally
        with open(f"../eval_results/llm/{st.session_state.id}/input/response.csv", "w") as file:
            uploaded_data.to_csv(file, index=False)

        # Display a success message
        st.success("User information saved successfully.")

        # Display an information message
        st.info(f"""Please wait while we evaluate the factuality of the LLM.
You will be able to download the evaluation report shortly, if you can wait. The report will also be delivered to your email address.
                
Please note your ID {st.session_state.id}, This will be used to track your evaluation.
If the report is not available, please contact the administrator and provide your ID.""")

        # Save information to a CSV file
        if st.session_state.disabled:
            if st.session_state.include_in_leaderboard:
                    data = {
                        "ID": st.session_state.id,
                        "Name": f"{st.session_state.first_name} {st.session_state.last_name}",
                        "Email": st.session_state.email,
                        "Organization": st.session_state.organization,
                        "LLM Model": st.session_state.llm_model,
                        "Factual Error Rate": "IN_PROGRESS",
                    }

                    # Append the data to the CSV file
                    save_leaderboard(data, "llm")

        # Set api key and configure pipeline
        factcheck_api.pipeline_configure_global({"openai_key": {"value": st.session_state.openai_apikey, "env_name": "OPENAI_API_KEY"}})

        # Evaluate the LLM
        _ = factcheck_api.evaluate_llm(st.session_state.id)

        # Create a path to the result file
        report_path = f"../eval_results/llm/{st.session_state.id}/report/report.pdf"

        # Display a waiting message
        with st.spinner("Checking factuality of the LLM..."):
            print("Checking factuality of the LLM...")

            # Keep checking if the file exists
            while not os.path.exists(report_path):
                pass

        # Display the evaluation report
        st.write("Evaluation report:")
        
        # Load the evaluation report
        if os.path.exists(report_path):
            with open(report_path, "rb") as file:
                report_bytes = file.read()
        else:
            st.error("File not found.")
        
        # Columns for images
        col1, col2, col3, col4 = st.columns(4)

        # Define image paths
        image_paths = [
            "../eval_results/llm/{}/figure/freshqa_piechart.png".format(st.session_state.id),
            "../eval_results/llm/{}/figure/selfaware_cm.png".format(st.session_state.id),
            "../eval_results/llm/{}/figure/selfaware_performance.png".format(st.session_state.id),
            "../eval_results/llm/{}/figure/snowballing_acc.png".format(st.session_state.id)
        ]

        # Display images in columns with specified width
        for i, col in enumerate([col1, col2, col3, col4]):
            with col:
                image = pad_image(image_paths[i])
                if image:
                    st.image(image, use_column_width=True)  # Adjusts image to column width

        # Display the Download button
        st.write("Download the evaluation report.")
        st.download_button(
            label="Download",
            data=report_bytes,
            file_name="evaluation_report.pdf",
            mime="application/pdf"
        )

            
