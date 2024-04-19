import os
import json
import pandas as pd
import streamlit as st

from api.service import FactCheckAPI

def save_leaderboard(data: dict, type: str):
    """
    This function appends the data to the existing leaderboard.
    """

    # Turn the data into a DataFrame
    df_data = pd.DataFrame([data])

    # Append the data to the existing leaderboard
    df_leaderboard = pd.read_csv(f"assets/leaderboards/{type}_leaderboard.csv")
   
    # Append the data to the existing leaderboard
    df_new_leaderboard = pd.concat([df_leaderboard, df_data], ignore_index=True)

    # Save the new leaderboard
    df_new_leaderboard.to_csv(f"assets/leaderboards/{type}_leaderboard.csv", index=False)

def update_llm_leaderboard():
    """
    This function checks if any IN_PROGRESS evaluations are now COMPLETED and updates the leaderboard.
    """

    # Read the leaderboard data from the specified file
    try:
        df_leaderboard = pd.read_csv(f"assets/leaderboards/llm_leaderboard.csv")
        
        # Check if there are any IN_PROGRESS results in Factual Error Rate column
        in_progress = df_leaderboard[df_leaderboard["Factual Error Rate"] == "IN_PROGRESS"]

        # Get the IDs of the IN_PROGRESS evaluations
        in_progress_ids = in_progress["ID"].tolist()

        # Check if the report is now available
        for id in in_progress_ids:
            # {"overall": 0.7441666666666666}
            report_path = f"../eval_results/llm/{id}/intermediate_results/overall_score.json"
            print(report_path)
            if os.path.exists(report_path):
                # Update the leaderboard
                with open(report_path, "r") as file:
                    report = json.load(file)
                    score = report["overall"]

                # Update the leaderboard
                df_leaderboard.loc[df_leaderboard["ID"] == id, "Factual Error Rate"] = score
                    
        # Save the updated leaderboard
        df_leaderboard.to_csv("assets/leaderboards/llm_leaderboard.csv", index=False)

    except FileNotFoundError:
        st.error("Leaderboard file not found. Please ensure the file path is correct.")

def leaderboards(factcheck_api: FactCheckAPI):
    """
    This function creates a Streamlit app to display the leaderboards.
    """
    # Allow the user to refresh the leaderboard
    if st.button("Refresh Leaderboard", use_container_width=True):
        update_llm_leaderboard()

    # Use Markdown with HTML to center the header
    st.markdown("<h3 style='text-align: center;'>LLM Factuality Leaderboard</h1>", unsafe_allow_html=True)

    # Read the leaderboard data from the specified file
    try:
        # Check if the LLM leaderboard file exists
        if os.path.exists("assets/leaderboards/llm_leaderboard.csv"):
            # Update the leaderboard
            update_llm_leaderboard()
           
        df_leaderboard = pd.read_csv("assets/leaderboards/llm_leaderboard.csv")
        
        # Shorten all IDs to 8 characters
        df_leaderboard["ID"] = df_leaderboard["ID"].str[:8]

        # Display the leaderboard using st.table
        st.table(df_leaderboard)

    except FileNotFoundError:
        st.error("Leaderboard file not found. Please ensure the file path is correct.")

    # Use Markdown with HTML to center the header
    st.markdown("<h3 style='text-align: center;'>FactChecker Factuality Leaderboard</h1>", unsafe_allow_html=True)

    # Read the leaderboard data from the specified file
    try:
        df_leaderboard = pd.read_csv("assets/leaderboards/factchecker_leaderboard.csv")
        
        # Shorten all IDs to 8 characters
        df_leaderboard["ID"] = df_leaderboard["ID"].str[:8]

        # Display the leaderboard using st.table
        st.table(df_leaderboard)

    except FileNotFoundError:
        st.error("Leaderboard file not found. Please ensure the file path is correct.")
