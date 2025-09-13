import streamlit as st
from llm_event import anser_question
# Example function that generates a report
def generate_report():
    return anser_question("give a detailed report based on the anomality's detected in the ARGO BGC data and also explain the features detected")

# Streamlit page
st.set_page_config(page_title="Report Viewer", layout="wide")
st.title("📄 Report")

# Call the function and display its text
report_text = generate_report()
st.markdown(report_text)
