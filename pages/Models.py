import streamlit as st
import pandas as pd
from dbagent import query_db_q,initialize_db
from text_embeddings import anser_question

from integration import sql_qa,netcdf_qa

# Dummy functions for each model
def query_facts(question: str) -> str:
    return anser_question(question)

def query_netcdf(question: str) -> str:
    return netcdf_qa(question)

def query_db(question: str) -> pd.DataFrame:
    conn=initialize_db()
    data = sql_qa(question,conn)
    return pd.DataFrame(data)

# Streamlit app
st.set_page_config(page_title="Multi-Model Chat", layout="wide")

st.title("💬 Multi-Model Chat App")

# Sidebar to pick model
model_choice = st.sidebar.radio(
    "Choose Model:",
    ["Facts Model", "NetCDF Model", "DB Model"]
)

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Append user query
    st.session_state.chat_history.append(("user", user_input))

    # Process based on model
    if model_choice == "Facts Model":
        answer = query_facts(user_input)
        st.session_state.chat_history.append(("bot", answer))

    elif model_choice == "NetCDF Model":
        answer = query_netcdf(user_input)
        st.session_state.chat_history.append(("bot", answer))

    elif model_choice == "DB Model":
        df = query_db(user_input)
        st.session_state.chat_history.append(("bot_df", df))

# Display chat
for role, content in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(content)

    elif role == "bot":
        with st.chat_message("assistant"):
            st.write(content)

    elif role == "bot_df":
        with st.chat_message("assistant"):
            st.write("Here’s the DataFrame:")
            st.dataframe(content)
