import streamlit as st
import os
import uuid
import pandas as pd

# Your processing and QA functions
from integration import integration, netcdf_qa, sql_qa
from text_embeddings import anser_question
from dbagent import initialize_db

# ----------------------
# Model query functions
# ----------------------
def query_facts(question: str) -> str:
    return anser_question(question)

def query_netcdf(question: str,user_id) -> str:
    return netcdf_qa(question,user_id)

def query_db(question: str,user_id,conn) -> pd.DataFrame:
    data = sql_qa(question, conn)
    return pd.DataFrame(data)

# ----------------------
# App setup
# ----------------------
st.set_page_config(page_title="Multi-Model NetCDF Chat", layout="wide")
st.title("📂 Upload & Chat with NetCDF + Multi-Model Support")

# --- Unique user/session folder ---
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

user_folder = os.path.join("uploads", st.session_state["user_id"])
argo_folder = os.path.join(user_folder, "argo_nc_files")
os.makedirs(user_folder, exist_ok=True)
os.makedirs(argo_folder, exist_ok=True)

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Upload up to 5 NetCDF files", 
    type=["nc"], 
    accept_multiple_files=True
)

if uploaded_files and "processed" not in st.session_state:
    saved_paths = []

    for uploaded_file in uploaded_files[:5]:  # ✅ limit to 5 files
        file_path = os.path.join(argo_folder, uploaded_file.name)

        # Save file to session folder
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_paths.append(file_path)

    st.success(f"✅ Saved {len(saved_paths)} files to {argo_folder}")

    try:
        semantic_output_path, conn = integration(
            st.session_state["user_id"], user_folder, argo_folder)
        conn = initialize_db(
            input_path=semantic_output_path,
            db_path=user_folder,
            user_id=st.session_state["user_id"],)
        st.session_state["files_processed"] = True
        st.session_state["processed"] = True  # ✅ mark as processed
        st.session_state["conn"] = conn     # ✅ keep DB connection
        
    except Exception as e:
        st.error(f"Could not process file: {e}")
        st.session_state["files_processed"] = False

# ----------------------
# Chat section
# ----------------------
if st.session_state.get("files_processed", False):
    st.subheader("💬 Chat with your data")

    # Sidebar to pick model
    model_choice = st.sidebar.radio(
        "Choose Model:",
        ["Facts Model", "NetCDF Model", "DB Model"]
    )
    with st.sidebar:
            st.success("✅ Processing complete!")
            st.subheader("📥 Download Processed Files")
            semantic_csv = os.path.join(user_folder, "argo_semantic_summary_1.csv")
            if os.path.exists(semantic_csv):
                with open(semantic_csv, "rb") as f:
                    st.download_button(
                            label=f"⬇️ Download Semantic Summary CSV",
                            data=f,
                            file_name=os.path.basename(semantic_csv),
                            mime="text/csv",
                            key=semantic_csv  # unique key required in sidebar
                        )

    # Maintain chat history for all models
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask a question...")

    if user_input:
        # Append user query
        st.session_state.chat_history.append(("user", user_input))

        if model_choice == "Facts Model":
            answer = query_facts(user_input)
            st.session_state.chat_history.append(("bot", answer))

        elif model_choice == "NetCDF Model":
            answer = query_netcdf(user_input,user_id=st.session_state["user_id"])
            st.session_state.chat_history.append(("bot", answer))

        elif model_choice == "DB Model":
            df = query_db(user_input,user_id=st.session_state["user_id"],conn=st.session_state["conn"])
            st.session_state.chat_history.append(("bot_df", df))
            pass

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
else:
    st.info("⬆️ Please upload and process your NetCDF file before starting the chat.")
