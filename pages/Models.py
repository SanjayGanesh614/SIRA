import streamlit as st
import pandas as pd
from dbagent import query_db_q,initialize_db
from text_embeddings import anser_question
import requests
import json

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

# Add navigation tabs
tab1, tab2 = st.tabs(["🤖 Chat Models", "🔬 Dataset Testing"])

with tab1:
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
                st.write("Here's the DataFrame:")
                st.dataframe(content)

with tab2:
    st.header("🔬 Dataset Testing")
    st.markdown("Upload your own oceanographic dataset and test it against our trained models")
    
    # Session state management
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "uploaded_datasets" not in st.session_state:
        st.session_state.uploaded_datasets = []

    # API base URL
    API_BASE_URL = "http://localhost:8000"

    def create_session():
        """Create a new session with the API"""
        try:
            response = requests.get(f"{API_BASE_URL}/user/session/new")
            if response.status_code == 200:
                data = response.json()
                return data.get("user_id")
            return None
        except Exception as e:
            st.error(f"Failed to create session: {str(e)}")
            return None

    def upload_dataset(file, user_id):
        """Upload dataset to the API"""
        try:
            files = {"file": (file.name, file.getvalue(), "text/csv")}
            response = requests.post(
                f"{API_BASE_URL}/user/upload/dataset",
                files=files,
                cookies={"session": user_id}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.json().get("error", "Upload failed")}
        except Exception as e:
            return {"error": str(e)}

    def test_model_with_dataset(dataset_filename, user_id):
        """Test model with uploaded dataset"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/user/test/model",
                params={"dataset_filename": dataset_filename},
                cookies={"session": user_id}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.json().get("error", "Testing failed")}
        except Exception as e:
            return {"error": str(e)}

    # Initialize session
    if st.session_state.session_id is None:
        st.session_state.session_id = create_session()

    if st.session_state.session_id is None:
        st.error("Failed to initialize session. Please refresh the page.")
        st.stop()

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📁 Upload Dataset")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with oceanographic data. Required columns: time, latitude, longitude, depth, Temperature, CTD_Salinity, Oxygen_1, CO2, TOC, POC, NO3_plus_NO2, PO4, Silicate"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.success(f"File uploaded: {uploaded_file.name}")
            st.info(f"File size: {uploaded_file.size} bytes")
            
            # Preview the data
            try:
                df_preview = pd.read_csv(uploaded_file)
                st.subheader("📊 Data Preview")
                st.dataframe(df_preview.head(10))
                
                # Check required columns
                required_columns = ['time', 'latitude', 'longitude', 'depth', 'Temperature', 'CTD_Salinity', 'Oxygen_1', 'CO2', 'TOC', 'POC', 'NO3_plus_NO2', 'PO4', 'Silicate']
                missing_columns = [col for col in required_columns if col not in df_preview.columns]
                
                if missing_columns:
                    st.warning(f"⚠️ Missing columns: {missing_columns}")
                else:
                    st.success("✅ All required columns present")
                
                # Upload button
                if st.button("🚀 Upload Dataset", type="primary"):
                    with st.spinner("Uploading dataset..."):
                        result = upload_dataset(uploaded_file, st.session_state.session_id)
                        
                    if "error" in result:
                        st.error(f"Upload failed: {result['error']}")
                    else:
                        st.success("Dataset uploaded successfully!")
                        st.session_state.uploaded_datasets.append({
                            "filename": result["filename"],
                            "path": result["dataset_path"]
                        })
                        st.rerun()
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    with col2:
        st.subheader("🧪 Test Models")
        
        if st.session_state.uploaded_datasets:
            st.write("Available Datasets:")
            
            for i, dataset in enumerate(st.session_state.uploaded_datasets):
                with st.expander(f"Dataset: {dataset['filename']}"):
                    st.write(f"Path: {dataset['path']}")
                    
                    if st.button(f"🔬 Test Model", key=f"test_{i}"):
                        with st.spinner("Running anomaly detection..."):
                            result = test_model_with_dataset(dataset['filename'], st.session_state.session_id)
                        
                        if "error" in result:
                            st.error(f"Testing failed: {result['error']}")
                        else:
                            st.success("Model testing completed!")
                            
                            # Display results
                            st.subheader("📈 Analysis Results")
                            st.text_area("Detailed Report", result["result"], height=400)
        else:
            st.info("No datasets uploaded yet. Upload a dataset first to test the models.")

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Dataset Requirements")
        st.markdown("""
        Your CSV file should contain the following columns:
        
        **Required Columns:**
        - `time`: Timestamp (any format)
        - `latitude`: Latitude coordinate
        - `longitude`: Longitude coordinate  
        - `depth`: Depth in meters
        - `Temperature`: Water temperature
        - `CTD_Salinity`: Salinity measurement
        - `Oxygen_1`: Dissolved oxygen
        - `CO2`: Carbon dioxide
        - `TOC`: Total organic carbon
        - `POC`: Particulate organic carbon
        - `NO3_plus_NO2`: Nitrate + Nitrite
        - `PO4`: Phosphate
        - `Silicate`: Silicate concentration
        
        **Notes:**
        - At least 10 rows of data required
        - Missing values will be filled with median values
        - Non-numeric values will be converted where possible
        """)
        
        st.header("🔧 Model Information")
        st.markdown("""
        **Models Used:**
        - LSTM Autoencoder for anomaly detection
        - XGBoost Classifier for event classification
        
        **Detection Capabilities:**
        - Marine heatwaves
        - Hypoxia events
        - Nutrient blooms
        - Cold water events
        - Deep water anomalies
        """)
