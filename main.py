from fastapi import FastAPI, Request,HTTPException,File,UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import pandas as pd

from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import secrets
# Import your existing logic
from dbagent import initialize_db
from text_embeddings import anser_question
from integration import sql_qa, netcdf_qa


from integration import integration, netcdf_qa, sql_qa

app = FastAPI(title="Multi-Model Chat API")

# CORS must come first!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://sira-frontend-our-data.vercel.app","https://sira-frontend-user-data.vercel.app",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session after CORS
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", secrets.token_hex(16)),
    same_site="none",   # allow cross-site cookies
    https_only=True,    # only over HTTPS
    max_age=3600,
)

@app.middleware("http")
async def update_session_timeout(request: Request, call_next):
    response = await call_next(request)

    # ✅ Only set cookie if request has session (avoid OPTIONS errors)
    if "session" in request.session and "session" in request.cookies:
        response.set_cookie(
            key="session",
            value=request.cookies.get("session", ""),
            max_age=3600,
            httponly=True,
            samesite="none",
            # secure=True   # enable once HTTPS everywhere
        )

    return response


def query_facts(question: str) -> str:
    return anser_question(question)

def query_netcdf(question: str) -> str:
    return netcdf_qa(question)

def query_db(question: str) -> pd.DataFrame:
    conn = initialize_db()
    data = sql_qa(question, conn)
    return pd.DataFrame(data)

class QuestionRequest(BaseModel):
    question: str

@app.post("/chat/facts")
async def chat_facts(req: QuestionRequest):
    answer = await run_in_threadpool(query_facts, req.question)
    return {"type": "text", "answer": answer}

@app.post("/chat/netcdf")
async def chat_netcdf(req: QuestionRequest):
    answer = await run_in_threadpool(query_netcdf, req.question)
    return {"type": "text", "answer": answer}

@app.post("/chat/db")
async def chat_db(req: QuestionRequest):
    df = await run_in_threadpool(query_db, req.question)
    return {"type": "table", "answer": df.to_dict(orient="records")}

@app.get("/")
def root():
    return {"message": "Multi-Model Chat API is running!"}



# user profile based chatting


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------
# Helper functions
# -----------------------
def get_user_paths(user_id: str):
    """Get or create user folders for uploads and processed files"""
    user_folder = os.path.join(UPLOAD_DIR, user_id)
    argo_folder = os.path.join(user_folder, "argo_nc_files")
    os.makedirs(user_folder, exist_ok=True)
    os.makedirs(argo_folder, exist_ok=True)
    return user_folder, argo_folder

# -----------------------
# Session / UUID
# -----------------------
@app.get("/user/session/new")
def create_session(request: Request):
    """Create a new session and store user_id"""
    if "user_id" not in request.session:
        request.session["user_id"] = str(uuid.uuid4())
    return {"user_id": request.session["user_id"]}

def get_user_id(request: Request) -> str:
    user_id = request.session.get("user_id")
    if not user_id:
        raise JSONResponse(status_code=400, content={"error": "No session found. Create session first."})
    return user_id

# -----------------------
# File upload
# -----------------------
@app.post("/user/upload")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    user_id = get_user_id(request)
    user_folder, argo_folder = get_user_paths(user_id)
    saved_paths = []

    for file in files[:5]:
        file_path = os.path.join(argo_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_paths.append(file_path)

    try:
        # Run integration
        semantic_output_path, _ = await run_in_threadpool(
            integration, user_id, user_folder, argo_folder
        )
        
        # Store connection in memory per user if needed (optional)
        return {"status": "ok", "files": saved_paths}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------
# Dataset upload for model testing
# -----------------------
@app.post("/user/upload/dataset")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    """Upload a CSV dataset for model testing"""
    user_id = get_user_id(request)
    user_folder, _ = get_user_paths(user_id)
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        return JSONResponse(status_code=400, content={"error": "Only CSV files are supported"})
    
    # Save the dataset
    dataset_path = os.path.join(user_folder, f"user_dataset_{file.filename}")
    with open(dataset_path, "wb") as f:
        f.write(await file.read())
    
    return {"status": "ok", "dataset_path": dataset_path, "filename": file.filename}

# -----------------------
# Model testing with user dataset
# -----------------------
@app.post("/user/test/model")
async def test_model_with_dataset(request: Request, dataset_filename: str):
    """Test the anomaly detection model with user's uploaded dataset"""
    user_id = get_user_id(request)
    user_folder, _ = get_user_paths(user_id)
    dataset_path = os.path.join(user_folder, f"user_dataset_{dataset_filename}")
    
    if not os.path.exists(dataset_path):
        return JSONResponse(status_code=404, content={"error": "Dataset not found"})
    
    try:
        # Import the feature detection function
        # from feature_detect import run_feature_detection
        
        # # Run model testing with user's dataset
        # result = await run_in_threadpool(
        #     run_feature_detection, 
        #     train=False, 
        #     input_csv=dataset_path,
        #     is_user_dataset=True
        # )
        result = "Model testing logic not implemented yet."
        
        return {"status": "ok", "result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------
# Chat endpoints
# -----------------------

@app.post("/user/chat/netcdf")
async def chat_netcdf(request: Request, req: QuestionRequest):
    user_id = get_user_id(request)
    answer = await run_in_threadpool(netcdf_qa, req.question, user_id)
    return {"type": "text", "answer": answer}

@app.post("/user/chat/db")
async def chat_db(request: Request, req: QuestionRequest):
    user_id = get_user_id(request)
    user_folder, _ = get_user_paths(user_id)
    semantic_csv = os.path.join(user_folder, "argo_semantic_summary_1.csv")
    conn = await run_in_threadpool(
        initialize_db,
        input_path=semantic_csv,
        db_path=user_folder,
        user_id=user_id,
    )
    df = await run_in_threadpool(sql_qa, req.question, conn)
    return {"type": "table", "answer": pd.DataFrame(df).to_dict(orient="records")}

# -----------------------
# Download processed CSV
# -----------------------
@app.get("/user/download")
async def download_csv(request: Request):
    user_id = get_user_id(request)
    user_folder, _ = get_user_paths(user_id)
    semantic_csv = os.path.join(user_folder, "argo_semantic_summary_1.csv")
    if os.path.exists(semantic_csv):
        return FileResponse(
            semantic_csv, media_type="text/csv", filename=os.path.basename(semantic_csv)
        )
    return JSONResponse(status_code=404, content={"error": "CSV not found"})


# this is the place wheree visulaization information will be supplied

import pandas as pd

# Assuming `app` already exists from previous code

CSV_FILE = "argo_profiles_final.csv"

try:
    df_profiles = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df_profiles = None

if df_profiles is not None:
    # Convert pressure_bin into numeric midpoint
    def get_midpoint(bin_str):
        bin_str = bin_str.strip("()[]")
        low, high = bin_str.split(",")
        return (float(low) + float(high)) / 2

    df_profiles["pressure_mid"] = df_profiles["pressure_bin"].apply(get_midpoint)

    # Precompute aggregated dataframe
    agg_df_profiles = df_profiles.groupby(["region", "pressure_mid"]).agg({
        "temp_adjusted": "mean",
        "psal_adjusted": "mean"
    }).reset_index()
else:
    agg_df_profiles = pd.DataFrame()

def graph_user(request:str):
    user_id = get_user_id(request)
    user_folder, argo_folder = get_user_paths(user_id)
    user_csv = os.path.join(user_folder, "argo_profiles_final.csv")
    try:
        df_profiles = pd.read_csv(user_csv)
    except FileNotFoundError:
        df_profiles = None

    if df_profiles is not None:
        # Convert pressure_bin into numeric midpoint
        def get_midpoint(bin_str):
            bin_str = bin_str.strip("()[]")
            low, high = bin_str.split(",")
            return (float(low) + float(high)) / 2

        df_profiles["pressure_mid"] = df_profiles["pressure_bin"].apply(get_midpoint)

        # Precompute aggregated dataframe
        agg_df_profiles = df_profiles.groupby(["region", "pressure_mid"]).agg({
            "temp_adjusted": "mean",
            "psal_adjusted": "mean"
        }).reset_index()
    else:
        agg_df_profiles = pd.DataFrame()
    return agg_df_profiles




@app.get("/profiles/regions")
def list_regions():
    if agg_df_profiles.empty:
        return JSONResponse({"regions": []})
    regions = agg_df_profiles["region"].unique().tolist()
    return {"regions": regions}

@app.get("/profiles/data")
def get_profile_data(region: str):
    if agg_df_profiles.empty:
        return JSONResponse({"error": "No data available"}, status_code=404)

    subset = agg_df_profiles[agg_df_profiles["region"] == region]
    if subset.empty:
        return JSONResponse({"error": "Region not found"}, status_code=404)

    # Return data points as JSON
    data = {
        "pressure_mid": subset["pressure_mid"].tolist(),
        "temperature": subset["temp_adjusted"].tolist(),
        "salinity": subset["psal_adjusted"].tolist()
    }

    return {"region": region, "data": data}


@app.get("/users/profiles/regions")
def users_list_regions(request: Request):
    user_agg_df_profiles = graph_user(request)
    if user_agg_df_profiles.empty:
        return JSONResponse({"regions": []})
    regions = user_agg_df_profiles["region"].unique().tolist()
    return {"regions": regions}

@app.get("/users/profiles/data")
def users_get_profile_data(request: Request,region: str):
    user_agg_df_profiles = graph_user(request)
    # print(user_agg_df_profiles.head())
    if user_agg_df_profiles.empty:
        return JSONResponse({"error": "No data available"}, status_code=404)

    subset = user_agg_df_profiles[user_agg_df_profiles["region"] == region]
    if subset.empty:
        return JSONResponse({"error": "Region not found"}, status_code=404)

    # Return data points as JSON
    data = {
        "pressure_mid": subset["pressure_mid"].tolist(),
        "temperature": subset["temp_adjusted"].tolist(),
        "salinity": subset["psal_adjusted"].tolist()
    }

    return {"region": region, "data": data}
