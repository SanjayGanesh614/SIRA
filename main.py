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

origins = [
    # "http://localhost:5173",  # React dev server
    "*",
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,  # Explicitly list allowed origins
    allow_credentials = True,  # Allow cookies and credentials
    allow_methods = ["*"],  # Allow all HTTP methods
    allow_headers = ["*"],  # Allow all headers
)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY", secrets.token_hex(16)),
    same_site="none",  # Required for cross-origin cookies
    https_only=True,  # Ensure cookies are sent only over HTTPS
    max_age= 3600,  # Set cookie expiration time (1 hour)
)
@app.middleware("http") 
async def update_session_timeout(request: Request, call_next): 
    response = await call_next(request) 
    if "session" in request.session: 
        # response.set_cookie("session", request.cookies["session"], max_age= 3600, httponly=True, samesite="strict", secure=True) #local
        response.set_cookie(
    key="session",
    value=request.cookies.get("session", ""),  # safer: returns empty string if not present
    max_age=3600,
    httponly=True,
    samesite="none",   # allow same-site requests plus normal form submissions
    # secure=True       
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
        # Initialize DB
        # conn = await run_in_threadpool(
        #     initialize_db,
        #     input_path=semantic_output_path,
        #     db_path=user_folder,
        #     user_id=user_id,
        # )
        # Store connection in memory per user if needed (optional)
        return {"status": "ok", "files": saved_paths}
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
