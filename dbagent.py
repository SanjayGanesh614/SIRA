import sqlite3
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ----------------------------
# Step 1: Load CSV into SQLite
# ----------------------------
def initialize_db(input_path="argo_semantic_summary_1.csv", db_path="argo.db",user_id=None):
    if not user_id:
        conn = sqlite3.connect(db_path,check_same_thread=False)
    else:
        cocat_dir=os.path.join(db_path,"argo.db") #db path mentions the home directoryof user in the case of per user 
        df = pd.read_csv(input_path)
        conn = sqlite3.connect(cocat_dir,check_same_thread=False)
        df.to_sql("argo_profiles", conn, if_exists="replace", index=False)
    return conn
# ----------------------------
# Step 2: Schema info
# ----------------------------
schema = """
Table: argo_profiles
Columns:
- pressure_bin (TEXT, e.g. '(500, 550]') [values from 0 to 2000 dbar, in 50 dbar increments]
- temp_adjusted (REAL, temperature in °C)
- psal_adjusted (REAL, salinity in PSU)
- juld (TEXT, datetime string, original timestamp)
- year (INTEGER, year extracted from juld)
- month (INTEGER, month extracted from juld)
- day (INTEGER, day extracted from juld)
- time (TEXT, time of day extracted from juld, e.g. '11:45:17')
- latitude (REAL, in degrees)
- longitude (REAL, in degrees)
- source_file (TEXT, original NetCDF filename, e.g. 'nodc_R1901302_273.nc')
- region (TEXT, one of the predefined ocean regions below)
- salinity_mean (REAL, mean salinity for the given region & pressure_bin)
- salinity_std (REAL, std deviation of salinity for the given region & pressure_bin)
- temp_mean (REAL, mean temperature for the given region & pressure_bin)
- temp_std (REAL, std deviation of temperature for the given region & pressure_bin)
- salinity_z (REAL, z-score of psal_adjusted compared to salinity_mean/std)
- temp_z (REAL, z-score of temp_adjusted compared to temp_mean/std)
- salinity_flag (TEXT, categorical anomaly flag: 'low', 'normal', 'high')
- temp_flag (TEXT, categorical anomaly flag: 'low', 'normal', 'high')
- summary (TEXT, human-readable semantic description of the observation)

Regions defined as:
Bay of Bengal
Arabian Sea
Equatorial Indian Ocean
Central Indian Ocean
Western Indian Ocean
Southwest Indian Ocean
Southeast Indian Ocean
Southern Ocean
Australian Waters
Eastern Indian Ocean
"""

# ----------------------------
# Step 3: Prompt template
# ----------------------------
prompt_template = """
You are an expert SQL query generator. The database is SQLite.

Schema:
{schema}

Task: {question}

Rules:
- Only return a syntactically valid SQL query.
- Do not execute the query.
- Do not explain anything, only output SQL.
"""

prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=prompt_template
)

# ----------------------------
# Step 4: LLM initialization
# ----------------------------
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# ----------------------------
# Step 5: Helper to get SQL & run it
# ----------------------------
def run_question(question: str, schema: str, conn: sqlite3.Connection) -> pd.DataFrame:
    # Prepare the prompt
    formatted_prompt = prompt.format(schema=schema, question=question)

    # Ask the LLM for SQL
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    sql_query = response.content.strip()

    print("Generated SQL:\n", sql_query)

    # Run the query in SQLite
    df_out = pd.read_sql_query(sql_query, conn)
    return df_out

def query_db_q(question: str,conn) -> pd.DataFrame:
    return run_question(question, schema, conn)
