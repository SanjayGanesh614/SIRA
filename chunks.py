import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,       # max chunk length (tokens ≈ characters / 4)
    chunk_overlap=200,     # overlap between chunks to preserve context
    separators=["\n\n", "\n", ".", " ", ""],  # preferred breakpoints
)

def get_all_chunks(input_path="argo_semantic_summary_1.csv", user_id=None):
# Load your enriched CSV
    df = pd.read_csv(input_path)

    # Group by source file
    grouped = df.groupby("source_file")
    # print(grouped.head())
    # Build chunks
    ids=[]
    summary_list=[]
    metadata_list=[]
    for src, group in grouped:
        # Metadata from first row (since same file = same profile)
        lat = group["latitude"].iloc[0]
        lon = group["longitude"].iloc[0]
        region = group["region"].iloc[0]
        year = group["year"].iloc[0]
        month = group["month"].iloc[0]
        day = group["day"].iloc[0]

        # Join all semantic sentences
        summary = "\n\n".join(group["summary"].tolist())
        chunks = text_splitter.split_text(summary)
        if user_id:
            for i,chunk in enumerate(chunks):
                chunk_id = f"{src}_part{i+1}"
                summary_list.append(chunk)
                metadata_list.append({
                    "source_file": str(src),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "region": str(region),
                    "year": int(year),
                    "month": int(month),
                    "day": int(day),
                    "n_profiles": int(len(group)),
                    "user_id": str(user_id),
                })
                ids.append(chunk_id)
        else:
            for i,chunk in enumerate(chunks):
                chunk_id = f"{src}_part{i+1}"
                summary_list.append(chunk)
                metadata_list.append({
                    "source_file": str(src),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "region": str(region),
                    "year": int(year),
                    "month": int(month),
                    "day": int(day),
                    "n_profiles": int(len(group)),
                })
                ids.append(chunk_id)  
    return summary_list, metadata_list, ids

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# Embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",  # cheap & fast; use "large" for accuracy
    api_key="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"  # 🔒 replace with real key
)

# Chroma persistent vectorstore
persist_directory = r"C:\Users\vijay\Documents\SANJAY\sih\chroma_db"
vectorstore = Chroma(
    collection_name="argo_profiles",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)
vectorstore_single = Chroma(
    collection_name="argo_profiles_single_user",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)
def run_embeddings_in_batches(texts:list, metadatas:list, ids:list, batch_size=100,user_id=None):
    vectorstore_to_use=vectorstore_single if user_id else vectorstore
    for i in range(0, len(texts), batch_size):
        vectorstore_to_use.add_texts(
            texts=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )
    print(f"Stored {len(texts)} profiles in ChromaDB")

def save_embeddings(input_path="argo_semantic_summary_1.csv", user_id=None):
    summary_list, metadata_list, ids = get_all_chunks(input_path=input_path, user_id=user_id)
    run_embeddings_in_batches(summary_list, metadata_list, ids, batch_size=100, user_id=user_id)
# run_embeddings_in_batches(summary_list, metadata_list, ids, batch_size=100)


from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Model-agnostic interface
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

prompt_template = """
You are an expert in oceanography and have extensive knowledge of Argo profiling float data. 
Use the retrieved Argo data to answer the question accurately.
you must answer in detail and provide specific data points from the provided data.
You are a RAG chat app, so do not make up answers if the data is not present.

Data: {data}

Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["data", "question"],
    template=prompt_template
)

# -----------------------------

def answer_question(question, vectorstore, llm, top_k=10,user_id=None):
    vectorstore_to_use=vectorstore_single if user_id else vectorstore
    # Retrieve top_k relevant chunks from Chroma
    if user_id:
        docs = vectorstore_to_use.similarity_search(question,k=top_k,filter={"user_id": user_id})
    else:
        docs = vectorstore_to_use.similarity_search(question, k=top_k)
    combined_data = "\n\n".join([doc.page_content for doc in docs])

    # Prepare prompt
    final_prompt = prompt.invoke({
        "data": combined_data,
        "question": question
    })

    # Call the LLM directly
    response = llm.invoke(final_prompt)
    return response, docs
# -----------------------------
def answer_query(question,user_id=None):
    response, source_docs = answer_question(question, vectorstore, llm, top_k=10,user_id=user_id)
    return response.content
    