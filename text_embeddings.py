from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import os

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",  # cheap & fast; use "large" for accuracy
    api_key=OPENAI_API_KEY  # 🔒 replace with real key
)

# Chroma persistent vectorstore
persist_directory = "./chroma_db"
vectorstore = Chroma(
    collection_name="buoy_data",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

def run_embeddings():
    # Configure the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # max chunk length (tokens ≈ characters / 4)
        chunk_overlap=100,     # overlap between chunks to preserve context
        separators=["\n\n", "\n", ".", " ", ""],  # preferred breakpoints
    )

    # Load your enriched CSV
    with open("Buoys_data.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    chunks = text_splitter.split_text(full_text)

    vectorstore.add_texts(
        texts=chunks,
        ids=[f"chunk_{i+1}" for i in range(len(chunks))],
    )
run_embeddings()
print("embeddings done")

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

def answer_question(question, vectorstore, llm, top_k=3):
    # Retrieve top_k relevant chunks from Chroma
    docs = vectorstore.similarity_search(question, k=top_k)
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
# Ask a question
# -----------------------------
# question = "What are buyos? "
# response, source_docs = answer_question(question, vectorstore, llm, top_k=4)

# print("Answer:\n", response)

def anser_question(question):
    response, source_docs = answer_question(question, vectorstore, llm, top_k=4)
    return response.content