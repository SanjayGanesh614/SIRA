from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# Embedding model
from feature_detect import run_feature_detection

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

def answer_question(question, llm):
    # Retrieve top_k relevant chunks from Chroma
    docs=run_feature_detection()

    # Prepare prompt
    final_prompt = prompt.invoke({
        "data": docs,
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
    response,src= answer_question(question, llm)
    return response.content