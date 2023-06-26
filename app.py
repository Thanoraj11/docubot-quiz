import streamlit as st
from llama_index import (
    GPTVectorStoreIndex, Document, SimpleDirectoryReader,
    QuestionAnswerPrompt, LLMPredictor, ServiceContext
)


from langchain import OpenAI
from tempfile import NamedTemporaryFile
from llama_index import download_loader

import openai
import os
from pathlib import Path
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
PDFReader = download_loader("PDFReader")

# loader = PDFReader()

openai.api_key = os.getenv("OPENAI_API_KEY")




def process_pdf(uploaded_file):
    loader = PDFReader()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        documents = loader.load_data(file=Path(temp_file.name))
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-davinci-003", max_tokens=1000))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
    if "index" not in st.session_state:
        index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
        retriever = index.as_retriever(retriever_mode='embedding')
        index = RetrieverQueryEngine(retriever)
        st.session_state.index = index
    # st.session_state.index = index
    return st.session_state.index




uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:

    if "index" not in st.session_state:
        st.session_state.index = process_pdf(uploaded_file)
        st.success("Index created successfully")



query = st.text_input("Enter query prompt")
asl = st.button("Submit")

if asl:
    resp  = st.session_state.index.query(query).response
    st.write(resp)
