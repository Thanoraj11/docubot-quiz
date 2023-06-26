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

from llama_index import Prompt

# loader = PDFReader()

openai.api_key = os.getenv("OPENAI_API_KEY")

question_rules = "Question Rule 1: Do not ask questions about very small details such as specific numbers.\nQuestion Rule 2: When you run out of questions, let me know this and then quiz me on the questions I've already responded to. Present the questions one at a time in random order."
answer_rules = "Feedback Rule 1: If my response is correct, you give feedback and then move on to the next question (Step 1 again).\nFeedback Rule 2: If the answer is incorrect or only partly correct, you give feedback that helps me move toward the correct answer. After your feedback, say something in the style of 'try again'.\nFeedback Rule 3: It is very important that you dont give away the correct answer in the feedback to partly correct or incorrect responses."
template = ("We have provided context information below. \n"
    "---------------------\n"
    f"{question_rules}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
    )

QA_TEMPLATE = Prompt(template)


def process_pdf(uploaded_file):
    loader = PDFReader()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        documents = loader.load_data(file=Path(temp_file.name))
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-davinci-003", max_tokens=1000))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
    if "index" not in st.session_state:
        index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
        query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE)
        st.session_state.index = query_engine
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
