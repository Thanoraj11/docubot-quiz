import streamlit as st
import openai
import os
from llama_index import (
    GPTVectorStoreIndex, Document, SimpleDirectoryReader,
    QuestionAnswerPrompt, LLMPredictor, ServiceContext
)

from llama_index import (GPTVectorStoreIndex, Document, SimpleDirectoryReader,QuestionAnswerPrompt, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage)

from tempfile import NamedTemporaryFile
from llama_index import download_loader
import openai
import os
from pathlib import Path
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from random import randint
import random
import string

openai.api_key = os.getenv("OPENAI_API_KEY")

ENGINE = "text-davinci-003"
MAX_TOKENS = 100

if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'conversations' not in st.session_state:
    st.session_state.conversations = []
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""

def generate_answer_pdf(index_path, query_prompt):
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    resp = query_engine.query(query_prompt)
    return resp.response

def grade_answer(question, user_answer):
    prompt = f"""
    The question was: {question}
    The user answered: {user_answer}
    return a True/False if answer is correct/wrong and also provide a comprehensive feedback to improve the answer.
    Resposne should be returned as a plain string in the following format,
    "True: Feedback."
    """
    
    response = openai.Completion.create(
        engine=ENGINE,
        prompt=prompt,
        max_tokens=MAX_TOKENS
    )
    
    output_text = response.choices[0].text.strip()
    bool_value, feedback = output_text.split(':', 1)
    bool_value = bool_value.strip().lower() == 'true'
    feedback = feedback.strip()

    return bool_value, feedback

def generate_question(keyword):
    prompt = f"Generate a question about the following topic : {keyword}"
    response = openai.Completion.create(engine=ENGINE, prompt=prompt, max_tokens=MAX_TOKENS)
    return response.choices[0].text.strip()

DATA_DIR = "data"
index_filenames_pdf = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

index_file = st.selectbox("Select a PDF file to load:", index_filenames_pdf)

st.title("Quizbot Application")
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if st.button("Start Learning Session"):
    index_path = os.path.join(DATA_DIR, index_file)
    query_prompt = "Generate 10 important areas that are covered in this book"
    vector_resp = generate_answer_pdf(index_path, query_prompt)
    
    st.session_state.keywords = vector_resp.split('\n')
    st.sidebar.write(st.session_state.keywords)
    st.session_state.counter = 2
    st.session_state.score = 0
    st.session_state.conversations = []
    st.session_state.current_answer = ""

st.write("Your current score:", st.session_state.score)

if st.session_state.counter < len(st.session_state.keywords):
    keyword = st.session_state.keywords[st.session_state.counter]
    question = generate_question(keyword)
    st.session_state.conversations.append({
        'keyword': keyword,
        'question': question,
        'user_answer': None,
        'feedback': None,
    })

if len(st.session_state.conversations) > 0:
    current_conversation = st.session_state.conversations[-1]
    st.session_state.current_answer = st.text_area("Your answer:", st.session_state.current_answer)

    if st.button('Submit Answer'):
        user_answer = st.session_state.current_answer

        # Grade the answer
        correct, feedback = grade_answer(current_conversation['question'], user_answer)
        st.session_state.score += int(correct)
        current_conversation['user_answer'] = user_answer
        current_conversation['feedback'] = feedback

        # Move to the next question
        st.session_state.counter += 1
        st.session_state.current_answer = ""

for i, conversation in enumerate(reversed(st.session_state.conversations), start=1):
    with st.expander(f"Thread {len(st.session_state.conversations)-i+1}", expanded=(i==1)):
        st.write(f"**Question**\n\n", f"\n{conversation['question']}")
        st.write(f"**Your Answer**\n\n", f"\n{conversation['user_answer']}")
        st.write(f"**Feedback**\n\n", f"\n{conversation['feedback']}")
