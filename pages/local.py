import streamlit as st
import openai
import os
from llama_index import (
    GPTVectorStoreIndex, Document, SimpleDirectoryReader,
    QuestionAnswerPrompt, LLMPredictor, ServiceContext
)
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

keywords = ['Python', 'Java', 'JavaScript']

if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'conversations' not in st.session_state:
    st.session_state.conversations = []

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

st.title("Quizbot Application")

if st.button("Start Learning Session"):
    st.session_state.counter = 0
    st.session_state.score = 0
    st.session_state.conversations = []

st.write("Your current score:", st.session_state.score)

if len(st.session_state.conversations) < len(keywords):
    user_answer = st.text_area("Your answer:")

    if user_answer:
        if len(st.session_state.conversations) > 0:
            current_conversation = st.session_state.conversations[-1]
            current_conversation['user_answer'] = user_answer

            correct, feedback = grade_answer(current_conversation['question'], user_answer)
            st.session_state.score += int(correct)
            current_conversation['feedback'] = feedback

            st.session_state.counter += 1

        if st.session_state.counter < len(keywords):
            keyword = keywords[st.session_state.counter]
            question = generate_question(keyword)

            st.session_state.conversations.append({
                'keyword': keyword,
                'question': question,
                'user_answer': None,
                'feedback': None,
            })

for i, conversation in enumerate(reversed(st.session_state.conversations), start=1):
    with st.expander(f"Thread {len(st.session_state.conversations)-i+1}", expanded=(i==1)):
        st.write(f"**Question**\n\n{conversation['question']}")
        if conversation['user_answer'] is not None:
            st.write(f"**Your Answer**\n\n{conversation['user_answer']}")
            st.write(f"**Feedback**\n\n{conversation['feedback']}")
