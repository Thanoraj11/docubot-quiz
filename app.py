import streamlit as st
from llama_index.llms import OpenAI, ChatMessage
from typing import List
from nltk.translate.bleu_score import sentence_bleu


from llama_index import (
    GPTVectorStoreIndex, Document, SimpleDirectoryReader,
    QuestionAnswerPrompt, LLMPredictor, ServiceContext
)


# from langchain import OpenAI
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




llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")






def process_pdf(uploaded_file):
    loader = PDFReader()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        documents = loader.load_data(file=Path(temp_file.name))
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-davinci-003", max_tokens=1000))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
    if "index" not in st.session_state:
        index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
        query_engine = index.as_query_engine()
        st.session_state.index = query_engine
    # st.session_state.index = index
    return st.session_state.index






class TutorAgent:
    def __init__(self, chat_history: List[ChatMessage] = []):
        self._llm = llm
        self._chat_history = chat_history
        self.score_threshold = 0.7  # adjust this as per your requirements
        self.expected_answer = ""  # Initialize expected_answer attribute

    def reset(self):
        self._chat_history = []

    def extract_keywords(self, text: str) -> List[str]:
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Please list 10 keywords or topics from the following text: {text}")])
        keywords = message.message.content.split('\n')  # Assuming the model returns a newline-separated list
        return keywords

    def generate_question_answer(self, keyword: str) -> (str, str):
        self.reset()
        message = self._llm.chat([ChatMessage(role="system", content=f"Generate a question about the topic: {keyword} with the answer separated by a newline.")])
        question, expected_answer = message.message.content.split('\n')  # Assuming the model returns question and answer separated by a newline
        # st.write(question)
        # st.write(expected_answer)

        self.expected_answer = expected_answer
        return question, expected_answer

    def give_feedback(self, user_answer: str) -> (str, float):
        self._chat_history.append(ChatMessage(role="user", content=user_answer))

        feedback_instructions = """
        Please provide detailed feedback based on the following principles:
       
        Do not give away the correct answer if the answer is incorrect or only partly correct.
        Make sure to mention the principle numbers that are relevant to your feedback.
        If the answer is incorrect or only partly correct, mention the areas to improve.

        Expected answer: {self.expected_answer}
        User's answer: {answer}
        """
        
        self._chat_history.append(ChatMessage(role="system", content=feedback_instructions))

        message = self._llm.chat(self._chat_history)
        feedback = message.message.content

        score = self.get_score(user_answer, self.expected_answer)
        return feedback, score

    def get_score(self, user_answer: str, expected_answer: str) -> float:
        return sentence_bleu([expected_answer.split()], user_answer.split())



try:

    tutor = TutorAgent()

    st.title("AI Tutor")