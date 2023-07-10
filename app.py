import streamlit as st
from llama_index.llms import OpenAI, ChatMessage
from typing import List
from nltk.translate.bleu_score import sentence_bleu
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



# import streamlit as st
# from llama_index.llms import OpenAI
# from llama_index import (
#     GPTVectorStoreIndex, Document, SimpleDirectoryReader,
#     QuestionAnswerPrompt, LLMPredictor, ServiceContext
# )
# from tempfile import NamedTemporaryFile
# from llama_index import download_loader
# import openai
# import os
# from pathlib import Path
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

PDFReader = download_loader("PDFReader")

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")

def process_pdf(uploaded_file):
    loader = PDFReader()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        documents = loader.load_data(file=Path(temp_file.name))
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-davinci-003", max_tokens=1000))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
    index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
    query_engine = index.as_query_engine()
    return query_engine

def save_chat_history(chat_history):
    with open('chat_history.json', 'w') as f:
        json.dump(chat_history, f)

def load_chat_history():
    try:
        with open('chat_history.json', 'r') as f:
            return json.loads(f)
    except FileNotFoundError:
        return {}

st.set_page_config(layout="wide")

st.title("AI Tutor")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    query_engine = process_pdf(uploaded_file)
    res  = query_engine.query("Please list 10 keywords or topics from the document").response
    keywords = res.split('\n')
    st.sidebar.write(keywords)


if st.button("Start learning Session"):
    current_keyword = keywords[0]
    st.sidebar.write(current_keyword)
    question.message = llm.chat([ChatMessage(role="system", content=f"Generate a question about the topic: {current_keyword}")])
    st.write(question)
    chat_history['question'] = question
    save_chat_history(chat_history)


answer = st.text_input("Your answer:")

if answer:
    feedback.message = llm.chat([ChatMessage(role="system", content=f"Give feedback on the answer: {answer}")])
    st.write(f"Feedback: {feedback}")
    chat_history['answer'] = answer
    chat_history['feedback'] = feedback
    save_chat_history(chat_history)





# PDFReader = download_loader("PDFReader")

# openai.api_key = os.getenv("OPENAI_API_KEY")

# llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# def process_pdf(uploaded_file):
#     loader = PDFReader()
#     with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(uploaded_file.getvalue())
#         documents = loader.load_data(file=Path(temp_file.name))
#     return documents

# st.set_page_config(layout="wide")  # Set layout to wide

# st.title("AI Tutor")

# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# if uploaded_file is not None:
#     documents = process_pdf(uploaded_file)
#     st.sidebar.write(documents)

# if st.button("Start learning Session"):
#     question = llm.chat([ChatMessage(role="system", content=f"Generate a question about the document")])
#     st.sidebar.write(question)

# answer = st.text_input("Your answer:")

# if answer:
#     feedback = llm.chat([ChatMessage(role="system", content=f"Give feedback on the answer: {answer}")])
#     st.write(f"Feedback: {feedback}")


# import streamlit as st
# from llama_index.llms import OpenAI, ChatMessage
# from typing import List
# from nltk.translate.bleu_score import sentence_bleu
# from llama_index import (
#     GPTVectorStoreIndex, Document, SimpleDirectoryReader,
#     QuestionAnswerPrompt, LLMPredictor, ServiceContext
# )
# from tempfile import NamedTemporaryFile
# from llama_index import download_loader
# import openai
# import os
# from pathlib import Path
# from llama_index.retrievers import VectorIndexRetriever
# from llama_index.query_engine import RetrieverQueryEngine
# from random import randint
# import random
# import string

# PDFReader = download_loader("PDFReader")

# openai.api_key = os.getenv("OPENAI_API_KEY")

# llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# def process_pdf(uploaded_file):
#     loader = PDFReader()
#     with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(uploaded_file.getvalue())
#         documents = loader.load_data(file=Path(temp_file.name))
    
#     llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.15, model_name="text-davinci-003", max_tokens=1000))
#     service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    
#     if "index" not in st.session_state:
#         index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
#         query_engine = index.as_query_engine()
#         st.session_state.index = query_engine
#     return st.session_state.index

# class TutorAgent:
#     def __init__(self, chat_history: List[ChatMessage] = []):
#         self._llm = llm
#         self._chat_history = chat_history
#         self.score_threshold = 7  # adjust this as per your requirements
#         self.expected_answer = ""  # Initialize expected_answer attribute

#     def reset(self):
#         self._chat_history = []

#     def extract_keywords(self, text: str) -> List[str]:
#         self.reset()
#         message = self._llm.chat([ChatMessage(role="system", content=f"Please list 10 keywords or topics from the following text: {text}")])
#         keywords = message.message.content.split('\n')  # Assuming the model returns a newline-separated list
#         return keywords

#     def generate_question_answer(self, keyword: str):
#         self.reset()
#         message = self._llm.chat([ChatMessage(role="system", content=f"Generate a question about the topic: {keyword} with the answer separated by a newline.")])
        
#         responses = message.message.content.split('\n')  # Assuming the model returns question and answer separated by a newline
#         self.expected_answer = responses[1]
#         return responses[0]

#     def give_feedback(self, user_answer: str):
#         self._chat_history.append(ChatMessage(role="user", content=user_answer))

#         feedback_instructions = """
#         Please provide detailed feedback based on the following principles:
       
#         Do not give away the correct answer if the answer is incorrect or only partly correct.
#         Make sure to mention the principle numbers that are relevant to your feedback.
#         If the answer is incorrect or only partly correct, mention the areas to improve.

#         Expected answer: {self.expected_answer}
#         User's answer: {answer}
#         """
        
#         self._chat_history.append(ChatMessage(role="system", content=feedback_instructions))

#         message = self._llm.chat(self._chat_history)
#         feedback = message.message.content

#         score = self.get_score(user_answer, self.expected_answer)
#         return feedback, int(score)

#     def get_score(self, user_answer: str, expected_answer: str) -> float:
#         Prompt = """
#         evaluate on a general understanding of how well the user's answer aligns with the expected answer.
#         In a simple format, I could use a scale of 1-10 where 1 signifies no alignment with the expected answer and 10 signifies perfect alignment.
#         Return only a number between 1 and 10 in the response.
#         Expected answer: {expected_answer}
#         User's answer: {user_answer}
#         """
#         message = self._llm.chat([ChatMessage(role="system", content=Prompt)])
#         score = message.message.content
#         return score

# tutor = TutorAgent()

# st.set_page_config(layout="wide")  # Set layout to wide

# st.title("AI Tutor")

# if "keywords" not in st.session_state:
#     st.session_state.keywords =[]

# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
# if uploaded_file is not None:
#     st.session_state.index = process_pdf(uploaded_file)
#     res  = st.session_state.index.query("Please list 10 keywords or topics from the document").response
#     st.session_state.keywords = res.split('\n')
#     st.sidebar.write(st.session_state.keywords)

# if "currentKeyword" not in st.session_state:
#     st.session_state.currentKeyword = 1

# #  keywords

# if st.button("Start learning Session") :
#     current_keyword = st.session_state.keywords[st.session_state.currentKeyword]
#     st.sidebar.write(current_keyword)
#     question = tutor.generate_question_answer(current_keyword)
#     st.session_state[f"Q{st.session_state.currentKeyword}"] = question

# # Current question
# if "Q{st.session_state.currentKeyword}" in st.session_state:
#     with st.expander(f"Question {st.session_state.currentKeyword} (Current)", expanded=True):
#         st.write(st.session_state[f"Q{st.session_state.currentKeyword}"])

# # Previous messages
# for i in range(st.session_state.currentKeyword):
#     # with st.expander(f"Question {i+1}"):
#     st.write(st.session_state[f"Q{i}"])
#     if f"A{i}" in st.session_state:
#         st.write(f"Your answer: {st.session_state[f'A{i}']}")
#     if f"F{i}" in st.session_state:
#         st.write(f"Feedback: {st.session_state[f'F{i}']}")

#     answer = st.text_input("Your answer:")
#     # if st.button("Submit Answer"):
#     feedback, score = tutor.give_feedback(answer)
#     # Store user answer and feedback
#     st.session_state[f"A{st.session_state.currentKeyword}"] = answer
#     st.session_state[f"F{st.session_state.currentKeyword}"] = feedback

#     if score < tutor.score_threshold:
#         subtopic = tutor.extract_keywords(feedback)
#         st.session_state.keywords.insert(st.session_state.currentKeyword + 1, subtopic[0])

#     st.session_state.currentKeyword += 1

#     if st.session_state.currentKeyword < len(st.session_state.keywords):
#         current_keyword = st.session_state.keywords[st.session_state.currentKeyword]
#         question = tutor.generate_question_answer(current_keyword)
#         st.session_state[f"Q{st.session_state.currentKeyword}"] = question
#     else:
#         st.write("You have completed all the selected topics. Well done!")
