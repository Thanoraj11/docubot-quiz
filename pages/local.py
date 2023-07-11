import streamlit as st
import openai
from openai import GPT3Completion

openai.api_key = 'your-openai-api-key'

# GPT-3.5 Turbo parameters
ENGINE = "text-davinci-0035"
MAX_TOKENS = 60

keywords = ['Python', 'Java', 'JavaScript']  # Your list of keywords here

# Initialization
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'conversations' not in st.session_state:
    st.session_state.conversations = []

def grade_answer(question, user_answer):
    # You'll need to implement this function depending on how you want to grade the answers.
    # In this mock function, it always returns True.
    return True, "Correct answer!"

def generate_question(keyword):
    prompt = f"Explain the concept of {keyword} in programming."
    response = openai.Completion.create(engine=ENGINE, prompt=prompt, max_tokens=MAX_TOKENS)
    return response.choices[0].text.strip()

st.title("Quizbot Application")

# Start learning session
if st.button("Start Learning Session"):
    st.session_state.counter = 0
    st.session_state.score = 0
    st.session_state.conversations = []

if st.session_state.counter < len(keywords):
    if st.button("Generate Question"):
        keyword = keywords[st.session_state.counter]
        question = generate_question(keyword)
        st.session_state.conversations.append({
            'keyword': keyword,
            'question': question,
            'user_answer': None,
            'feedback': None,
        })

    if len(st.session_state.conversations) > 0:
        current_conversation = st.session_state.conversations[-1]

        user_answer = st.text_input("Your answer:")
        if user_answer:
            current_conversation['user_answer'] = user_answer

            # Grade the answer
            correct, feedback = grade_answer(current_conversation['question'], user_answer)
            st.session_state.score += int(correct)
            current_conversation['feedback'] = feedback

            # Move to the next question
            st.session_state.counter += 1

for i, conversation in enumerate(reversed(st.session_state.conversations), start=1):
    with st.expander(f"Question {len(st.session_state.conversations)-i+1}", expanded=(i==1)):
        st.write("Question:", conversation['question'])
        st.write("Your answer:", conversation['user_answer'])
        st.write("Feedback:", conversation['feedback'])

st.write("Your current score:", st.session_state.score)
