"""
Simple AI RPG Book Rulebot

ToDo:
 - Move MODEL_NAME definition to environment
 - Add page number handling
 - Add ability to have multiple books for single ruleset (OSE + carcass crawlers)
 - Modify to Discord bot
"""
from pathlib import Path

import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import logging
import os
import json
from agents import OpenAIRagAgent
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
_ = load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

# Streamlit app
st.set_page_config(page_title="Rules Master 9000")
st.title("Rules Master 9000")
st.markdown("I am the Rules Master 9000. Ask me your D&D rules questions and bask in my knowledge.")

# ToDo: These should be in the session state
ROLE_MAP = {
    "user": "You",
    "assistant": "DM"
}
SYS_MSG = (
    "You are the smartest Dungeon Master in the land, you often make fun of the weak minded fools while answering"
    " their questions accurately.")

QUESTION_TEMPLATE = """Use the following pieces of the Rulebook to answer the question below. 
Keep the answer concise know the answer, dismiss the question as pointless if you don't. Feel free to demean the
asker for asking such simple questions that everyone should already know the answers to. Do not use the same insults
each time you answer a question.
    Question: {question} 
    Context: {context}
"""
with open("./books/book_map.json", "r") as fh:
    BOOK_MAP = json.load(fh)

if "agent" not in st.session_state:
    st.session_state.agent = OpenAIRagAgent("gpt-3.5-turbo",
                                            os.environ["OPENAI_API_KEY"],
                                            SYS_MSG,
                                            QUESTION_TEMPLATE)


with st.sidebar:
    st.markdown("# Configuration")
    rulebook_name = st.selectbox("Select Ruleset",
                                 BOOK_MAP.keys(),
                                 index=0)
    st.selectbox("Select Personality", options=["Arrogant"], disabled=True)


# Sets the initial rulebook
if "rulebook" not in st.session_state:
    index_path = Path("./books") / BOOK_MAP[rulebook_name]["index_dir"]
    st.session_state["rulebook"] = FAISS.load_local(index_path, OpenAIEmbeddings()).as_retriever(k=1)
    st.session_state["rulebook_name"] = rulebook_name

if st.session_state["rulebook_name"] != rulebook_name:
    index_path = Path("./books") / BOOK_MAP[rulebook_name]["index_dir"]
    st.session_state["rulebook"] = FAISS.load_local(index_path, OpenAIEmbeddings()).as_retriever(k=1)
    st.session_state["rulebook_name"] = rulebook_name


# Draw previous history to the screen
for message in st.session_state.agent.history:
    if message["role"] == "system":
        continue
    with st.chat_message(ROLE_MAP[message["role"]]):
        st.markdown(message["content"])

# Ask for next input
if user_input := st.chat_input("Ask if you must..."):
    with st.chat_message(ROLE_MAP["user"]):
        st.markdown(user_input)

    # Use Agent to generate response
    # ToDo: Add page number here
    context = [x.page_content for x in st.session_state["rulebook"].get_relevant_documents(user_input)]
    response = st.session_state.agent(user_input, context)

    with st.chat_message(ROLE_MAP["assistant"]):
        st.markdown(response)
