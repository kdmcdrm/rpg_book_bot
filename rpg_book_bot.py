"""
Simple AI RPG Book Rulebot

ToDo:
 - Add page number handling
 - Add ability to have multiple books for single ruleset (OSE + carcass crawlers) - Library class + Additional details in book_map
 - Modify to Discord bot
 - Improvement: Get LLM to process question to include any needed context from previous conversation, currently only
    the immediate question goes to the bot.
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
st.markdown(
    "I am the Rules Master 9000. Ask me your D&D rules questions and bask in my superior knowledge.")

ROLE_MAP = {
    "user": "You",
    "assistant": "RM9000"
}
SYS_MSG = (
    "You are the smartest Dungeon Master in the land, you often make fun of the weak minded fools while answering"
    " their questions accurately.")

QUESTION_TEMPLATE = """Use the following pairs of page number and Rulebook passage to answer the question below. 
Provide the page number and keep the answer concise if you know the answer, dismiss the question as pointless if you 
don't. Feel free to demean the asker for asking such simple questions that everyone should already know the answers 
to. Do not use the same insults each time you answer a question. Question: {question} Context: {context}"""
with open("./books/book_map.json", "r") as fh:
    st.session_state["book_map"] = json.load(fh)

if "agent" not in st.session_state:
    st.session_state.agent = OpenAIRagAgent(os.environ["OPENAI_MODEL_NAME"],
                                            os.environ["OPENAI_API_KEY"],
                                            SYS_MSG,
                                            QUESTION_TEMPLATE)

with st.sidebar:
    st.markdown("# Configuration")
    rulebook_name = st.selectbox("Select Ruleset",
                                 st.session_state["book_map"].keys(),
                                 index=0)
    st.selectbox("Select Personality",
                 options=["Arrogant", "Mean-Spirited", "Smarter Than You", "This doesn't do anything lol"])

# Sets the initial rulebook
if "rulebook" not in st.session_state:
    index_path = Path("./books") / st.session_state["book_map"][rulebook_name]["index_dir"]
    st.session_state["rulebook"] = FAISS.load_local(index_path, OpenAIEmbeddings()).as_retriever(search_kwargs={"k": 3})
    st.session_state["rulebook_name"] = rulebook_name

if st.session_state["rulebook_name"] != rulebook_name:
    index_path = Path("./books") / st.session_state["book_map"][rulebook_name]["index_dir"]
    st.session_state["rulebook"] = FAISS.load_local(index_path, OpenAIEmbeddings()).as_retriever(search_kwargs={"k": 3})
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
    context = [f'pg {doc.metadata["page"]}: {doc.page_content}'
               for doc in st.session_state["rulebook"].get_relevant_documents(user_input)]
    response = st.session_state.agent(user_input, context)

    with st.chat_message(ROLE_MAP["assistant"]):
        st.markdown(response)
