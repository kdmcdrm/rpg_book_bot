"""
RPG Rulebook Bot

ToDo:
 - Modify to Discord bot
 - Improvement: Get LLM to process question to include any needed context from previous conversation, currently only
    the immediate question goes to the bot.
 - Improvement: Do LLM Response post processing to avoid repetition
"""

import streamlit as st
import openai
import logging
import os
import json
from agents import OpenAIRagAgent
from dotenv import load_dotenv
from ruleset import load_rulesets, Ruleset

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
don't. Do not use the same insults each time you answer a question. Question: {question} Context: {context}"""

# Load all rulesets
if "rulesets" not in st.session_state:
    st.session_state["rulesets"] = load_rulesets("./books/book_map.yaml")


if "agent" not in st.session_state:
    st.session_state.agent = OpenAIRagAgent(os.environ["OPENAI_MODEL_NAME"],
                                            os.environ["OPENAI_API_KEY"],
                                            SYS_MSG,
                                            QUESTION_TEMPLATE)

with st.sidebar:
    st.markdown("# Configuration")
    ruleset_name = st.selectbox("Select Ruleset",
                                st.session_state["rulesets"].keys(),
                                index=0)
    st.selectbox("Select Personality",
                 options=["Arrogant", "Mean-Spirited", "Smarter Than You", "This doesn't do anything lol"])


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

    # Get context based on the selected ruleset
    context = st.session_state["rulesets"][ruleset_name].get_context(user_input)

    # Use Agent to generate response
    response = st.session_state.agent(user_input, context)
    response_varied = st.session_state.agent.vary_last_message()

    with st.chat_message(ROLE_MAP["assistant"]):
        st.markdown(response_varied)
