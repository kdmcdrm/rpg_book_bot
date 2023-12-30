"""
RPG Rulebook Bot

ToDo:
 - Pick book map based on environment file
 - Improvement: Get LLM to process question to include any needed context from previous conversation, currently only
    the immediate question goes to the bot.
 - Improvement: Model selection
 - Modify to Discord bot
"""

import streamlit as st
import openai
import logging
import os
from agents import OpenAIRagAgent
from dotenv import load_dotenv
from ruleset import load_rulesets

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
    with st.chat_message(ROLE_MAP["assistant"]):
        placeholder = st.empty()
        full_res = ""
        # Stream output, based on ChatGPT
        completion = st.session_state.agent(user_input, context)
        for chunk in completion:
            full_res += (chunk.choices[0].delta.content or "")
            placeholder.markdown(full_res + "▌")
        placeholder.markdown(full_res)

    # Add question and answer to agent history
    agent = st.session_state.agent
    agent.history.append(agent.format_user_message(user_input))
    agent.history.append(agent.format_agent_message(full_res))
