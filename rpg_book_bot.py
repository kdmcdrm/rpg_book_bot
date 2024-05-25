"""
RPG Rulebook Bot

ToDo:
 - Improvement: Agent should store question history, but not context to avoid overflowing the history
 - Refactor: Agent could wrap up question template, ruleset, etc.
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
import yaml

logger = logging.getLogger(__name__)
_ = load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

# Streamlit app
st.set_page_config(page_title="Rules Master 9000")
st.title("Rules Master 9000")
st.markdown(
    "I am the Rules Master 9000. Ask me your rules questions and bask in my superior knowledge.")

ROLE_MAP = {
    "user": "You",
    "assistant": "RM9000"
}
SYS_MSG = (
    "You are the smartest Rules Master in the land, you often make fun of the weak minded fools while answering"
    " their questions accurately.")


# Load all rulesets
if "rulesets" not in st.session_state:
    st.session_state["rulesets"] = load_rulesets("./books/book_map.yaml")


# Create agent
if "agent" not in st.session_state:
    st.session_state.agent = OpenAIRagAgent(os.environ["OPENAI_MODEL_NAME"],
                                            os.environ["OPENAI_API_KEY"],
                                            SYS_MSG)

# Load question templates
if "templates" not in st.session_state:
    with open("./books/book_map.yaml", "r") as fh:
        conf = yaml.safe_load(fh)
    st.session_state["templates"] = {}
    for name in conf.keys():
        st.session_state["templates"][name] = conf[name]["question_template"]


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
        # Build prompt based on current question template
        quest_msg = st.session_state["templates"][ruleset_name].format(question=user_input, context=context)
        completion = st.session_state.agent(quest_msg)
        for chunk in completion:
            full_res += (chunk.choices[0].delta.content or "")
            placeholder.markdown(full_res + "▌")
        placeholder.markdown(full_res)

    # Add question and answer to agent history
    agent = st.session_state.agent
    agent.history.append(agent.format_user_message(user_input))
    agent.history.append(agent.format_agent_message(full_res))
