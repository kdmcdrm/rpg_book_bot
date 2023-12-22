"""
Simple AI RPG Book Rulebot
"""
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import logging
import os
from agents import OpenAIRagAgent
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
_ = load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

# Streamlit app
st.set_page_config(page_title="D&D Rulebot")
st.title("D&D Rulebot")
st.markdown("He knows the rules better than you do.")
# ToDo: Have GPT generate a tagline on the fly

ROLE_MAP = {
    "user": "You",
    "assistant": "DM"
}
SYS_MSG = (
    "You are the smartest Dungeon Master in the land, you often make fun of the weak minded fools while answering"
    " their questions accurately")

QUESTION_TEMPLATE = """Use the following pieces of the Rulebook to answer the question below. 
Keep the answer concise know the answer, dismiss the question as pointless if you don't. If you refer to the Rulebook
always refer to it as the Rulebook.
    Question: {question} 
    Context: {context}
"""

if "agent" not in st.session_state:
    st.session_state.agent = OpenAIRagAgent("gpt-3.5-turbo",
                                            os.environ["OPENAI_API_KEY"],
                                            SYS_MSG,
                                            QUESTION_TEMPLATE)

if "retriever" not in st.session_state:
    vector_store = FAISS.load_local("./books/faiss_index", OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    st.session_state["retriever"] = retriever

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
    context = [x.page_content for x in st.session_state.retriever.get_relevant_documents(user_input)]
    response = st.session_state.agent(user_input, context)

    with st.chat_message(ROLE_MAP["assistant"]):
        st.markdown(response)
