"""
Simple AI RPG Book Rulebot
"""
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import logging
import os
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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system",
         "content": "You are the smartest Dungeon Master in the land, you often make fun of the weak minded "
                    "fools while answering their questions accurately. You don't repeat yourself."}
    ]

if "retriever" not in st.session_state:
    # ToDo: Make a command line arg
    vector_store = FAISS.load_local("./books/faiss_index", OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    st.session_state["retriever"] = retriever


def get_and_format_context(question, ret):
    docs = ret.get_relevant_documents(question)
    return "\n\n".join(doc.page_content for doc in docs)


def generate_response(question: str,
                      retriever_,
                      chat_history: list[dict]):
    # Just add retreived sources here, don't keep in history and display
    prompt_template = """
Use the following pieces of retrieved context to answer the following question. If you don't know the answer, 
mock the questioner to hide your shame. Keep the answer concise if you do know it, feel free to make it clear that the
questioner should have known it themselves. Refer to the context as The Rulebook, always capitalize it as you 
hold it in great esteem.
    Question: {question} 
    Context: {context}
"""
    prompt_str = prompt_template.format(question=question, context=get_and_format_context(question, retriever_))
    prompt_msg = {"role": "user", "content": prompt_str}
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the ChatGPT model
        messages=chat_history[:-1] + [prompt_msg],  # replace last message with prompted message
        temperature=0.2
    )
    return res.choices[0].message["content"].strip()


# Draw previous history to the screen
for message in st.session_state.chat_history:
    if message["role"] == "system":
        continue
    with st.chat_message(ROLE_MAP[message["role"]]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask if you must..."):
    with st.chat_message(ROLE_MAP["user"]):
        st.markdown(user_input)

    # Add request to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Use chat history to generate AI response
    response = generate_response(user_input,
                                 st.session_state.retriever,
                                 st.session_state.chat_history,
                                 )

    with st.chat_message(ROLE_MAP["assistant"]):
        st.markdown(response)
    # Append AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
