"""
Simple AI RPG Book Rulebot
"""
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
_ = load_dotenv()

# Streamlit app
st.title("DM Genius")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system",
         "content": "You are the smartest Dungeon Master in the land, you make fun of the weak minded fools while "
                    "answering their questions."}
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
mock the questioner to hide your shame. Keep the answer concise if you do know it, but make it clear that the
questioner should have known it themselves.
    Question: {question} 
    Context: {context}
"""
    prompt_str = prompt_template.format(question=question, context=get_and_format_context(question, retriever_))
    prompt_msg = {"role": "user", "content": prompt_str}
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the ChatGPT model
        messages=chat_history[:-1] + [prompt_msg],  # replace last message with prompted message
        temperature=0.2
    )
    return response.choices[0].message["content"].strip()


user_input = st.text_input("You:", "")
if user_input:
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    st.text("DM:")
    # Use chat history to generate AI response
    ai_response = generate_response(user_input,
                                    st.session_state.retriever,
                                    st.session_state.chat_history)

    # Append AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

    st.write(ai_response)
