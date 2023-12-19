"""
Script to load the rulebook pdf and create the vectorstore
"""
import os

from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)
_ = load_dotenv()


def read_rulebook(book_dir):
    logger.info("---  Read Rulebook --- ")
    logger.info(f"Loading PDFs from {str(book_dir)}")
    loader = PyPDFDirectoryLoader(book_dir)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    logger.info("Creating vector store")
    vector_store = FAISS.from_documents(splits, OpenAIEmbeddings())
    out_path = os.path.join(book_dir, "faiss_index")
    logger.info(f"Saving vectorstore to {out_path}")
    vector_store.save_local(out_path)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("book_dir", help="Directory containing the rule book to digest")

    args = parser.parse_args()

    read_rulebook(args.book_dir)


