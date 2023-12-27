
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from pathlib import Path
import logging
import yaml


logger = logging.getLogger(__name__)
_ = load_dotenv()


def process_rulebooks():
    """
    Processes a rulebook to create a vector store for Retrieval
    Augmented Generation.
    """
    logger.info("---  Process Rulebooks --- ")
    with open("./books/book_map.yaml", "r") as fh:
        rulesets = yaml.safe_load(fh)

    for ruleset, ruleset_rec in rulesets.items():
        logger.info(f"Starting ruleset {ruleset}")
        docs = []
        for book in ruleset_rec["books"]:
            rel_path = Path("./books") / book["filename"]
            logger.info(f"Loading book {book['book_name']} from {str(rel_path)}")
            loader = PyPDFLoader(str(rel_path))
            docs += loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        logger.info("Creating vector store")
        vector_store = FAISS.from_documents(splits, OpenAIEmbeddings())
        out_path = Path("./books") / ruleset_rec["index_dir"]
        logger.info(f"Saving vectorstore to {out_path}")
        vector_store.save_local(out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_rulebooks()


