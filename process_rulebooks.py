
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

logger = logging.getLogger(__name__)
_ = load_dotenv()


def process_rulebooks():
    """
    Processes a rulebook to create a vector store for Retrieval
    Augmented Generation.
    Requires a ./books subdirectory with a "book_map.yaml" config file. Will
    write the indexes to the ./books directory as well.
    """
    logger.info("---  Process Rulebooks --- ")
    with open("./books/book_map.yaml", "r") as fh:
        rulesets = yaml.safe_load(fh)

    for ruleset, ruleset_rec in rulesets.items():
        index_path = Path("./books") / ruleset_rec["index_dir"]
        if index_path.exists():
            logger.info(f"Skipping ruleset {ruleset}, index found at {ruleset_rec['index_dir']}")
            continue
        else:
            logger.info(f"Starting ruleset {ruleset}")
        docs = []
        for book_path, book_rec in ruleset_rec["books"].items():
            rel_path = Path("./books") / book_path
            logger.info(f"Loading book {book_rec['title']} from {str(rel_path)}")
            loader = PyPDFLoader(str(rel_path))
            docs += loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        logger.info("Creating vector store")
        vector_store = FAISS.from_documents(splits, OpenAIEmbeddings())
        logger.info(f"Saving vectorstore to {index_path}")
        vector_store.save_local(index_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_rulebooks()


