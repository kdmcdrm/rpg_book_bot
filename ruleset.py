import re

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import yaml
from pathlib import Path


class Ruleset:
    def __init__(self, index_dir: str, book_recs: dict[str, str]):
        """
        A Ruleset can be made up of multiple books, but they are all in a single
        index.
        Args:
            index_dir: The index directory
            book_recs: A set of dictionaries mapping filename to book details
        """

        self.ret = FAISS.load_local(index_dir, OpenAIEmbeddings()).as_retriever(
            search_kwargs={"k": 3}
        )
        self.book_recs = book_recs

    def get_context(self, question: str) -> str:
        """
        Gets the relevant context for a question.
        """
        doc_template = "{book} pg {pg}: {context}"
        context = []
        # Overwrite the title and applies page offset to match page numbers
        for doc in self.ret.get_relevant_documents(question):
            # Pathlib would not handle the Windows paths here, do a replacement
            path = Path(doc.metadata["source"].replace("\\", "/"))
            context.append(doc_template.format(
                book= self.book_recs[path.name]["title"],
                pg=str(doc.metadata["page"] - self.book_recs[path.name]["page_offset"]),
                context=doc.page_content)
            )
        return "\n".join(context)


def load_rulesets(config_path: str) -> dict[str, Ruleset]:
    """
    Loads a dictionary of Rulesets from a config file.
    """
    with open(config_path, "r") as fh:
        conf = yaml.safe_load(fh)
    rulesets = {}
    for name in conf:
        # Assumes indices are in the config directory
        rulesets[name] = Ruleset(Path(config_path).parent / conf[name]["index_dir"],
                                 conf[name]["books"])
    return rulesets
