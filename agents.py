import openai
from abc import ABC, abstractmethod


class RagAgent(ABC):
    """
    Base LLM Agent Abstract Class
    """
    @abstractmethod
    def __call__(self, question: str, context: list[str]) -> str:

        """
        Calls the agent

        Args:
            question: The user's request
            context: The context for the response

        Returns:
            response: The agent response
        """
        pass


class OpenAIRagAgent(RagAgent):
    def __init__(self,
                 model_name: str,
                 api_key: str,
                 sys_msg: str,
                 question_template: str):
        """
        OpenAI RAG Agent, full context is provided for the call to the
        agent but is not stored to the history (to save tokens).

        Args:
            model_name: The model name to use
            api_key: The OpenAI API key
            sys_msg: The system message to use
            question_template: The template for answering questions, needs to have {question} and {context} tags
                to be filled at calling
        """
        self.model_name = model_name
        self.client = openai.Client(api_key=api_key)
        self.sys_message = \
            {"role": "system", "content": sys_msg}
        self.history = [self.sys_message]
        self.question_template = question_template

    @staticmethod
    def _format_user_message(content: str) -> dict[str, str]:
        return {"role": "user", "content": content}

    @staticmethod
    def _format_agent_message(content: str) -> dict[str, str]:
        return {"role": "assistant", "content": content}

    def __call__(self, question: str, context_docs: list[str]) -> str:
        """
        Call the LLM agent with a specific question and context.

        Args:
            question: The question asked
            context_docs: The context documents

        Returns:
            response: The agent's response
        """
        quest_msg = self._format_user_message(self.question_template.format(question=question,
                                                                            context="\n".join(context_docs)))
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.history + [quest_msg],
            temperature=0.2
        ).choices[0].message.content.strip()

        # Add question (without contexT) and answer to the history
        self.history.append(self._format_user_message(question))
        self.history.append(self._format_agent_message(res))

        return res
