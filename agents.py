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

    def __call__(self, question: str,
                 context_docs: list[str],
                 tweak_response: bool = True) -> str:
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
        if tweak_response:
            res = self._tweak_response_repetition(res)

        # Add question (without context) and answer to the history
        self.history.append(self._format_user_message(question))
        self.history.append(self._format_agent_message(res))

        return res

    def _tweak_response_repetition(self, response) -> str:
        """
        Varies the last LLM response if the LLM is being repetitive (which it tends to be).
        Args:
            response: The response to vary.
        """
        template = \
            """
        Given the following LLM Response and previous response History, determine if the response is 
        following a repetitive structure. If it is, reword it to vary the structure while maintaining the facts and the 
        general tone of the message. Respond with only the new, more original, response.
        
        Response: {response}
        History: {history}
        
        New Response:
        """
        history = ""
        for msg in self.history[-5:]:
            if msg["role"] == "assistant":
                history += f"{msg['role']}: {msg['content']}\n"

        req = self._format_user_message(template.format(response=response, history=history))
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[self.sys_message] + [req],
            temperature=0.0
        ).choices[0].message.content.strip()
        return res
