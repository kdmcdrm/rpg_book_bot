import openai
from openai import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class OpenAIRagAgent:
    def __init__(self,
                 model_name: str,
                 api_key: str,
                 sys_msg: str,
                 ):
        """
        OpenAI RAG Agent, full context is provided for the call to the
        agent but is not stored to the history (to save tokens).

        Args:
            model_name: The model name to use
            api_key: The OpenAI API key
            sys_msg: The system message to use
        """
        self.model_name = model_name
        self.client = openai.Client(api_key=api_key)
        self.sys_message = \
            {"role": "system", "content": sys_msg}
        self.history = [self.sys_message]

    @staticmethod
    def format_user_message(content: str) -> dict[str, str]:
        return {"role": "user", "content": content}

    @staticmethod
    def format_agent_message(content: str) -> dict[str, str]:
        return {"role": "assistant", "content": content}

    def __call__(self, quest_msg: str) -> Stream[ChatCompletionChunk]:
        """
        Call the LLM agent with a specific question and context.

        Args:
            quest_msg: The full question prompt, with context and question

        Returns:
            response: The agent's response
        """
        quest_msg = self.format_user_message(quest_msg)
        return self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history + [quest_msg],
                temperature=0.3,
                stream=True
        )

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

        req = self.format_user_message(template.format(response=response, history=history))
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[self.sys_message] + [req],
            temperature=0.0
        ).choices[0].message.content.strip()
        return res
