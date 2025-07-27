from abc import ABC, abstractmethod
from ollama import chat, ChatResponse
import re

class DeepSeekR1LocalConnector(ABC):
    """
    A base class for connecting to the DeepSeek R1 model locally.
    This class provides methods to manage the system behavior and chat history,
    and defines an abstract method for sending requests to the model.
    """
    
    #region System Behavior
    __system_behavior: str = "You are a helpful assistant. Your goal is to help user with anything they will ask of you, in the best way possible, but not contradicting your ethics."

    @property
    def _system_behavior(self) -> str:
        return self.__system_behavior

    @_system_behavior.setter
    def _system_behavior(self, value: str):
        self.__system_behavior = value
    
    #endregion

    #region Chat History
    __chat_history: list[dict[str, str]] = []

    @property
    def _chat_history(self) -> list[dict[str, str]]:
        """
        Returns the chat history.
        Returns:
            list[dict[str, str]]: The chat history containing messages with roles and content.
        """
        return self.__chat_history

    @_chat_history.setter
    def _chat_history(self, value: list[dict[str, str]]):
        """
        Sets the chat history to a new value.
        Args:
            value (list[dict[str, str]]): The new chat history to set.
        """
        # remove the system behavior from value if it exists.
        if any(str(msg['role']).lower() == 'system' for msg in value):
            value = [msg for msg in value if str(msg['role'].lower()) != 'system']
        
        self.__chat_history.extend(value)

    def _add_to_chat_history(self, role: str, content: str) -> None:
        """
        Adds a message to the chat history.
        Args:
            role (str): The role of the message sender ('system', 'user', or 'assistant').
            content (str): The content of the message.
        """
        role = role.lower().strip()
        if role not in ["system", "user", "assistant"]:
            raise ValueError("Role must be one of 'system', 'user', or 'assistant'.")

        self.__chat_history.append({"role": role, "content": content})

    def _add_user_message(self, content: str) -> str:
        """
        Adds a user message to the chat history.

        Args:
            content (str): The content of the user message.

        Returns:
            str: The same content that was added.
        """
        self._add_to_chat_history("user", content)
        return content

    def _add_assistant_message(self, content: str) -> str:
        """
        Adds an assistant message to the chat history.

        Args:
            content (str): The content of the assistant message.

        Returns:
            str: The cleaned content that was added.
        """
        if not content:
            raise ValueError("Content cannot be empty.") 
        content = self.__strip_special_characters(content)
        
        self._add_to_chat_history("assistant", content)
        return content

    def __strip_special_characters(self, content: str) -> str:
        """
        Strips special characters from the content, specifically the <think> tags used by DeepSeek R1.

        Args:
            content (str): The content to be stripped.

        Returns:
            str: The cleaned content without special characters.
        """
        # as per 8th of June, 2025 - the DeepSeek R1 still returns the reasoning process in the response, enclosed in <think></think> pair of tags.
        # For now we need explicitly strip them out.
        return re.sub(r'<think>\s+(?:\w|\W)*?\s+</think>', '', content, flags=re.IGNORECASE | re.MULTILINE).strip()

    #endregion

    #region Constants
    """
    Constants used in the DeepSeek R1 Local Connector.
    These constants include the default system behavior, model ID, and newline separators.
    """
    MODEL_ID: str = "deepseek-r1:8b"
    __model_id: str = MODEL_ID
    NEWLINE_SEPARATOR: str = "\n\n"
    NEWLINE_SYNTACTIC_SEPARATOR_FOR_PROMPT: str = "\n\"\"\"\n"
    #endregion

    @property
    def _model_id(self) -> str:
        return self.__model_id

    @_model_id.setter
    def _model_id(self, value: str):
        cleaned = (value or "").strip()
        self.__model_id = cleaned or self.MODEL_ID

    def __init__(self, system_behavior: str = "", model_id: str = MODEL_ID) -> None:
        # no need to set the system behavior if it is not provided. Default one will do fine, as set in the class variable.
        self._model_id = model_id
        if system_behavior:
            self._system_behavior = system_behavior
        self._add_to_chat_history("system", self._system_behavior)

    @abstractmethod
    def ask(self, request: str) -> str:
        """
        Sends a request to the LLM and returns the response.
        
        Args:
            request (str): The message to send to the LLM.

        Returns:
            str: The response from the LLM.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _query(self) -> str:
        response: ChatResponse = chat(model=self._model_id, messages=self._chat_history, stream=False)
        if response.message.content:
            return self._add_assistant_message(response.message.content)
        else:
            raise ValueError("No content in the response from the model.")

    def _format_for_prompt(self, content: str) -> str:
        return f"{self.NEWLINE_SYNTACTIC_SEPARATOR_FOR_PROMPT}{content}{self.NEWLINE_SYNTACTIC_SEPARATOR_FOR_PROMPT}"

    def reset_chat_history(self):
        """
        Resets the chat history.

        Returns:
            None
        """
        self._chat_history = []
        self._add_to_chat_history("system", self._system_behavior)
