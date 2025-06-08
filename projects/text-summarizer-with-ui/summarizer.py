from ..DeepSeekR1LocalConnector import DeepSeekR1LocalConnector
from ollama import chat, ChatResponse

class DeepSeekR1Summarizer(DeepSeekR1LocalConnector):
    """
    DeepSeek R1 Summarizer class that inherits from DeepSeekR1LocalConnector.
    This class is used to summarize text using the DeepSeek R1 model.
    """

    def __init__(self):
        """
        Initializes the DeepSeekR1Summarizer with the specified model name.
        """
        super().__init__(system_behavior=("You are helping user to summarize text."
                                          "You will be given a text and you need to summarize it in a concise manner."
                                          "You will respond with concise summary of a text and bullet list of main subjects covered in the text, that was provided by user."))

    def ask(self, request: str) -> str | None:
        """
        Sends a request to the LLM and returns the response.
        Args:
            request (str): The text to summarize.
        Returns:
            str: The summary of the text.
        """
        if not request or request.strip() == "":
            raise ValueError("Request cannot be empty.")
        prompt: str = (f"Please summarize the following text:\n\n{request}\n\n")
        self._add_user_message(prompt)
        response: ChatResponse = chat(model=self.MODEL_ID, messages=self._chat_history, stream=False)
        if response.message.content:
            self._add_assistant_message(response.message.content)
            return response.message.content
        else:
            raise ValueError("No content in the response from the model.")

if __name__ == "__main__":
    summarizer = DeepSeekR1Summarizer()
    while True:
        text_to_summarize = input("Enter the text to summarize (or type 'exit' to quit): ")
        if text_to_summarize.lower() == 'exit':
            break
        try:
            summary = summarizer.ask(text_to_summarize)
            print("Summary:", summary)
        except ValueError as e:
            print("Error:", e)