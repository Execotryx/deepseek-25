from deepseek_connector import DeepSeekR1LocalConnector
from ollama import chat, ChatResponse
import gradio as gr

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

    def ask(self, request: str) -> str:
        """
        Sends a request to the LLM and returns the response.
        Args:
            request (str): The text to summarize.
        Returns:
            str: The summary of the text.
        """
        if not request or request.strip() == "":
            raise ValueError("Request cannot be empty.")
        prompt: str = (f"Summarize the following text:\n\n{request}\n\n")
        self._add_user_message(prompt)
        response: ChatResponse = chat(model=self.MODEL_ID, messages=self._chat_history, stream=False)
        if response.message.content:
            return self._add_assistant_message(response.message.content)
        else:
            raise ValueError("No content in the response from the model.")


summarizer = DeepSeekR1Summarizer()
summarizer_interface: gr.Interface = gr.Interface(
    fn=summarizer.ask,
    inputs=gr.Textbox(label="Text to Summarize", placeholder="Enter the text you want to summarize here..."),
    outputs=gr.Textbox(label="Summary", placeholder="The summary will appear here..."),
    title="DeepSeek R1 Text Summarizer",
    description="This application uses the DeepSeek R1 model to summarize text. Enter the text you want to summarize in the input box and click 'Submit' to get the summary.",
)
    
if __name__ == "__main__":
    summarizer_interface.launch()