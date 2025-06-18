from deepseek_connector import DeepSeekR1LocalConnector
from ollama import chat, ChatResponse
import gradio as gr

class DeepSeekR1GrammarChecker(DeepSeekR1LocalConnector):
    def __init__(self):
        super().__init__(system_behavior=("You are an editor that checks the grammar of the text."
                                          "You will correct the grammar mistakes in the text provided by the user."
                                          "Your response will contain ONLY the corrected text, without explanations, maintaining the original meaning and context."
                                          "If there are no mistakes, your response will contain ONLY the original text without changes."))

    def ask(self, request: str) -> str:
        """
        Checks the grammar of the provided text.
        Args:
            request (str): The text to be checked for grammar.
        Returns:
            str: The corrected text with grammar mistakes fixed.
        """
        prompt = f"Check the grammar of the following text:\n\n{request}\n\n"
        self._add_user_message(prompt)
        response: ChatResponse = chat(model=self.MODEL_ID, messages=self._chat_history, stream=False)
        if response.message.content:
            return self._add_assistant_message(response.message.content)
        else:
            raise ValueError("No content in the response from the model.")

spellchecker_interface = gr.Interface(
    fn=DeepSeekR1GrammarChecker().ask,
    inputs=gr.Textbox(label="Text to Check", placeholder="Enter the text you want to check for grammar..."),
    outputs=gr.Textbox(label="Corrected Text"),
    title="DeepSeek R1 Grammar Checker",
    description="This tool checks the grammar of the provided text and returns the corrected version."
)
if __name__ == "__main__":
    spellchecker_interface.launch()