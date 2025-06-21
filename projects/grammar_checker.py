from deepseek_connector import DeepSeekR1LocalConnector
from ollama import chat, ChatResponse
import gradio as gr

class DeepSeekR1GrammarChecker(DeepSeekR1LocalConnector):
    """DeepSeek R1 Grammar Checker that corrects grammar mistakes in provided text."""

    def __init__(self):
        super().__init__(system_behavior=("You are an editor that checks the grammar of the text."
                                          "You will correct the grammar mistakes in the text provided by the user."
                                          "Your response will contain ONLY the corrected text, without explanations, maintaining the original meaning and context."
                                          "If there are no mistakes, your response will contain ONLY the original text without changes."))

    def ask(self, request: str) -> str:
        """Checks grammar of the provided text.

        Args:
            request (str): The text to be checked for grammar.

        Returns:
            str: The corrected text with grammar mistakes fixed.

        Raises:
            ValueError: If the request is empty.
        """
        if not request or request.strip() == "":
            raise ValueError("Request cannot be empty.")
        prompt = f"Check the grammar of the following text:\n\n{request}\n\n"
        self._add_user_message(prompt)
        return self._query()

grammar_checker_interface: gr.Interface = gr.Interface(
    fn=DeepSeekR1GrammarChecker().ask,
    inputs=gr.Textbox(label="Text to Check", placeholder="Enter the text you want to check for grammar..."),
    outputs=gr.Textbox(label="Corrected Text"),
    title="DeepSeek R1 Grammar Checker",
    description="This tool checks the grammar of the provided text and returns the corrected version."
)
if __name__ == "__main__":
    grammar_checker_interface.launch()