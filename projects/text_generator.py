from deepseek_connector import DeepSeekR1LocalConnector
from ollama import chat, ChatResponse
import gradio as gr

class DeepSeekR1TextGenerator(DeepSeekR1LocalConnector):

    def __init__(self):
        super().__init__(system_behavior=("You are assisting user in text generation tasks."
                                          "Before you start, you will infer the basic points of the text to be generated from the user's request."
                                          "You will derive that judgement basing on your own knowledge.\n"
                                          "Your response will be a text that is coherent, relevant, and follows the user's request."
                                          "Your response should be in the Markdown format, if applicable."))

    def ask(self, request: str, word_limit: int = 500) -> str:
        """
        Generates a text response based on the user's request.

        Args:
            request (str): The user's request for text generation.
            word_limit (int): The maximum number of words in the generated text.

        Returns:
            str: The generated text response.
        """
        request = f"Generate a text based on the following request in {word_limit} words:\n\n{request}\n\n"
        
        self._add_user_message(request)
        
        response: ChatResponse = chat(model=self.MODEL_ID, messages=self._chat_history, stream=False)
        
        if response.message.content:
            return self._add_assistant_message(response.message.content)
        else:
            raise ValueError("No content in the response from the model.")

# replace the Interface with Blocks layout
generator = DeepSeekR1TextGenerator()
with gr.Blocks() as generator_interface:
    text_request = gr.Textbox(label="Text Request", placeholder="Enter the text request you want to generate here...")
    word_limit = gr.Slider(minimum=100, maximum=1000, step=50, label="Word Limit", value=500)
    submit_btn = gr.Button("Submit")
    output = gr.Markdown(label="Generated Text")
    submit_btn.click(fn=generator.ask, inputs=[text_request, word_limit], outputs=[output], show_progress="full")

if __name__ == "__main__":
    generator_interface.launch()