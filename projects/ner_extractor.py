import gradio as gr
from deepseek_connector import DeepSeekR1LocalConnector
from ollama import ChatResponse

class DeepSeekR1NERExtractor(DeepSeekR1LocalConnector):
    SYSTEM_BEHAVIOR = (
        "You are assisting the user in extracting named entities from text. "
        "You will receive a text input and your task is to identify and extract named entities from it.\n"
        "You will respond ONLY with a bulleted list of named entities, compatible with SpaCy.\n\n"
    )

    def __init__(self):
        super().__init__(system_behavior=self.SYSTEM_BEHAVIOR)

    def ask(self, request: str) -> str:
        if not request or request.strip() == "":
            raise ValueError("Request cannot be empty.")
        prompt = f"Extract named entities from the following text, compatible with SpaCy:\n\n{request}\n\n"
        self._add_user_message(prompt)
        return self._query()

ner_extractor = DeepSeekR1NERExtractor()
ner_extractor_interface = gr.Interface(
    fn=ner_extractor.ask,
    inputs=gr.Textbox(label="Text Input", placeholder="Enter the text from which you want to extract named entities..."),
    outputs=gr.Textbox(label="Extracted Named Entities", placeholder="The named entities will be listed here...", show_copy_button=True),
    title="Named Entity Recognition Extractor",
    description="This tool extracts named entities from the provided text. It identifies people, organizations, locations, dates, and other relevant information."
)
if __name__ == "__main__":
    ner_extractor_interface.queue().launch()