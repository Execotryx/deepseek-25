import gradio as gr
from deepseek_connector import DeepSeekR1LocalConnector
from ollama import chat, ChatResponse

class DeepSeekR1NERExtractor(DeepSeekR1LocalConnector):
    SYSTEM_BEHAVIOR = (
        "You are assisting the user in extracting named entities from text. "
        "Your response will contain ONLY the bulleted list of named entities, each on a new line. "
        "Types of entities that you should extract include:\n"
        "- People (type of entity: Person)\n"
        "- Organizations (type of entity: Organization)\n"
        "- Locations (type of entity: Location)\n"
        "- Dates (type of entity: Date)\n"
        "- Other relevant information (type of entity: Other (classify as needed))\n"
        "Example of the response:\n"
        "- DeepSeek (Organization)\n"
        "- Liang Wenfeng (Person)\n"
        "Avoid including any additional text or explanations in your response.\n\n"
    )

    def __init__(self):
        super().__init__(system_behavior=self.SYSTEM_BEHAVIOR)

    def ask(self, request: str) -> str:
        prompt = f"Extract named entities from the following text:\n\n{request}\n\n"
        self._add_user_message(prompt)
        response: ChatResponse = chat(model=self.MODEL_ID, messages=self._chat_history, stream=False)
        if response.message.content:
            return self._add_assistant_message(response.message.content)
        else:
            raise ValueError("No content in the response from the model.")

ner_extractor = DeepSeekR1NERExtractor()
ner_extractor_interface = gr.Interface(
    fn=ner_extractor.ask,
    inputs=gr.Textbox(label="Text Input", placeholder="Enter the text from which you want to extract named entities..."),
    outputs=gr.Markdown(label="Extracted Named Entities"),
    title="Named Entity Recognition Extractor",
    description="This tool extracts named entities from the provided text. It identifies people, organizations, locations, dates, and other relevant information."
)
if __name__ == "__main__":
    ner_extractor_interface.launch()