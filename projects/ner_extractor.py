import gradio as gr
from deepseek_connector import DeepSeekR1LocalConnector
from ollama import ChatResponse

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
        "Given text:\n"
        "\"Hangzhou DeepSeek Artificial Intelligence Basic Technology Research Co., Ltd., doing business as DeepSeek, is a Chinese artificial intelligence company that develops large language models (LLMs). Based in Hangzhou, Zhejiang, Deepseek is owned and funded by the Chinese hedge fund High-Flyer. DeepSeek was founded in July 2023 by Liang Wenfeng, the co-founder of High-Flyer, who also serves as the CEO for both companies. The company launched an eponymous chatbot alongside its DeepSeek-R1 model in January 2025.\"\n\n"
        "Extracted named entities:\n"
        "- DeepSeek (Organization)\n"
        "- Hangzhou (Location)\n"
        "- Zhejiang (Location)\n"
        "- High-Flyer (Organization)\n"
        "- July 2023 (Date)\n"
        "- January 2025 (Date)\n"
        "- Hangzhou DeepSeek Artificial Intelligence Basic Technology Research Co., Ltd. (Organization)\n"
        "- DeepSeek-R1 (Other: Product name)\n"
        "- Liang Wenfeng (Person)\n\n"
    )

    def __init__(self):
        super().__init__(system_behavior=self.SYSTEM_BEHAVIOR)

    def ask(self, request: str) -> str:
        prompt = f"Given text:\n\"{request}\"\n\nExtracted named entities:\n"
        self._add_user_message(prompt)
        response: ChatResponse = self._query()
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
    ner_extractor_interface.queue().launch()