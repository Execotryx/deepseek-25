import gradio as gr
from deepseek_connector import DeepSeekR1LocalConnector

class DeepSeekR1NERExtractor(DeepSeekR1LocalConnector):
    SYSTEM_BEHAVIOR = (
        "You are assisting the user in extracting named entities from text. "
        "You will receive a text input and your task is to identify and extract named entities from it.\n"
        "You will respond ONLY with a bulleted list of named entities, compatible with SpaCy.\n"
        "Here are some examples of text provided and named entities extracted to follow:\n"
        "\"\"\""
        "Example 1:\n"
        "Text: 'Apple Inc. is looking at buying U.K. startup for $1 billion'\n"
        "Named Entities:\n"
        "- [ORG] Apple Inc.\n"
        "- [GPE] U.K.\n"
        "- [MONEY] $1 billion\n"
        "Example 2:\n"
        "Text: 'Barack Obama was the 44th President of the United States.'\n"
        "Named Entities:\n"
        "- [PERSON] Barack Obama\n"
        "- [ORDINAL] 44th\n"
        "- [ORG] President of the United States\n"
        "Example 3:\n"
        "Text: 'The Eiffel Tower is located in Paris.'\n"
        "Named Entities:\n"
        "- [LOC] Eiffel Tower\n"
        "- [GPE] Paris\n"
        "Example 4:\n"
        "Text: 'On July 20, 1969, Neil Armstrong became the first person to walk on the moon.'\n"
        "Named Entities:\n"
        "- [DATE] July 20, 1969\n"
        "- [PERSON] Neil Armstrong\n"
        "- [ORDINAL] first\n"
        "- [LOC] moon\n"
        "Example 5:\n"
        "Text: 'Microsoft Corporation announced its new product in Redmond, Washington.'\n"
        "Named Entities:\n"
        "- [ORG] Microsoft Corporation\n"
        "- [GPE] Redmond\n"
        "- [GPE] Washington\n"
        "\"\"\""
    )

    def __init__(self):
        super().__init__(system_behavior=self.SYSTEM_BEHAVIOR, model_id="deepseek-r1:8b")

    def ask(self, request: str) -> str:
        if not request or request.strip() == "":
            raise ValueError("Request cannot be empty.")
        prompt = f"Text: {request}\nNamed Entities:\n"
        self._add_user_message(prompt)
        return self._query()

if __name__ == "__main__":
    ner_extractor = DeepSeekR1NERExtractor()
    ner_extractor_interface = gr.Interface(
        fn=ner_extractor.ask,
        inputs=gr.Textbox(label="Text Input", placeholder="Enter the text from which you want to extract named entities..."),
        outputs=gr.Textbox(label="Extracted Named Entities", placeholder="The named entities will be listed here...", show_copy_button=True),
        title="Named Entity Recognition Extractor",
        description="This tool extracts named entities from the provided text. It identifies people, organizations, locations, dates, and other relevant information."
    )
    ner_extractor_interface.queue().launch()