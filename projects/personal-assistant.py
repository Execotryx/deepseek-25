from deepseek_connector import DeepSeekR1LocalConnector
import gradio as gr
import speech_recognition as sr
import pyttsx3 as tts
from typing import Any
from speech_recognizer import SpeechRecognizer

class PersonalAIAssistant(DeepSeekR1LocalConnector):
    __tts_engine: tts.Engine = None # type: ignore
    __speech_recognizer: SpeechRecognizer = None # type: ignore

    @property
    def _tts_engine(self) -> tts.Engine:
        return self.__tts_engine

    @property
    def _speech_recognizer(self) -> SpeechRecognizer:
        return self.__speech_recognizer

    def __init__(self):
        system_behavior: str = (
            "You are a personal AI assistant, that helping user with various tasks."
            "You will answer questions, provide information, and assist with tasks based on user requests."
            "Answer to the user only if you sure for 95%% about accuracy of your response."
        )
        super().__init__(system_behavior=system_behavior, model_id="deepseek-r1:8b")
        self.__tts_engine = tts.init()
        self.__speech_recognizer = SpeechRecognizer()

    def listen_for_request_and_ask_assistant(self, mic_data: list) -> str:
        print(type(mic_data))
        request: str = self.listen_for_request()
        if not request:
            return "No request received."
        response: str = self.ask(request)
        self._tts_engine.say(response)
        self._tts_engine.runAndWait()
        return response

    def listen_for_request(self) -> str:
        return self.__speech_recognizer.listen_for_request()

    def ask(self, request: str) -> str:
        prompt = f"User: {request}\nAssistant:"
        self._add_user_message(prompt)
        response: str = self._query().strip()
        return response

if __name__ == "__main__":
    personal_ai_assistant: PersonalAIAssistant = PersonalAIAssistant()
    personal_ai_assistant_interface: gr.Interface = gr.Interface(
        fn=personal_ai_assistant.listen_for_request_and_ask_assistant,
        inputs=gr.Audio(sources="microphone"),
        outputs=gr.Textbox(),
        title="Personal AI Assistant",
        description="A personal AI assistant that listens to your requests and provides answers.",
    )
    personal_ai_assistant_interface.launch()