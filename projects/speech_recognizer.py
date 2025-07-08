from io import BytesIO
import speech_recognition as sr
import numpy as np

class SpeechRecognizer:
    def listen_for_request(self, audio: list[int] | tuple[int, np.ndarray] | sr.AudioData | None = None) -> str:
        recognizer: sr.Recognizer = sr.Recognizer()
        if audio is None:
            with sr.Microphone() as micro:
                recognizer.adjust_for_ambient_noise(micro)
                audio = recognizer.listen(micro)
            recognized_audio = recognizer.recognize_google(audio)
        elif isinstance(audio, tuple):
            sample_width: int = audio[1].itemsize
            audio_data = sr.AudioData(audio[1].tobytes(), audio[0], sample_width)
            recognized_audio = recognizer.recognize_google(audio_data)
        elif isinstance(audio, sr.AudioData):
            recognized_audio = recognizer.recognize_google(audio)
        else:
            raise ValueError("Unsupported audio type. Must be list, AudioData, or None.")
        return recognized_audio

if __name__ == "__main__":
    recognizer = SpeechRecognizer()
    print("Listening for request...")
    text = recognizer.listen_for_request()
    print(f"Recognized text: {text}")