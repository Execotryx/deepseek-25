from io import BytesIO
import speech_recognition as sr

class SpeechRecognizer:
    def listen_for_request(self) -> str:
        recognizer: sr.Recognizer = sr.Recognizer()
        with sr.Microphone() as micro:
            recognizer.adjust_for_ambient_noise(micro)
            audio = recognizer.listen(micro)
            recognized_audio = recognizer.recognize_google(audio)
        return recognized_audio

if __name__ == "__main__":
    recognizer = SpeechRecognizer()
    print("Listening for request...")
    text = recognizer.listen_for_request()
    print(f"Recognized text: {text}")