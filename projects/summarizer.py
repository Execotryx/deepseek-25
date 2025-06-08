from deepseek_connector import DeepSeekR1LocalConnector
from ollama import chat, ChatResponse

class DeepSeekR1Summarizer(DeepSeekR1LocalConnector):
    """
    DeepSeek R1 Summarizer class that inherits from DeepSeekR1LocalConnector.
    This class is used to summarize text using the DeepSeek R1 model.
    """

    def __init__(self):
        """
        Initializes the DeepSeekR1Summarizer with the specified model name.
        """
        super().__init__(system_behavior=("You are helping user to summarize text."
                                          "You will be given a text and you need to summarize it in a concise manner."
                                          "You will respond with concise summary of a text and bullet list of main subjects covered in the text, that was provided by user."))

    def ask(self, request: str) -> str | None:
        """
        Sends a request to the LLM and returns the response.
        Args:
            request (str): The text to summarize.
        Returns:
            str: The summary of the text.
        """
        if not request or request.strip() == "":
            raise ValueError("Request cannot be empty.")
        prompt: str = (f"Please summarize the following text:\n\n{request}\n\n")
        self._add_user_message(prompt)
        response: ChatResponse = chat(model=self.MODEL_ID, messages=self._chat_history, stream=False)
        if response.message.content:
            return self._add_assistant_message(response.message.content)
        else:
            raise ValueError("No content in the response from the model.")

if __name__ == "__main__":
    summarizer = DeepSeekR1Summarizer()
    text_to_summarize = "Several open-source-friendly BCIs can be purchased outside the USA at prices well below the NexStem Instinct ($2 499). Notable options include OpenBCI (Ganglion and Cyton boards with Ultracortex headsets), NexStem 16-Channel headset, and BrainBit (MINDO headband and Callibri sensor). Each platform provides permissive SDKs (MIT, BSD-3-Clause, or BSD-style) with full raw-data access and ships globally (including Europe, Asia, and beyond). Additionally, DIY boards like FreeEEG32 and hybrid systems such as g.tec’s Unicorn Hybrid are available internationally, though Unicorn Hybrid is more expensive. Below, we outline each device’s key features, pricing, licensing, and shipping information to help you identify cost-effective, developer-friendly BCIs outside the USA."
    try:
        summary = summarizer.ask(text_to_summarize)
        print("Summary:", summary)
    except ValueError as e:
        print("Error:", e)