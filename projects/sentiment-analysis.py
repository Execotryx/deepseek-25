from deepseek_connector import DeepSeekR1LocalConnector
import gradio as gr

class DeepSeekR1SentimentAnalyzer(DeepSeekR1LocalConnector):

    def __init__(self) -> None:
        super().__init__(system_behavior=("You are a sentiment analysis assistant. "
                                          "Your task is to analyze the sentiment of the given text.\n"
                                          "You will respond ONLY with the sentiment (positive, negative, neutral) of a provided text.\n"))


    def ask(self, request: str) -> str:
        """
        Analyzes the sentiment of the given text and returns a summary of findings.

        Args:
            request (str): The text to analyze for sentiment.

        Returns:
            str: The sentiment analysis result.
        """
        prompt = f"Analyze the sentiment of the following text:\n\n{request}\n\n"
        self._add_user_message(prompt)
        return self._query()

sentiment_analyzer: DeepSeekR1SentimentAnalyzer = DeepSeekR1SentimentAnalyzer()

if __name__ == "__main__":
    sentiment_analyzer_interface: gr.Interface = gr.Interface(
        fn=sentiment_analyzer.ask,
        inputs=gr.Textbox(label="Text to Analyze", placeholder="Enter the text you want to analyze here..."),
        outputs=gr.Textbox(label="Sentiment Analysis Result", placeholder="The sentiment analysis result will appear here..."),
        title="DeepSeek R1 Sentiment Analyzer",
        description="This application uses the DeepSeek R1 model to analyze sentiment. Enter the text you want to analyze in the input box and click 'Submit' to get the sentiment analysis result.",
    )
    sentiment_analyzer_interface.launch()
