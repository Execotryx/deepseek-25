from deepseek_connector import DeepSeekR1LocalConnector
import gradio as gr
from enum import Enum

class EmailTone(Enum):
    FORMAL = "formal"
    INFORMAL = "informal"
    FRIENDLY = "friendly"

class AIEmailResponseGenerator(DeepSeekR1LocalConnector):

    def __format_for_prompt(self, content: str) -> str:
        return f"{self.NEWLINE_SYNTACTIC_SEPARATOR_FOR_PROMPT}{content}{self.NEWLINE_SYNTACTIC_SEPARATOR_FOR_PROMPT}"
    
    def __init__(self, tone: EmailTone = EmailTone.FORMAL) -> None:
        examples = (
            f"1. Email:{self.__format_for_prompt('I hope this email finds you well.')}\n"
            f"\tResponse to the email:{self.__format_for_prompt('Thank you for your kind words. I hope you are doing well too.')}\n"
            f"2. Email:{self.__format_for_prompt('Can you provide an update on the project?')}\n"
            f"\tResponse to the email:{self.__format_for_prompt('Sure, I will get back to you with the latest updates shortly.')}\n"
            f"3. Email:{self.__format_for_prompt('Let\'s catch up over coffee next week.')}\n"
            f"\tResponse to the email:{self.__format_for_prompt('That sounds great! Let me know your available times.')}\n"
            f"4. Email:{self.__format_for_prompt('I have attached the report for your review.')}\n"
            f"\tResponse to the email:{self.__format_for_prompt('Thank you for sharing the report. I will review it and get back to you soon.')}\n"
            f"5. Email:{self.__format_for_prompt('Hey, just want an update on the project\'s progress')}\n"
            f"\tResponse to the email:{self.__format_for_prompt('Sure! The project is on track and we are making good progress. I will send you a detailed update by the end of the day.')}\n"
            f"6. Email:{self.__format_for_prompt('I appreciate your help with this matter.')}\n"
            f"\tResponse to the email:{self.__format_for_prompt('You are welcome! I am glad I could assist you.')}\n"
            f"7. Email:{self.__format_for_prompt('Can we schedule a meeting to discuss this further?')}\n"
            f"\tResponse to the email:{self.__format_for_prompt('Absolutely! Please let me know your available times and I will arrange the meeting.')}\n"
        )
        super().__init__(
            model_id="deepseek-r1:1.5b",
            system_behavior=(
                "You are an AI assistant that generates content for an email, that will serve as a response, based on the provided content of the email for which a response is requested."
                f"The tone of the response should be adjusted based on the specified tone: {tone.value}\n"
                "Refer to these examples as a baseline to start with:\n"
                f"{examples}"
                "Your response should contain ONLY the new response and nothing more.\n\n"
            )
        )

    def ask(self, request: str) -> str:
        prompt: str = (
            f"Email:{self.__format_for_prompt(request)}"
            "\tResponse to the email:\n"
        )
        self._add_user_message(prompt)
        return self._query()

if __name__ == "__main__":
    email_generator = AIEmailResponseGenerator(tone=EmailTone.FRIENDLY)
    email_generator_interface = gr.Interface(
        fn=email_generator.ask,
        inputs=gr.Textbox(label="Email Content", placeholder="Enter the email content you want to respond to..."),
        outputs=gr.Textbox(label="Generated Email Response", placeholder="The generated email response will appear here..."),
        title="AI Email Response Generator",
        description="This application uses the DeepSeek R1 model to generate email responses. Enter the email content you want to respond to in the input box and click 'Submit' to get the generated response.",
    )
    email_generator_interface.launch()