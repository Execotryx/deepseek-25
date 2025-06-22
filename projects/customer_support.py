from deepseek_connector import DeepSeekR1LocalConnector
import gradio as gr

class CustomerSupportBot(DeepSeekR1LocalConnector):
    __FAQ_DATABASE: dict[str, str] = {
        "return_policy": "Our return policy allows you to return items within 30 days of purchase for a full refund.",
        "shipping_info": "We offer free shipping on orders over $50. Standard shipping takes 5-7 business days.",
        "product_warranty": "All our products come with a one-year warranty covering manufacturing defects.",
        "customer_service_hours": "Our customer service is available Monday to Friday from 9 AM to 5 PM EST.",
        "payment_methods": "We accept all major credit cards, PayPal, and Apple Pay.",
        "technical_support": "For technical support, please visit our help center or contact us via email at support@example.com",
        "account_management": "You can manage your account settings, including password changes and email preferences, in the account section of our website.",
        "privacy_policy": "We take your privacy seriously. Please read our privacy policy on our website for more information.",
        "product_availability": "You can check product availability on our website. If an item is out of stock, you can sign up for restock notifications.",
    }

    def __init__(self):
        categories = "\n".join(self.__FAQ_DATABASE.keys())
        examples = (
            '"""\n'
            'Example 1:\n'
            'Question: What is your return policy?\n'
            'Matching category: return_policy\n\n'
            'Example 2:\n'
            'Question: What are your customer service hours?\n'
            'Matching category: customer_service_hours\n\n'
            'Example 3:\n'
            'Question: How can I manage my account settings?\n'
            'Matching category: account_management\n\n'
            'Example 4:\n'
            'Question: could you please tell me about shipping?\n'
            'Matching category: shipping_info\n'
            '"""'
        )
        system_behavior = (
            "You are a customer support agent. You will find a most appropriate category present currently in the FAQ database. "
            "Categories of questions include strictly the following:\n"
            f"{categories}\n"
            "You will respond only with the most appropriate category, nothing more.\n"
            "Here are some examples of response:\n"
            f"{examples}"
        )
        super().__init__(model_id="deepseek-r1:1.5b", system_behavior=system_behavior)

    def ask(self, request: str) -> str:
        prompt = (
            f"Question: {request}\n"
            "Matching category: "
        )
        self._add_user_message(prompt)
        category_of_question: str = self._query().strip().lower()
        return self.__FAQ_DATABASE.get(category_of_question, "Sorry, I can't assist with that.")


def customer_support_interface():
    chatbot = CustomerSupportBot()
    with gr.Blocks() as demo:
        gr.Markdown("## Customer Support Chatbot")
        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(label="Ask a question:")
                submit_btn = gr.Button("Submit")
            with gr.Column():
                chatbot_output = gr.Markdown("")

        submit_btn.click(fn=chatbot.ask, inputs=user_input, outputs=chatbot_output)

    demo.launch()

if __name__ == "__main__":
    customer_support_interface()
