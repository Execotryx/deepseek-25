from deepseek_connector import DeepSeekR1LocalConnector
import gradio as gr
import requests

class LinkedInCrawler(DeepSeekR1LocalConnector):
    def __init__(self) -> None:
        """Initializes the LinkedInCrawler with a system behavior for extracting LinkedIn profile information."""
        super().__init__(system_behavior=(
            "You are a helpful assistant specialized in crawling LinkedIn profiles."
            "Your goal is to help users extract relevant information from LinkedIn profiles."
            "Before you can create resume - you will need to extract all relevant information from the LinkedIn profile."
            "The mentioned information includes general information, education, experience, skills, and certifications."
            "It will accumulated in the chat history."
            "If some information from mentioned sections is not available, you will stop immediately."
            "Your response will be ONLY the extracted information in Markdown format."
        ))

    def __get_general_information(self, profile_url: str) -> None:
        """Fetches and extracts general information from the given LinkedIn profile URL."""
        response = requests.get(profile_url)
        if response.status_code == 200:
            general_page: str = response.text
            prompt: str = (
                f"Clean this page of LinkedIn profile from HTML:{self._format_for_prompt(general_page)}"
                "Respond ONLY with the general information, nothing more."
            )
            self._add_user_message(prompt)
            general_information: str = self._query()
            self._add_assistant_message(f"Profile:{self._format_for_prompt(general_information)}")
        else:
            self._add_assistant_message("No general information available.")

    def __get_skills_information(self, profile_url: str) -> None:
        """Fetches and extracts skills information from the LinkedIn profile's skills page."""
        response = requests.get(f"{profile_url}/details/skills")
        if response.status_code == 200:
            skills_page: str = response.text
            prompt: str = (
                f"Extract the information about skills from this LinkedIn skills page:{self._format_for_prompt(skills_page)}"
                "Respond ONLY with the information of skills, nothing more."
            )
            self._add_user_message(prompt)
            skills_information: str = self._query()
            self._add_assistant_message(f"Skills:{self._format_for_prompt(skills_information)}")
        else:
            self._add_assistant_message("No skills information available.")

    def __get_certifications_information(self, profile_url: str) -> None:
        """Fetches and extracts certifications information from the LinkedIn profile's certifications page."""
        response: requests.Response = requests.get(f"{profile_url}/details/certifications")
        if response.status_code == 200:
            certifications_page: str = response.text
            prompt: str = (
                f"Extract the information about certifications from this LinkedIn certifications page:{self._format_for_prompt(certifications_page)}"
                "Respond ONLY with the information of certifications, nothing more."
            )
            self._add_user_message(prompt)
            certifications_information: str = self._query()
            self._add_assistant_message(f"Certifications:{self._format_for_prompt(certifications_information)}")
        else:
            self._add_assistant_message("No certifications information available.")

    def __get_experience_information(self) -> None:
        """Extracts experience information from the LinkedIn profile data stored in the chat history."""
        prompt: str = (
            f"Extract the information about experience from the LinkedIn profile, stored in this chat.\n"
            "Respond ONLY with the information of experience, nothing more."
        )
        self._add_user_message(prompt)
        experience_information: str = self._query()
        self._add_assistant_message(f"Experience:{self._format_for_prompt(experience_information)}")

    def __get_education_information(self) -> None:
        """Extracts education information from the LinkedIn profile data stored in the chat history."""
        prompt: str = (
            "Extract the information about education from the LinkedIn profile, stored in this chat.\n"
            "Respond ONLY with the information of education, nothing more."
        )
        self._add_user_message(prompt)
        education_information: str = self._query()
        self._add_assistant_message(f"Education:{self._format_for_prompt(education_information)}")

    def ask(self, request: str) -> str:
        """Orchestrates the extraction of all relevant LinkedIn profile information and returns a summary."""
        self.__get_general_information(request)
        self.__get_education_information()
        self.__get_experience_information()
        self.__get_skills_information(request)
        self.__get_certifications_information(request)

        prompt: str = f"Basing on the history in this chat, summarize the information in this profile."
        self._add_user_message(prompt)
        general_information = self._query()
        return general_information

class AIResumeGenerator(DeepSeekR1LocalConnector):
    def __init__(self) -> None:
        super().__init__(system_behavior=(
            "You are a helpful assistant specialized in generating resumes."
            "Your goal is to help users create professional resumes based on their input and preferences."
            "Your response will be ONLY a professional, well-structured, ATS-compliant resume in Markdown format."
        ))

    def ask(self, request: str) -> str:
        prompt: str = f"Generate a professional resume based on the following information:{self._format_for_prompt(request)}"
        self._add_user_message(prompt)
        return self._query()

if __name__ == "__main__":
    crawler: LinkedInCrawler = LinkedInCrawler()
    info: str = crawler.ask("https://www.linkedin.com/in/viktoriya-perepletkina-934a26252")
    print(info)