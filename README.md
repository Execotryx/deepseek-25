# deepseek-25

A collection of small Python/Gradio applications built around a local DeepSeek R1 model served through [Ollama](https://ollama.com/).

The repository is structured as a learning playground: one reusable connector handles the local model call and chat history, while each project file demonstrates a different AI use case such as summarization, grammar correction, sentiment analysis, named entity extraction, email replies, customer support routing, resume generation, and a voice-based personal assistant.

## What is inside

```text
deepseek-25/
├── requirements.txt
└── projects/
    ├── deepseek_connector.py
    ├── summarizer.py
    ├── grammar_checker.py
    ├── sentiment-analysis.py
    ├── ner_extractor.py
    ├── text_generator.py
    ├── ai-email-response-generator.py
    ├── customer_support.py
    ├── ai-resume-generator.py
    ├── personal-assistant.py
    └── speech_recognizer.py
```

## Core idea

`projects/deepseek_connector.py` defines `DeepSeekR1LocalConnector`, a base class for local DeepSeek R1 integrations. It:

- stores the system prompt and chat history;
- calls Ollama's `chat()` API;
- uses `deepseek-r1:8b` as the default model;
- strips DeepSeek R1 `<think>...</think>` blocks from responses;
- provides helper methods for formatting prompts and resetting history.

The other scripts inherit from this connector and customize the prompt, model, and interface for a specific task.

## Included mini-apps

| File | Purpose | Interface |
|---|---|---|
| `summarizer.py` | Summarizes user-provided text and returns a concise summary plus main subjects. | Gradio |
| `grammar_checker.py` | Corrects grammar while preserving the original meaning. | Gradio |
| `sentiment-analysis.py` | Classifies text as positive, negative, or neutral. | Gradio |
| `ner_extractor.py` | Extracts named entities in a SpaCy-like labelled bullet-list format. | Gradio |
| `text_generator.py` | Generates Markdown-compatible text from a request and word limit. | Gradio Blocks |
| `ai-email-response-generator.py` | Generates an email reply in a selected tone. | Gradio |
| `customer_support.py` | Maps customer questions to a small built-in FAQ database and returns the matching answer. | Gradio Blocks |
| `ai-resume-generator.py` | Demonstrates LinkedIn-profile information extraction and resume generation prompts. | CLI/demo code |
| `personal-assistant.py` | Voice-enabled assistant: microphone input, speech recognition, DeepSeek response, and text-to-speech output. | Gradio |
| `speech_recognizer.py` | Helper wrapper around `speech_recognition` for microphone or Gradio audio input. | CLI/helper |

## Requirements

You need:

- Python 3.10 or newer;
- [Ollama](https://ollama.com/) installed and running;
- at least one DeepSeek R1 model pulled locally;
- a working microphone if you want to use the voice assistant;
- PortAudio/PyAudio system support for microphone input.

Python dependencies are listed in `requirements.txt`:

```text
ollama
gradio
speechrecognition
pyttsx3
faster-whisper
pyaudio
```

> Note: `faster-whisper` is currently listed as a dependency, but the existing speech recognizer implementation uses `speech_recognition` with Google's recognizer backend.

## Setup

Clone the repository:

```bash
git clone https://github.com/Execotryx/deepseek-25.git
cd deepseek-25
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install/pull the Ollama models used by the project:

```bash
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:1.5b
```

Make sure Ollama is running:

```bash
ollama serve
```

Depending on your Ollama installation, the service may already be running in the background.

## Running the apps

Run any app from the repository root:

```bash
python projects/summarizer.py
```

Then open the local Gradio URL printed in the terminal.

Common commands:

```bash
python projects/summarizer.py
python projects/grammar_checker.py
python projects/sentiment-analysis.py
python projects/ner_extractor.py
python projects/text_generator.py
python projects/ai-email-response-generator.py
python projects/customer_support.py
python projects/personal-assistant.py
```

For the speech recognizer helper:

```bash
python projects/speech_recognizer.py
```

## Model usage

Most applications use the default connector model:

```text
deepseek-r1:8b
```

Some smaller task-specific demos use:

```text
deepseek-r1:1.5b
```

Specifically:

- `customer_support.py` uses `deepseek-r1:1.5b`;
- `ai-email-response-generator.py` uses `deepseek-r1:1.5b`;
- `personal-assistant.py` explicitly uses `deepseek-r1:8b`;
- the base connector defaults to `deepseek-r1:8b`.

You can change the model by passing another `model_id` to `DeepSeekR1LocalConnector` or by editing the relevant subclass constructor.

## Example: using the connector directly

```python
from projects.deepseek_connector import DeepSeekR1LocalConnector


class MyDeepSeekTool(DeepSeekR1LocalConnector):
    def ask(self, request: str) -> str:
        self._add_user_message(request)
        return self._query()


tool = MyDeepSeekTool(
    system_behavior="You are a concise technical assistant.",
    model_id="deepseek-r1:8b",
)

print(tool.ask("Explain what Gradio is in one paragraph."))
```

## Notes and limitations

- The project is designed for local experimentation rather than production deployment.
- The LinkedIn crawler in `ai-resume-generator.py` is demonstration-only. LinkedIn generally does not allow simple unauthenticated scraping, so a production implementation would need proper authorization and compliance handling.
- The voice assistant uses local DeepSeek inference, but speech-to-text is currently handled through `speech_recognition.recognize_google`, which requires network access.
- There are no automated tests yet.
- There is no license file in the repository at the moment. Add a license before encouraging reuse or redistribution.

## Suggested next improvements

- Add a `.env.example` or configuration file for model names and runtime options.
- Move common app-launching logic into a shared helper.
- Add input validation and error handling around Ollama availability.
- Add tests for prompt formatting, chat history handling, and `<think>` tag stripping.
- Replace or supplement Google speech recognition with the listed `faster-whisper` dependency for local speech-to-text.
- Add screenshots or GIFs of each Gradio app.
- Add a license file.

## Status

This repository is best read as a compact portfolio of local DeepSeek R1 experiments: each script demonstrates a practical GenAI use case while sharing a simple common connector for Ollama-based inference.
