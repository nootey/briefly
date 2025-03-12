# Briefly

Briefly is an effective way to transcribe and summarize your meetings, to create actionable steps and follow-ups, by leveraging AI.

<hr>

## Notice. 

⚠️ This was a research project made in collaboration with my employer. It also got made redundant with the launch of OpenAIs web agent research API. 

## Setup

### Environment

Create a virtual enviroment for Python
```shell
python3 -m venv venv
```

Activate it with
```shell
source venv/bin/activate
```

### Dependencies

#### Services
You need to get api keys for the following services:
- [OpenAI API](https://platform.openai.com/api-keys)
  - Must have for text correction and data fetching.

### Configuration

The application can be configured through environment variables, by creating a .env file in project root. The following options can be configured:
```
OPENAI_API_KEY="key"
OLLAMA_HOST="http://localhost:11434"
```

### LLM Specific
Check if CUDA is working (for GPU Acceleration)
```shell
python3 -c "import torch; print(torch.cuda.is_available())"
```

Install Ollama for WSL:
```shell
curl -fsSL https://ollama.com/install.sh | sh
```
Verify installation:
`ollama --version`

In order to use the models, you need to pull them with `ollama pull <name>`. 

## Notes

- This was done as an attempt in researching capabilities of different LLMs,
- Speaker diarization is missing,
- It's not optimized,
- The output is rather generic, meaning that the app does not include company values and context as of yet,


