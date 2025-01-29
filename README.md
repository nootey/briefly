<p align="center" style="padding: 25px">
    briefly - Transcribe meetings into text and provide impactful summaries and action steps.
</p>


Briefly is an effective way to transcribe and summarize your meetings, to create actionable steps and follow-ups, by leveraging AI.

<hr>

Create a virtual enviroment for Python
```shell
python3 -m venv venv/briefly
```
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

For now, I'll be using mistral.


In order to use the pipeline to diarize speakers with whisper, you need 
Install pyannote.audio with pip install pyannote.audio
Accept pyannote/segmentation-3.0 user conditions
https://hf.co/pyannote/segmentation-3.0

Accept pyannote/speaker-diarization-3.1 user conditions
https://hf.co/pyannote/speaker-diarization-3.1

Create access token at hf.co/settings/tokens.

