# Day 11 Fintech NLP Prototype

This project classifies support tickets, summarizes earnings snippets, and answers questions over a policy PDF.

## Run

From the project root:

```powershell
python -m pip install -r requirements.txt
python .\src\main.py
```

The run writes CSV outputs to `outputs/`.

## Groq Key

Create your Groq API key in the Groq console, then set it in PowerShell:

```powershell
$env:GROQ_API_KEY = "your_groq_key"
```

Optional model override:

```powershell
$env:GROQ_MODEL = "llama-3.1-8b-instant"
```

## Notes

The code now falls back to local deterministic behavior when Groq is not available, so the task can run in this workspace without external model access.
