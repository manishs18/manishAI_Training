import re

try:
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover - optional dependency
    rouge_scorer = None

few_shot_examples = """
Example 1:
Text: Revenue grew 12% YoY driven by payments segment...
Summary: Revenue increased 12% YoY led by payments growth.

Example 2:
Text: EBITDA declined due to higher compliance costs...
Summary: EBITDA dropped due to increased compliance expenses.
"""

def _local_summary(text):
    text = text.strip()
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if sentences:
        return sentences[0][:180].strip()
    return text[:180].strip()

def summarize(texts):
    summaries = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if rouge_scorer is not None else None
    
    for i, text in enumerate(texts):
        mode = "ZERO-SHOT MODE" if i==0 else "FEW-SHOT MODE"
        examples = "" if i==0 else few_shot_examples
        try:
            from groq_utils import chat_completion

            summary = chat_completion(
                prompt=f"""
{mode}

{examples}

Summarize the following earnings call snippet in 1-2 lines.

Text:
{text}
""",
                system="You are a financial analyst assistant.",
            )
        except Exception:
            summary = _local_summary(text)
        summaries.append(summary)
        # Optional: compute ROUGE-L if reference available
    return summaries