from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from rouge_score import rouge_scorer

llm = OpenAI(temperature=0)

few_shot_examples = """
Example 1:
Text: Revenue grew 12% YoY driven by payments segment...
Summary: Revenue increased 12% YoY led by payments growth.

Example 2:
Text: EBITDA declined due to higher compliance costs...
Summary: EBITDA dropped due to increased compliance expenses.
"""

prompt_template = PromptTemplate(
    input_variables=["mode", "examples", "text"],
    template="""
You are a financial analyst assistant.

{mode}

{examples}

Summarize the following earnings call snippet in 1-2 lines.

Text:
{text}

Summary:
"""
)

def summarize(texts):
    summaries = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    for i, text in enumerate(texts):
        mode = "ZERO-SHOT MODE" if i==0 else "FEW-SHOT MODE"
        examples = "" if i==0 else few_shot_examples
        prompt = prompt_template.format(mode=mode, examples=examples, text=text)
        summary = llm(prompt)
        summaries.append(summary)
        # Optional: compute ROUGE-L if reference available
    return summaries