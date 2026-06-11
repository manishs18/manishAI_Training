from pathlib import Path
import sys

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval import save_ticket_eval
from qa_bot import build_qa_chain
from summarizer import summarize
from ticket_classifier import classify_ticket


def load_snippets(snippets_path):
    text = Path(snippets_path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def run():
    data_dir = PROJECT_DIR / "data"
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    tickets = pd.read_csv(data_dir / "tickets.csv")
    ticket_results = []

    for _, row in tickets.iterrows():
        classification = classify_ticket(row["ticket"])
        ticket_results.append(
            {
                "ticket": row["ticket"],
                "ground_truth": row["ground_truth"],
                "pred_label": classification["category"],
                "reasoning": classification.get("reasoning", ""),
                "confidence": classification.get("confidence", ""),
                "sla_hours": classification.get("sla_hours", ""),
            }
        )

    save_ticket_eval(ticket_results, output_dir / "ticket_eval.csv")

    snippets = load_snippets(data_dir / "earnings_snippets" / "snippets.txt")
    summaries = summarize(snippets)
    pd.DataFrame({"snippet": snippets, "summary": summaries}).to_csv(
        output_dir / "summaries.csv",
        index=False,
    )

    qa_chain = build_qa_chain(data_dir / "policy_docs" / "Day4_coding_Assignment.pdf")
    questions = [
        "What is the policy document about?",
        "How should support issues be escalated?",
    ]
    answers = []

    for question in questions:
        answer = qa_chain.run(question) if hasattr(qa_chain, "run") else qa_chain(question)
        answers.append({"question": question, "answer": answer})

    pd.DataFrame(answers).to_csv(output_dir / "qa_answers.csv", index=False)
    print("Fintech NLP prototype completed successfully")


if __name__ == "__main__":
    run()