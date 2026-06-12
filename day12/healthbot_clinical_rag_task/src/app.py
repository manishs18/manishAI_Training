from __future__ import annotations

from pathlib import Path

from prompts import build_answer
from retrieval import load_text, retrieve_chunks, split_into_chunks


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "sample_clinical_notes.txt"


def main() -> None:
    question = "What should clinicians do when a patient has a suspected drug interaction?"
    text = load_text(str(DATA_FILE))
    chunks = split_into_chunks(text)
    results = retrieve_chunks(question, chunks, top_k=3)

    print(build_answer(question, results))


if __name__ == "__main__":
    main()
