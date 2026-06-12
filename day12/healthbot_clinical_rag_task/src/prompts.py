from __future__ import annotations

from typing import List

from retrieval import ChunkScore


def build_answer(question: str, results: List[ChunkScore]) -> str:
    lines = ["Clinical RAG Assistant", "", f"Question: {question}", ""]
    lines.append("Retrieved Evidence:")
    for index, result in enumerate(results, start=1):
        snippet = result.chunk[:260].strip()
        lines.append(f"{index}. Score={result.score} | {snippet}...")
    lines.extend(["", "Suggested Answer:"])

    if not results or results[0].score == 0:
        lines.append("The local knowledge base does not contain strong evidence for this question.")
    else:
        lines.append(
            "Based on the retrieved clinical notes, the assistant should answer using the highest-scoring evidence and avoid unsupported claims."
        )

    return "\n".join(lines)
