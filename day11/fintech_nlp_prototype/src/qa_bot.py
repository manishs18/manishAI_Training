from dataclasses import dataclass
from pathlib import Path

from groq_utils import chat_completion


@dataclass
class LocalQAChain:
    chunks: list[str]

    def run(self, query):
        query_terms = {term for term in query.lower().split() if len(term) > 2}
        best_chunk = ""
        best_score = -1

        for chunk in self.chunks:
            score = sum(1 for term in query_terms if term in chunk.lower())
            if score > best_score:
                best_chunk = chunk
                best_score = score

        if not best_chunk:
            return "No relevant policy text could be extracted locally."

        return best_chunk[:300]


def _extract_pdf_text(pdf_path):
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return ""

def build_qa_chain(pdf_path):
    pdf_path = Path(pdf_path)

    text = _extract_pdf_text(pdf_path)
    if not text:
        return LocalQAChain(["Policy document text could not be extracted locally."])

    chunks = []
    chunk_size = 500
    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    class GroqQAChain(LocalQAChain):
        def run(self, query):
            context = super().run(query)
            try:
                return chat_completion(
                    prompt=f"""
Use the context below to answer the question briefly and accurately.

Context:
{context}

Question:
{query}
""",
                    system="You answer questions about a policy document using only the provided context.",
                )
            except Exception:
                return context

    return GroqQAChain(chunks or [text[:300]])