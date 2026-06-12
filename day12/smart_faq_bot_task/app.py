import os
import json
import urllib.request
from langchain_community.vectorstores import FAISS

class LocalEmbeddings:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str):
        vector = [0.0] * self.dim
        for token in text.lower().split():
            index = hash(token) % self.dim
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    def __call__(self, text):
        return self.embed_query(text)

OPENAI_API_KEY = "sk-p............................"
GROQ_API_KEY = "gsk....."

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

FAQS = [
    "How do I reset my password? - Use the reset link on the login page.",
    "How can I update my profile? - Open settings and edit your details.",
    "How do I contact support? - Email support@example.com or use chat.",
    "What payment methods are accepted? - Credit card, debit card, and UPI.",
    "How do I track my order? - Visit the orders page and select track order.",
    "Can I cancel my subscription? - Yes, from billing settings before renewal.",
    "How do I change my email address? - Go to account settings and update email.",
    "Where can I download invoices? - In the billing section under invoices.",
    "How do I enable two-factor authentication? - Turn it on from security settings.",
    "What is the refund policy? - Refunds are available within 7 days.",
]

embeddings = LocalEmbeddings()
store = FAISS.from_texts(FAQS, embeddings)


def ask_groq(context: str, query: str) -> str:
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Answer only from the provided FAQs. Be concise."},
            {"role": "user", "content": f"FAQs:\n{context}\n\nQuestion: {query}"},
        ],
        "max_tokens": 180,
        "temperature": 0,
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def local_answer(context: str, query: str) -> str:
    return (
        "Based on the nearest FAQs, the best answer is: "
        + context.split("-", 1)[-1].strip()
        + f"\n\nQuestion: {query}"
    )

while True:
    query = input("\nAsk a question (or type 'exit'): ").strip()
    if query.lower() == "exit":
        break
    docs = store.similarity_search(query, k=2)
    context = "\n".join(f"- {doc.page_content}" for doc in docs)
    try:
        answer = ask_groq(context, query)
    except Exception:
        answer = local_answer(context, query)
    print("\nAnswer:", answer)
    print("Sources:")
    for doc in docs:
        print("-", doc.page_content)
