from groq import Groq
from app.core.config import settings

_client = None

def get_groq_client()->Groq:
    global _client

    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)

    return _client


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n".join([
        f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)
    ])
    return f"""You are a helpful assistant. Answer the question using only the context provided below.
If the answer is not in the context, say "I could not find relevant information in the documents."

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, chunks: list[str]) -> str:
    client = get_groq_client()
    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.1   
    )

    return response.choices[0].message.content