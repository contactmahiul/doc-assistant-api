import logging
import asyncio
import json

from app.utils.llm import get_groq_client

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """You are a strict fact-checker.
Given a CONTEXT and an ANSWER, score how well the answer is supported by the context.
Rules:
- Score 1.0 if every claim in the answer is directly supported by the context
- Score 0.5 if most claims are supported but some are vague or slightly beyond context
- Score 0.0 if the answer contains claims not found in the context, or contradicts it
- Output ONLY a JSON object like: {{"faithfulness": 0.8, "reason": "one sentence"}}
- No preamble, no markdown, just the JSON
CONTEXT:
{context}
ANSWER:
{answer}"""

FALLBACK_ANSWER = (
    "I was unable to produce a fully supported answer from the provided documents. "
    "Please rephrase your question or consult the source material directly."
)

FAITHFULNESS_THRESHOLD = 0.5  


def _check(answer: str, chunks: list[str]) -> dict:
    context = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(chunks)])
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)

    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
        score = float(result.get("faithfulness", 0.0))
        reason = result.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"Faithfulness parse failed raw='{raw}' — defaulting score=0.5")
        score = 0.5           
        reason = "parse error"

    logger.info(f"Faithfulness | score={score} reason='{reason}'")
    return {"score": score, "reason": reason}


async def faithfulness_check(answer: str, chunks: list[str]) -> dict:
    return await asyncio.to_thread(_check, answer, chunks)