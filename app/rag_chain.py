"""RAG chain: question -> retrieve chunks -> build prompt -> LLM -> grounded answer."""

import logging

from app.config import CHAT_MODEL, TOP_K, get_openai_client
from app.retriever import retrieve

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = (
    "I don't know. I could not find this in the automotive documents I have."
)

SYSTEM_PROMPT = """You are an assistant for an automotive software team.
You answer questions about automotive software, standards (AUTOSAR, ISO 26262, etc.)
and the vehicle user guide.

Rules:
1. Use ONLY the information inside the numbered context below.
2. If the answer is not in the context, reply exactly: "I don't know. I could not find this in the automotive documents I have."
3. Never invent facts, numbers or standards.
4. Add citations like [1] or [2] after the sentences that use that context.
5. Keep the answer clear and short. Use bullet points for steps or lists."""


def format_context(chunks: list[dict]) -> str:
    """Number each chunk so the LLM can cite it as [1], [2], ..."""
    parts = []
    for number, chunk in enumerate(chunks, start=1):
        place = chunk["source"]
        if chunk["page"]:
            place += f", page {chunk['page']}"
        parts.append(f"[{number}] (Source: {place})\n{chunk['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, top_k: int = TOP_K) -> dict:
    """Return {"answer": str, "sources": [chunk, ...]}."""
    # Step 1: find the relevant chunks
    chunks = retrieve(query, top_k)
    if not chunks:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    # Step 2: build the prompt = context + question
    user_prompt = f"Context:\n{format_context(chunks)}\n\nQuestion: {query}"

    # Step 3: ask the LLM (temperature 0 = most factual, least creative)
    client = get_openai_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "sources": chunks}
