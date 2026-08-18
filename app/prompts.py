"""System prompt and context formatting.

The prompt went through a few rounds against a set of failure cases. The rules
that mattered, in order of how much they helped:

1. Numbered [1], [2] source markers -- unnumbered context produced answers that
   cited nothing, or invented document names.
2. An explicit "say you don't know" instruction -- without it the model happily
   answered general-knowledge questions from pretraining, which is exactly the
   off-topic behaviour grounding is supposed to prevent.
3. Putting the question *after* the context -- with the question first, long
   contexts pushed it out of the model's focus.
"""

SYSTEM_PROMPT = """You are an internal knowledge assistant. Answer questions \
using ONLY the numbered sources provided below.

Rules:
- Base every factual statement on the sources. Do not use outside knowledge.
- Cite the sources you used inline with their numbers, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly:
  "I couldn't find that in the indexed documents."
  Do not guess, and do not fall back on general knowledge.
- If the sources disagree, say so and cite both.
- Be concise and factual. No preamble, no restating the question."""

NO_ANSWER = "I couldn't find that in the indexed documents."


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as a numbered source block."""
    blocks = []
    for position, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{position}] (source: {chunk['source']})\n{chunk['content'].strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, chunks: list[dict]) -> str:
    return (
        "SOURCES\n"
        "=======\n"
        f"{format_context(chunks)}\n\n"
        "QUESTION\n"
        "========\n"
        f"{question}"
    )
