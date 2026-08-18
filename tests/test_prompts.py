from app.prompts import NO_ANSWER, SYSTEM_PROMPT, build_user_message, format_context

CHUNKS = [
    {"content": "Claims must be submitted within 30 days.", "source": "expense.md"},
    {"content": "Alcohol is not reimbursable.", "source": "expense.md"},
]


def test_context_is_numbered_from_one():
    rendered = format_context(CHUNKS)
    assert rendered.startswith("[1]")
    assert "[2]" in rendered
    assert "(source: expense.md)" in rendered


def test_user_message_puts_question_last():
    message = build_user_message("How long do I have?", CHUNKS)
    assert message.index("SOURCES") < message.index("QUESTION")
    assert message.rstrip().endswith("How long do I have?")


def test_system_prompt_carries_the_refusal_string():
    assert NO_ANSWER in SYSTEM_PROMPT


def test_empty_context_renders_empty():
    assert format_context([]) == ""
