import json

MAX_QUIZ_SOURCE_CHARS = 14000
MAX_QUIZ_QUESTIONS = 15
QUIZ_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_quiz",
        "description": (
            "Generate a multiple-choice quiz from one of the user's "
            "documents. Use this when the user asks to be quizzed, tested, "
            "or wants practice questions. Do not use it to answer questions "
            "about content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": (
                        "Filename of the document to build the quiz from, "
                        "as returned by list_documents."
                    ),
                },
                "num_questions": {
                    "type": "integer",
                    "description": "How many questions. Default 5, max 15.",
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Optional. Narrow the quiz to one topic within the "
                        "document, e.g. 'dynamic programming'. Omit to cover "
                        "the whole document."
                    ),
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": "Default medium.",
                },
            },
            "required": ["filename"],
        },
    },
}


QUIZ_SYSTEM_PROMPT = """You write multiple-choice quizzes from source material.

Return ONLY a JSON object. No preamble, no explanation, no markdown fences.

Shape:
{
  "questions": [
    {
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "Why this answer is correct, referencing the source."
    }
  ]
}

Rules:
- Base every question ONLY on the provided text. Invent nothing.
- Exactly 4 options per question. Exactly one is correct.
- Wrong options must be plausible and related to the topic, not obviously
  silly. Vary which index is correct.
- Keep questions self-contained: do not write "according to the passage".
- If the text does not support the requested number of questions, return
  fewer rather than padding.
"""


def _extract_json(text):
    """Models sometimes wrap JSON in fences or prose. Pull out the object."""
    text = (text or "").strip()

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")

    return json.loads(text[start : end + 1])


def _validate_questions(raw_questions, limit):
    """Drop anything malformed rather than trusting the model."""
    clean = []

    for item in raw_questions:
        if not isinstance(item, dict):
            continue

        question = item.get("question")
        options = item.get("options")
        index = item.get("correct_index")

        if not question or not isinstance(options, list):
            continue
        if len(options) < 2:
            continue

        try:
            index = int(index)
        except (TypeError, ValueError):
            continue

        if not 0 <= index < len(options):
            continue

        clean.append(
            {
                "question": str(question),
                "options": [str(option) for option in options],
                "correct_index": index,
                "explanation": str(item.get("explanation", "")),
            }
        )

        if len(clean) >= limit:
            break

    return clean

def _load_document_text(self, document_id, limit_chars):
    """Shared by the summarizer and the quiz generator."""
    response = (
        self.supabase.table("document_chunks")
        .select("content, pagenumber")
        .eq("document_id", document_id)
        .eq("userId", self.user_id)
        .execute()
    )

    rows = response.data or []
    if not rows:
        return ""

    rows.sort(key=lambda row: row.get("pagenumber") or 0)

    text = ""
    for row in rows:
        piece = f"[Page {row['pagenumber']}]\n{row['content']}\n\n"
        if len(text) + len(piece) > limit_chars:
            break
        text += piece

    return text


def tool_generate_quiz(
    self,
    filename,
    num_questions=5,
    topic=None,
    difficulty="medium",
):
    document = self._resolve_document(filename)
    if document is None:
        return (
            f"Could not find a document called '{filename}'.\n"
            + self.tool_list_documents()
        )

    try:
        num_questions = int(num_questions)
    except (TypeError, ValueError):
        num_questions = 5
    num_questions = max(1, min(num_questions, MAX_QUIZ_QUESTIONS))

    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"

    source = self._load_document_text(
        document["id"],
        MAX_QUIZ_SOURCE_CHARS,
    )
    if not source:
        return "That document has no extracted text to build a quiz from."

    focus = (
        f"Focus the questions on: {topic}."
        if topic
        else "Cover the document broadly."
    )

    instruction = (
        f"Write {num_questions} {difficulty}-difficulty questions. "
        f"{focus}\n\nSource text:\n\n{source}"
    )

    questions = []
    last_error = None

    for _ in range(2):
        try:
            response = self.groq.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                reasoning_format="hidden",
                temperature=0.4,
            )
            parsed = _extract_json(response.choices[0].message.content)
            questions = _validate_questions(
                parsed.get("questions", []),
                num_questions,
            )
            if questions:
                break
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e

    if not questions:
        return (
            "Could not generate a valid quiz from that document."
            + (f" ({last_error})" if last_error else "")
        )

    self.last_quiz = {
        "filename": document["original_filename"],
        "topic": topic,
        "difficulty": difficulty,
        "questions": questions,
    }

    preview = "\n".join(
        f"{i + 1}. {question['question']}"
        for i, question in enumerate(questions)
    )

    return (
        f"Generated {len(questions)} {difficulty} questions from "
        f"{document['original_filename']}. The quiz is displayed to the user "
        f"below — tell them it is ready and do not repeat the questions or "
        f"reveal any answers.\n\nQuestions asked:\n{preview}"
    )
