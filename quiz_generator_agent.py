import json

QUIZ_MODEL = "openai/gpt-oss-120b"

MAX_QUIZ_QUESTIONS = 15
MAX_QUIZ_STEPS = 8
READ_CHUNK_CHARS = 8000
MAX_SEARCH_CANDIDATES = 40

QUIZ_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_quiz",
        "description": (
            "Hand off to the quiz agent to build a multiple-choice quiz from "
            "one of the user's documents. Use this when the user asks to be "
            "quizzed or tested, or wants practice questions. Do not use it to "
            "answer questions about document content - use search_documents "
            "for that."
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

QUIZ_AGENT_PROMPT = """You are a quiz-writing agent. You build a multiple-choice
quiz from ONE document, using your tools to gather the source material.

Workflow:
1. Gather material.
   - If a topic is given, start with search_document for that topic. If the
     results are thin, try one or two other phrasings.
   - If no topic is given, use read_document. Call it again with the
     next_offset it gives you so the quiz covers the whole document, not
     just the beginning.
2. Write questions ONLY from text you actually retrieved with your tools.
   Invent nothing and use no outside knowledge.
3. Deliver the quiz by calling submit_quiz. If it rejects the quiz, fix the
   listed problems and call submit_quiz again with the full corrected list.

Question rules:
- Exactly 4 options per question. Exactly one is correct.
- Wrong options must be plausible and related to the topic. Vary which
  index is correct.
- Keep questions self-contained: do not write "according to the passage".
- The explanation says why the answer is correct, based on the source.
- source_page is the page number of the passage that supports the answer.
  It must be a page you retrieved.
- Match the requested difficulty.
- If the material does not support the requested number of questions,
  submit fewer rather than padding.
"""

QUIZ_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Read the document in order, one section at a time. Returns "
                "text with page markers and a next_offset for the following "
                "section."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "offset": {
                        "type": "integer",
                        "description": (
                            "Where to start reading. Use 0 first, then the "
                            "next_offset from the previous read."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": (
                "Semantic search within this document only. Returns the most "
                "relevant passages with page numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, in natural language.",
                    },
                    "match_count": {
                        "type": "integer",
                        "description": "How many passages. Default 6, max 15.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_quiz",
            "description": (
                "Submit the finished quiz. It is validated; if rejected, fix "
                "the problems and submit again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "correct_index": {"type": "integer"},
                                "explanation": {"type": "string"},
                                "source_page": {"type": "integer"},
                            },
                            "required": [
                                "question",
                                "options",
                                "correct_index",
                                "explanation",
                                "source_page",
                            ],
                        },
                    },
                },
                "required": ["questions"],
            },
        },
    },
]


def _validate_questions(raw_questions, limit, seen_pages):
    clean = []
    problems = []

    if isinstance(raw_questions, str):
        try:
            raw_questions = json.loads(raw_questions)
        except json.JSONDecodeError:
            return clean, ["'questions' was not valid JSON."]

    if not isinstance(raw_questions, list):
        return clean, ["'questions' must be a list."]

    for number, item in enumerate(raw_questions, start=1):
        label = f"Question {number}"

        if not isinstance(item, dict):
            problems.append(f"{label}: must be an object.")
            continue

        question = item.get("question")
        options = item.get("options")

        if not question:
            problems.append(f"{label}: missing question text.")
            continue
        if not isinstance(options, list) or len(options) != 4:
            problems.append(f"{label}: must have exactly 4 options.")
            continue

        try:
            index = int(item.get("correct_index"))
        except (TypeError, ValueError):
            problems.append(f"{label}: correct_index must be an integer.")
            continue
        if not 0 <= index < len(options):
            problems.append(f"{label}: correct_index must be 0-3.")
            continue

        try:
            page = int(item.get("source_page"))
        except (TypeError, ValueError):
            problems.append(f"{label}: source_page must be an integer.")
            continue
        if page not in seen_pages:
            problems.append(
                f"{label}: source_page {page} is not a page you retrieved. "
                "Only cite pages returned by read_document or search_document."
            )
            continue

        clean.append(
            {
                "question": str(question),
                "options": [str(option) for option in options],
                "correct_index": index,
                "explanation": str(item.get("explanation", "")),
                "source_page": page,
            }
        )

    return clean[:limit], problems


class QuizAgent:

    def __init__(self, supabase, groq_client, embed_model, user_id):
        self.supabase = supabase
        self.groq = groq_client
        self.embed_model = embed_model
        self.user_id = user_id
        self._reset(None, 0)

    def _reset(self, document, num_questions):
        self._document = document
        self._num_questions = num_questions
        self._chunks = []
        self._seen_pages = set()
        self._result = None
        self._best_partial = []

    def _load_chunks(self):
        response = (
            self.supabase.table("document_chunks")
            .select("content, pagenumber")
            .eq("document_id", self._document["id"])
            .eq("userId", self.user_id)
            .execute()
        )
        rows = response.data or []
        rows.sort(key=lambda row: row.get("pagenumber") or 0)
        return rows

    def _remember_page(self, page):
        try:
            self._seen_pages.add(int(page))
        except (TypeError, ValueError):
            pass

    def tool_read_document(self, offset=0):
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0

        total_chunks = len(self._chunks)
        if offset >= total_chunks:
            return "End of document. There is no more text."

        parts = []
        size = 0
        index = offset
        while index < total_chunks:
            row = self._chunks[index]
            piece = f"[Page {row['pagenumber']}]\n{row['content']}\n"
            if parts and size + len(piece) > READ_CHUNK_CHARS:
                break
            parts.append(piece)
            size += len(piece)
            self._remember_page(row.get("pagenumber"))
            index += 1

        if index < total_chunks:
            footer = f"\n(next_offset: {index} of {total_chunks} chunks)"
        else:
            footer = "\n(End of document.)"
        return "\n".join(parts) + footer

    def tool_search_document(self, query, match_count=6):
        try:
            match_count = int(match_count)
        except (TypeError, ValueError):
            match_count = 6
        match_count = max(1, min(match_count, 15))

        query_embedding = self.embed_model.encode([query])[0]
        results = self.supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding.tolist(),
                "match_user_id": self.user_id,
                "match_count": MAX_SEARCH_CANDIDATES,
            },
        ).execute()

        filename = self._document["original_filename"]
        rows = [
            row
            for row in results.data or []
            if row.get("Filename") == filename
            and row.get("similarity", 0) >= 0.3
        ][:match_count]

        if not rows:
            return (
                "No relevant passages in this document. Try different "
                "wording, or use read_document."
            )

        parts = []
        for row in rows:
            self._remember_page(row.get("pagenumber"))
            parts.append(
                f"[Page {row['pagenumber']}]"
                f" (similarity {row['similarity']:.2f})\n"
                f"{row['content']}"
            )
        return "\n\n".join(parts)

    def tool_submit_quiz(self, questions):
        clean, problems = _validate_questions(
            questions,
            self._num_questions,
            self._seen_pages,
        )

        if len(clean) > len(self._best_partial):
            self._best_partial = clean

        if problems:
            return (
                "Quiz rejected. Fix these problems and call submit_quiz again "
                "with the full corrected list:\n- " + "\n- ".join(problems)
            )
        if not clean:
            return "Quiz rejected: it contained no questions."

        self._result = clean
        return f"Quiz accepted with {len(clean)} questions."

    def _call_tool(self, name, arguments):
        try:
            if name == "read_document":
                return self.tool_read_document(arguments.get("offset", 0))
            if name == "search_document":
                return self.tool_search_document(
                    arguments.get("query", ""),
                    arguments.get("match_count", 6),
                )
            if name == "submit_quiz":
                return self.tool_submit_quiz(arguments.get("questions", []))
            return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool '{name}' failed: {e}"
    def _finish(self, topic, difficulty, questions):
        quiz = {
            "filename": self._document["original_filename"],
            "topic": topic,
            "difficulty": difficulty,
            "questions": questions,
        }
        return quiz, f"Generated {len(questions)} questions."

    def run(
        self,
        document,
        num_questions=5,
        topic=None,
        difficulty="medium",
        on_step=None,
    ):
        """
        Returns (quiz_dict, message). quiz_dict is None on failure and
        message explains why.
        """
        try:
            num_questions = int(num_questions)
        except (TypeError, ValueError):
            num_questions = 5
        num_questions = max(1, min(num_questions, MAX_QUIZ_QUESTIONS))

        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"

        self._reset(document, num_questions)
        self._chunks = self._load_chunks()
        if not self._chunks:
            return None, "That document has no extracted text to build a quiz from."

        task = (
            f"Document: {document['original_filename']} "
            f"({len(self._chunks)} chunks)\n"
            f"Number of questions: {num_questions}\n"
            f"Difficulty: {difficulty}\n"
            + (
                f"Topic: {topic}"
                if topic
                else "Topic: none - cover the whole document."
            )
        )

        messages = [
            {"role": "system", "content": QUIZ_AGENT_PROMPT},
            {"role": "user", "content": task},
        ]
        nudged = False

        for _ in range(MAX_QUIZ_STEPS):
            response = self.groq.chat.completions.create(
                model=QUIZ_MODEL,
                messages=messages,
                tools=QUIZ_AGENT_TOOLS,
                tool_choice="auto",
                reasoning_format="hidden",
                temperature=0.4,
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                if nudged:
                    break
                nudged = True
                messages.append(
                    {"role": "assistant", "content": message.content or ""}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Deliver the quiz by calling submit_quiz.",
                    }
                )
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            for call in tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                if on_step:
                    on_step(call.function.name, arguments)

                result = self._call_tool(call.function.name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

            if self._result is not None:
                return self._finish(topic, difficulty, self._result)

        if self._best_partial:
            return self._finish(topic, difficulty, self._best_partial)

        return None, "The quiz agent could not produce a valid quiz from that document."
