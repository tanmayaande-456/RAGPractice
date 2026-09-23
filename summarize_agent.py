import json

AGENT_MODEL = "openai/gpt-oss-120b"

MAX_BATCH_CHARS = 12000
MAX_FINAL_CHARS = 12000
MAX_SUMMARY_RETURN_CHARS = 8000

SYSTEM_PROMPT = """You are a retrieval assistant for the user's uploaded documents.

You have tools. Use them instead of guessing:
- list_documents: when you need to know which documents exist, or the user
  refers to a document vaguely ("the PDF", "my notes").
- search_documents: for specific questions about content. This is your default.
- summarize_document: ONLY when the user wants an overview of a whole document.
  It is slow and expensive, so never call it to answer a specific question.

Rules:
- Answer only from tool results. Never use outside knowledge.
- If the tools return nothing relevant, say:
  "I could not find that information in the documents."
- Mention page numbers and filenames when relevant.
- Keep answers concise.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List the filenames of every document the user has uploaded, "
                "with their sizes. Takes no arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Semantic search over the user's documents. Returns the most "
                "relevant chunks with filename and page number. Use this to "
                "answer specific questions."
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
                        "description": "How many chunks to retrieve. Default 5, max 15.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": (
                "Produce a full summary of one entire document. Expensive; use "
                "only when the user asks for an overview or summary of a whole "
                "document, not for specific questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Filename of the document, as returned by "
                            "list_documents."
                        ),
                    },
                },
                "required": ["filename"],
            },
        },
    },
]


class DocumentAgent:
    def __init__(self, supabase, groq_client, embed_model, user_id):
        self.supabase = supabase
        self.groq = groq_client
        self.embed_model = embed_model
        self.user_id = user_id

    def _get_documents(self):
        response = (
            self.supabase.table("documents")
            .select("*")
            .eq("userId", self.user_id)
            .order("uploaded_at", desc=True)
            .execute()
        )
        return response.data or []

    def _resolve_document(self, filename):
        documents = self._get_documents()
        if not documents:
            return None

        wanted = (filename or "").strip().lower()

        for document in documents:
            if document["original_filename"].lower() == wanted:
                return document

        matches = [
            document
            for document in documents
            if wanted and wanted in document["original_filename"].lower()
        ]
        if len(matches) == 1:
            return matches[0]

        return None

    def tool_list_documents(self):
        documents = self._get_documents()
        if not documents:
            return "No documents have been uploaded."

        lines = []
        for document in documents:
            size_kb = document.get("filesize", 0) / 1024
            lines.append(
                f"- {document['original_filename']} ({size_kb:.1f} KB)"
            )
        return "Uploaded documents:\n" + "\n".join(lines)

    def tool_search_documents(self, query, match_count=5):
        try:
            match_count = int(match_count)
        except (TypeError, ValueError):
            match_count = 5
        match_count = max(1, min(match_count, 15))

        query_embedding = self.embed_model.encode([query])[0]

        results = self.supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding.tolist(),
                "match_user_id": self.user_id,
                "match_count": match_count,
            },
        ).execute()

        chunks = results.data or []

        if not chunks or chunks[0]["similarity"] < 0.3:
            return "No relevant passages found."

        parts = []
        for row in chunks:
            parts.append(
                f"[{row['Filename']} — page {row['pagenumber']}]"
                f" (similarity {row['similarity']:.2f})\n"
                f"{row['content']}"
            )
        return "\n\n".join(parts)

    def tool_summarize_document(self, filename):
        document = self._resolve_document(filename)
        if document is None:
            available = self.tool_list_documents()
            return f"Could not find a document called '{filename}'.\n{available}"

        response = (
            self.supabase.table("document_chunks")
            .select("content, Filename, pagenumber")
            .eq("document_id", document["id"])
            .eq("userId", self.user_id)
            .execute()
        )

        chunk_data = response.data or []
        if not chunk_data:
            return "That document has no extracted text."

        # stable sort keeps insertion order within a page
        chunk_data.sort(key=lambda row: row.get("pagenumber") or 0)

        chunks = [
            f"[Page {row['pagenumber']}]\n{row['content']}\n"
            for row in chunk_data
        ]

        batches = []
        current_batch = ""
        for chunk in chunks:
            if len(current_batch) + len(chunk) > MAX_BATCH_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = ""
            current_batch += chunk + "\n"
        if current_batch:
            batches.append(current_batch)

        partial_summaries = []
        for batch in batches:
            response = self.groq.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize ONLY the provided document text. "
                            "Include main topics, important concepts, "
                            "definitions, formulas or facts, and examples when "
                            "present. Do not add outside knowledge. Do not "
                            "invent information. Keep the summary concise."
                        ),
                    },
                    {"role": "user", "content": batch},
                ],
                reasoning_format="hidden",
            )
            partial_summaries.append(response.choices[0].message.content or "")

        if len(partial_summaries) == 1:
            final_summary = partial_summaries[0]
        else:
            combined = "\n\n".join(partial_summaries)[:MAX_FINAL_CHARS]
            response = self.groq.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Create one organized summary from the provided "
                            "partial summaries. Use ONLY those summaries. Do "
                            "not add outside knowledge. Remove repetition. "
                            "Keep it concise and organized."
                        ),
                    },
                    {"role": "user", "content": combined},
                ],
                reasoning_format="hidden",
            )
            final_summary = response.choices[0].message.content or ""

        final_summary = final_summary.strip()
        if not final_summary:
            return "The summarizer returned nothing for that document."

        header = f"Summary of {document['original_filename']}:\n\n"
        return header + final_summary[:MAX_SUMMARY_RETURN_CHARS]

    def _call_tool(self, name, arguments):
        try:
            if name == "list_documents":
                return self.tool_list_documents()
            if name == "search_documents":
                return self.tool_search_documents(
                    arguments.get("query", ""),
                    arguments.get("match_count", 5),
                )
            if name == "summarize_document":
                return self.tool_summarize_document(
                    arguments.get("filename", "")
                )
            return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool '{name}' failed: {e}"

    def run(self, user_query, history=None, max_steps=6, on_step=None):
        """
        history: list of {"role": "user"|"assistant", "content": str} from the
                 current chat (tool turns are not persisted).
        on_step: optional callback(tool_name, arguments) for UI feedback.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for message in history or []:
            if message.get("role") in ("user", "assistant"):
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                )

        messages.append({"role": "user", "content": user_query})

        for _ in range(max_steps):
            response = self.groq.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                reasoning_format="hidden",
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                return (message.content or "").strip() or (
                    "I could not find that information in the documents."
                )

            assistant_turn = {
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
            messages.append(assistant_turn)

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

        return (
            "I ran out of steps before finishing. Try asking something more specific."
        )


def build_agent(supabase, groq_client, embed_model, user_id):
    return DocumentAgent(supabase, groq_client, embed_model, user_id)
