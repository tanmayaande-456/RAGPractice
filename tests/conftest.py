import os
import uuid

import pytest
from supabase import create_client
from utils import loadpdf

REQUIRED = ["SUPABASE_URL", "SUPABASE_KEY", "TEST_USER_EMAIL", "TEST_USER_PASSWORD"]


@pytest.fixture(scope="session")
def client_and_user():
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        pytest.skip(f"Supabase test credentials not set: {', '.join(missing)}")

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    auth = client.auth.sign_in_with_password({
        "email": os.environ["TEST_USER_EMAIL"],
        "password": os.environ["TEST_USER_PASSWORD"],
    })
    yield client, auth.user.id
    client.auth.sign_out()


@pytest.fixture(scope="session")
def seeded(client_and_user):
    """Insert a known chat, messages and documents for the test user, then remove them."""
    client, user_id = client_and_user
    run = uuid.uuid4().hex[:8]  # keeps parallel CI runs from colliding

    chat = client.table("chats").insert({
        "userId": user_id,
        "title": f"CI test chat {run}",
    }).execute().data[0]

    messages = [
        {"chat_id": chat["id"], "role": "user", "content": "What is dynamic programming?",
         "created_at": "2026-01-01T10:00:00+00:00"},
        {"chat_id": chat["id"], "role": "assistant", "content": "It breaks problems into overlapping subproblems.",
         "created_at": "2026-01-01T10:01:00+00:00"},
        {"chat_id": chat["id"], "role": "quiz", "content": '{"questions": []}',
         "created_at": "2026-01-01T10:02:00+00:00"},
    ]
    client.table("messages").insert(messages).execute()

    documents = [
        {"id": f"{user_id}_ci_{run}_old", "userId": user_id, "original_filename": f"ci_{run}_notes.pdf",
         "storage_path": f"{user_id}/ci_{run}_notes.pdf", "storage_filename": f"ci_{run}_old",
         "mime_type": "application/pdf", "filesize": 1024, "uploaded_at": "2026-01-01T00:00:00+00:00"},
        {"id": f"{user_id}_ci_{run}_new", "userId": user_id, "original_filename": f"ci_{run}_lecture.txt",
         "storage_path": f"{user_id}/ci_{run}_lecture.txt", "storage_filename": f"ci_{run}_new",
         "mime_type": "text/plain", "filesize": 2048, "uploaded_at": "2026-01-02T00:00:00+00:00"},
    ]
    client.table("documents").insert(documents).execute()

    yield {
        "client": client,
        "user_id": user_id,
        "chat": chat,
        "messages": messages,
        "documents": documents,
    }

    # Cleanup runs even if tests fail
    client.table("messages").delete().eq("chat_id", chat["id"]).execute()
    client.table("chats").delete().eq("id", chat["id"]).execute()
    for doc in documents:
        client.table("documents").delete().eq("id", doc["id"]).eq("userId", user_id).execute()