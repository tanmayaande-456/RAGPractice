import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)) 
load_dotenv(HERE.parent / ".env")

REFUSAL = "I could not find that information in the documents."


def check(answer, keywords):
    text = answer.lower()
    if not keywords:
        return [] if REFUSAL.lower() in text else [f'refusal ("{REFUSAL}")']
    return [k for k in keywords if k.lower() not in text]


def make_bot():
    from groq import Groq
    from sentence_transformers import SentenceTransformer
    from supabase import create_client
    from summarize_agent import build_agent
    # EMAIL= "ande.tanmaya2@gmail.com"
    # PASSWORD= "123abc"
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    user = supabase.auth.sign_in_with_password({
        "email": os.environ["TEST_USER_EMAIL"],
        "password": os.environ["TEST_USER_PASSWORD"],
        # "email": EMAIL,
        # "password": PASSWORD
    }).user
    agent = build_agent(supabase, Groq(api_key=os.environ["GROQ_API_KEY"]), SentenceTransformer("all-MiniLM-L6-v2"), user.id)
    return lambda question: agent.run(question, history=[], on_step=lambda name, args: print(f"    -> {name} {args}"))


def main():
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))["Questions"]
    use_reference = "--reference" in sys.argv
    ask = None if use_reference else make_bot()

    passed = 0
    for q in questions:
        if use_reference:
            answer = q["Answer"]
        else:
            answer = ask(q["Question"])
            time.sleep(2)
        missing = check(answer, q["Must_include"])

        if missing:
            print(f"FAIL {q['Number']:>2}. {q['Question']}")
            print(f"       missing: {missing}")
            print(f"       got:     {' '.join(answer.split())}")
        else:
            passed += 1
            print(f"PASS {q['Number']:>2}. {q['Question']}")

    print(f"\n{passed}/{len(questions)} passed")


if __name__ == "__main__":
    main()