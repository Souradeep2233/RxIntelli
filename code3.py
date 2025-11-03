import os, glob, re, json, faiss
from datetime import datetime
from typing import Optional
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

# ==============================
# 🔐 LLM Wrapper Configuration
# ==============================
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_POLICY = (
    "You are LoopCare, a friendly healthcare assistant. "
    "You help users manage reports, appointments, insurance, health tracking, and wellness. "
    "Always stay polite, helpful, and concise. "
    "Never diagnose — escalate to a doctor if needed. "
    "Keep responses under 80 words unless summarizing reports."
)

FALLBACK_REPLY = "Sorry, something went wrong. Please try again."


# -------------------------------
# 🧠 LLM Wrapper
# -------------------------------
class GroqLLM:
    """Light wrapper for Groq chat-completion API."""
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, user_text: str, system_prompt: str = SYSTEM_POLICY, max_tokens=120) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text or "(silence)"}
        ]
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            msg = rsp.choices[0].message
            text = (getattr(msg, "content", None) or "").strip()
            return text or FALLBACK_REPLY
        except Exception as e:
            print(f"[LLM] API error: {e}")
            return FALLBACK_REPLY


# ==============================
# 🩺 LOOPCARE CHATBOT SETUP
# ==============================

llm = GroqLLM(api_key=None)
conversation_log = []
last_intent = None


# -------------------------------
# FAISS Semantic Search Setup
# -------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
report_paths = []


def build_faiss_index():
    """Build FAISS index from all report transcriptions."""
    global index, report_paths
    transcriptions = sorted(glob.glob("transcription*.txt"))
    if not transcriptions:
        print("⚠️ No transcription files found.")
        return

    embeddings = []
    report_paths = []

    print("🔧 Building FAISS index...")
    for tfile in transcriptions:
        with open(tfile, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue  # skip empty files
        emb = embed_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
        report_paths.append(tfile)

    if not embeddings:
        print("⚠️ No valid text reports found.")
        return

    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(np.array(embeddings, dtype=np.float32))
    print(f"✅ FAISS index ready with {len(embeddings)} reports.")


def extract_report_number(path):
    """Extract number from filename like transcription82.txt."""
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def semantic_fetch_reports(user_query, n=3, top_k=15):
    """
    Use FAISS to retrieve semantically similar reports.
    Step 1: Sort all by semantic similarity (desc)
    Step 2: Take top_k
    Step 3: From those, sort by recency (desc)
    Step 4: Return latest N reports
    """
    global index, report_paths
    if index is None:
        build_faiss_index()
        if index is None:
            return []

    # Encode the query
    query_emb = embed_model.encode(user_query, convert_to_numpy=True, normalize_embeddings=True)

    # FAISS search
    D, I = index.search(np.array([query_emb], dtype=np.float32), min(top_k, len(report_paths)))

    # Collect (path, similarity_score)
    matches = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(report_paths):
            matches.append((report_paths[idx], float(dist)))

    # --- Step 1: Sort by semantic similarity (desc)
    matches = sorted(matches, key=lambda x: x[1], reverse=True)

    # --- Step 2: Take top_k
    top_semantic = matches[:top_k]

    # --- Step 3: Sort those top_k by recency (desc)
    top_semantic_sorted = sorted(top_semantic, key=lambda x: extract_report_number(x[0]), reverse=True)

    # --- Step 4: Take top N among those
    latest_matches = top_semantic_sorted[:n]

    # Replace txt with pdf if exists
    final_reports = []
    for txt, score in latest_matches:
        pdf = txt.replace("transcription", "report").replace(".txt", ".pdf")
        final_reports.append(pdf if os.path.exists(pdf) else txt)

    return final_reports

# -------------------------------
# Intent classification
# -------------------------------
def classify_intent(user_input):
    prompt = f"""
    You are a triage model for a healthcare assistant chatbot (LoopCare).
    Classify the user's message into EXACTLY ONE of these categories:
    [health_info, appointment, insurance_billing, tracking_progress,
    records_management, lifestyle_support, help_escalation, notifications_alerts, general]
    Input: "{user_input}"
    Output: category only.
    """
    response = llm.generate(prompt)
    return response.strip().lower()


# -------------------------------
# Extract how many reports to fetch
# -------------------------------
def extract_report_count(user_input, default_n=3):
    prompt = f"""
    From the following request, extract how many reports the user wants.
    If none mentioned, return just '{default_n}'.
    Example output: '1', '3', '5' etc.
    Query: "{user_input}"
    """
    response = llm.generate(prompt)
    match = re.search(r"\d+", response)
    return int(match.group()) if match else default_n


# -------------------------------
# Conversation logging
# -------------------------------
def log_conversation(user_input, intent, llm_output):
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_input": user_input,
        "intent": intent,
        "llm_output": llm_output
    }
    conversation_log.append(entry)
    with open("loop_chatlog.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ==============================
# 💬 MAIN CHAT LOOP
# ==============================
def main():
    global last_intent

    print("🩺 LoopCare Assistant (Groq + FAISS)\nType 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("👋 Goodbye! Stay healthy!")
            break
        if user_input.lower() == "show log":
            for c in conversation_log[-5:]:
                print(f"[{c['timestamp']}] ({c['intent']}) You: {c['user_input']} → Bot: {c['llm_output']}")
            continue

        # ---- Intent classification ----
        intent = classify_intent(user_input).strip().lower()

        # Context carry-over
        if intent == "general" and last_intent not in [None, "general"]:
            print(f"[Intent → (context carry-over: {last_intent})]")
            intent = last_intent
        else:
            print(f"[Intent → {intent}]")

        reply = ""

        # ---- Intent-specific handling ----
        if intent == "health_info" or intent == "records_management":
            n = extract_report_count(user_input)
            print(f"🔍 Searching FAISS for top {n} most recent semantic matches...")
            matched = semantic_fetch_reports(user_input, n)
            if matched:
                reply = f"📂 Found {len(matched)} report(s): " + ", ".join(matched)
            else:
                reply = "💬 No semantically similar reports found."

        elif intent == "appointment":
            prompt = f"You are a scheduling assistant. Help handle this appointment request: {user_input}"
            reply = "📅 " + llm.generate(prompt)

        elif intent == "insurance_billing":
            prompt = f"You are an insurance assistant. Clearly answer or guide: {user_input}"
            reply = "💰 " + llm.generate(prompt)

        elif intent == "tracking_progress":
            prompt = f"You are a health data analyst. Summarize or visualize this tracking request: {user_input}"
            reply = "📈 " + llm.generate(prompt)

        elif intent == "lifestyle_support":
            prompt = f"You are a health coach. Respond conversationally and encouragingly: {user_input}"
            reply = "🧘 " + llm.generate(prompt)

        elif intent == "help_escalation":
            reply = "🚨 This seems important. Escalating your request to a doctor or support team..."

        elif intent == "notifications_alerts":
            prompt = f"You are a reminder assistant. Respond appropriately: {user_input}"
            reply = "🔔 " + llm.generate(prompt)

        else:
            reply = "💬 " + llm.generate(user_input)

        print(reply)
        log_conversation(user_input, intent, reply)
        last_intent = intent


# ==============================
# 🚀 ENTRY POINT
# ==============================
if __name__ == "__main__":
    main()
