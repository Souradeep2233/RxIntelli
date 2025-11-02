import streamlit as st
import uuid
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig
)
import torch
from PIL import Image
import io
from huggingface_hub import login
from groq import Groq
from typing import Optional

# --- 1. ENVIRONMENT & API KEY SETUP ---
# Set which GPU visible (modify if needed)
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "2")

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SYSTEM_POLICY = (
    "You are LoopCare, a friendly healthcare assistant. "
    "You help users manage reports, appointments, insurance, health tracking, and wellness. "
    "Always stay polite, helpful, and concise. "
    "Never diagnose — escalate to a doctor if needed. "
    "Keep responses under 80 words unless summarizing reports."
)

# --- 2. GROQ LLM WRAPPER (resilient) ---
class GroqLLM:
    """Light wrapper for Groq chat-completion API with graceful failure handling."""
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        # Prefer api_key param, else env, else fallback string if provided earlier
        api_key = api_key or os.getenv("GROQ_API_KEY") or "USE YOUR OWN API"
        if not api_key:
            st.error("GROQ_API_KEY not provided. Set GROQ_API_KEY env var.")
            raise RuntimeError("GROQ_API_KEY not provided")
        try:
            self.client = Groq(api_key=api_key)
            self.model = model
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}")
            raise

    def generate(self, user_text: str, system_prompt: str = SYSTEM_POLICY) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                model=self.model,
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
            )
            # Defensive access
            choice = getattr(chat_completion, "choices", None)
            if choice and len(choice) > 0:
                content = choice[0].message.content
                return content if isinstance(content, str) else str(content)
            return str(chat_completion)
        except Exception as e:
            # Don't crash app — return a helpful fallback string
            err = f"[Groq error: {e}]"
            print(err)
            return "Sorry — the Groq LLM failed to respond right now. " \
                   "Try again in a moment."

# --- 3. MODEL LOADING (cached) ---
@st.cache_resource(show_spinner=False)
def get_quantization_config():
    # Keep but it's only used for huge models; if you don't need 4-bit, remove usage
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

@st.cache_resource(show_spinner=False)
def load_groq_llm():
    try:
        return GroqLLM()
    except Exception as e:
        print(f"Groq init error: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_retriever_model():
    # local sentence transformer used for FAISS vectors
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def load_vlm_model():
    # PaliGemma is large and gated — ensure token is valid
    try:
        login(token=os.getenv("HF_TOKEN", "USE YOUR OWN API"))
    except Exception as e:
        print("HuggingFace login failed or HF_TOKEN not set:", e)
        # continue; loader might still fail below and will be caught

    vlm_model_id = "google/paligemma-3b-mix-224"
    try:
        vlm_processor = PaliGemmaProcessor.from_pretrained(vlm_model_id)
        vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(
            vlm_model_id,
            quantization_config=get_quantization_config(),
            device_map="auto",
        )
        return vlm_processor, vlm_model
    except Exception as e:
        print("PaliGemma load failed:", e)
        return None, None

# --- App header & model loading UI ---
st.set_page_config(page_title="LoopCare - Prescription Logger", layout="wide")
st.title("🩺 Live Prescription Logger & Search (Groq + FAISS)")
st.caption("Upload a prescription image to translate & log. Ask questions about your logs.")

with st.spinner("Loading models (may take a while)..."):
    llm = load_groq_llm()
    retriever = load_retriever_model()
    vlm_processor, vlm_model = load_vlm_model()

# --- 4. FAISS SETUP (in session_state) ---
d = retriever.get_sentence_embedding_dimension()
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(d)
    st.session_state.text_db = []  # list of strings
    # for tracking ids (optional)
    st.session_state.doc_ids = []

# small helper to safe-add vector to FAISS
def add_text_to_faiss(text: str):
    vec = retriever.encode([text])
    vec = np.asarray(vec).astype("float32")
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    st.session_state.faiss_index.add(vec)
    st.session_state.text_db.append(text)
    st.session_state.doc_ids.append(str(uuid.uuid4()))

# --- 5. INTENT CLASSIFIER (cached) ---
@st.cache_data(show_spinner=False)
def classify_intent(user_input: str, _llm: Optional[GroqLLM]) -> str:
    # If LLM not present, naive rules
    if _llm is None:
        ui = user_input.lower()
        if any(w in ui for w in ["prescription", "record", "log", "med", "medicine"]):
            return "records_management"
        if any(w in ui for w in ["appointment", "book", "schedule", "reschedule"]):
            return "appointment"
        if any(w in ui for w in ["insurance", "claim", "billing"]):
            return "insurance_billing"
        return "general_inquiry"

    prompt = f"""
    Classify the user's intent into one of these categories:
    [records_management, appointment, insurance_billing, general_inquiry]
    User Input: "{user_input}"
    Classification:
    """
    response = _llm.generate(prompt, system_prompt="You are a concise text classifier.")
    classification = (response or "").lower()
    if "records_management" in classification: return "records_management"
    if "appointment" in classification: return "appointment"
    if "insurance_billing" in classification: return "insurance_billing"
    return "general_inquiry"

# --- 6. PIPELINES ---
def run_rag_pipeline(query: str, k=3) -> str:
    # If FAISS empty
    if st.session_state.faiss_index.ntotal == 0:
        return "I haven't logged any prescriptions yet. Please upload an image first."

    # encode query and search
    qvec = retriever.encode([query]).astype("float32")
    if qvec.ndim == 1:
        qvec = qvec.reshape(1, -1)

    k_results = min(k, st.session_state.faiss_index.ntotal)
    distances, indices = st.session_state.faiss_index.search(qvec, k_results)

    # indices may contain -1; guard
    docs = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(st.session_state.text_db):
            continue
        docs.append(st.session_state.text_db[idx])

    if not docs:
        return "I searched my records, but couldn't find anything matching your query."

    context_string = "\n---\n".join(docs)
    rag_prompt = f"""
    Based ONLY on the following logged prescription records (do not invent details):
    ---
    {context_string}
    ---
    Answer the user's question: "{query}"
    """

    # Use Groq if available else a simple concatenation reply
    if llm:
        return llm.generate(rag_prompt)
    else:
        # fallback: show retrieved context
        return "Retrieved records:\n\n" + context_string

def run_vision_and_log_pipeline(image_bytes: bytes) -> str:
    """Transcribe with PaliGemma and log to FAISS. Return transcription."""
    if vlm_processor is None or vlm_model is None:
        return "[VLM model not loaded — cannot transcribe image.]"

    prompt = """Transcribe all text from this image.
Perform these replacements:
- 'BID' -> 'twice a day'
- 'PO' -> 'by mouth'
- 'TID' -> 'three times a day'
- 'QID' -> 'four times a day'
- 'PRN' -> 'as needed'
"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return f"[Failed to open image: {e}]"

    try:
        inputs = vlm_processor(text=prompt, images=image, return_tensors="pt").to(vlm_model.device)
        response = vlm_model.generate(**inputs, max_new_tokens=400)
        # decode result
        translated_text = vlm_processor.decode(response[0], skip_special_tokens=True)
        # model output may include the prompt; strip conservatively
        if prompt.strip() in translated_text:
            translated_text = translated_text.split(prompt.strip(), 1)[-1].strip()
        translated_text = translated_text.strip()
        if not translated_text:
            translated_text = "[Transcription was empty]"
    except Exception as e:
        print("VLM generation error:", e)
        translated_text = f"[VLM generation failed: {e}]"

    # Add to FAISS/text_db safely
    try:
        add_text_to_faiss(translated_text)
    except Exception as e:
        print("Failed to add to FAISS:", e)

    return translated_text

# --- 7. CHAT UI (kept layout as you requested) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! My database is ready. Upload a prescription image to log it, or ask me a question about your logged prescriptions."
    }]

# Sidebar status & helpful notes
st.sidebar.title("Database Status")
st.sidebar.metric("Prescriptions Logged", st.session_state.faiss_index.ntotal if "faiss_index" in st.session_state else 0)
if vlm_processor is None or vlm_model is None:
    st.sidebar.warning("PaliGemma VLM not loaded. Image transcription will not work.")
if llm is None:
    st.sidebar.info("Groq LLM not available — using local fallbacks for some tasks.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            try:
                st.image(message["image"], width=200)
            except Exception:
                pass
        st.markdown(message["content"])

# --- Unified Input Section (UI preserved) ---
col1, col2 = st.columns([2, 1])
with col1:
    user_text = st.chat_input("Ask a question or type a request...")  # text-only
with col2:
    uploaded_file = st.file_uploader("📸 Upload prescription", type=["png", "jpg", "jpeg"])

# --- Handle File Upload first ---
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    # append user message with image
    st.session_state.messages.append({
        "role": "user",
        "image": image_bytes,
        "content": f"Please translate and log this: `{uploaded_file.name}`"
    })
    # show user's uploaded message immediately
    with st.chat_message("user"):
        st.image(image_bytes, width=200)
        st.markdown(f"Please translate and log this: `{uploaded_file.name}`")

    # assistant processes transcription and logs to FAISS
    with st.chat_message("assistant"):
        with st.spinner("PaliGemma translating and logging..."):
            transcription = run_vision_and_log_pipeline(image_bytes)
            # show the transcription and a human confirmation (LLM if available)
            human_confirmation = f"**Translation complete and logged:**\n\n```\n{transcription}\n```"
            st.markdown(human_confirmation)
            # ask LLM to craft a friendly confirmation if available
            if llm:
                try:
                    confirm_prompt = f"Confirm to the user that this new prescription has been successfully logged: '{transcription}'"
                    llm_resp = llm.generate(confirm_prompt)
                    st.markdown(llm_resp)
                    st.session_state.messages.append({"role": "assistant", "content": llm_resp})
                except Exception as e:
                    print("Groq confirm error:", e)
                    st.session_state.messages.append({"role": "assistant", "content": human_confirmation})
            else:
                st.session_state.messages.append({"role": "assistant", "content": human_confirmation})

    # No explicit rerun required: UI updates on interaction

# --- Else handle text chat input ---
elif user_text:
    # append user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Groq brain is thinking..."):
            intent = classify_intent(user_text, llm)
            st.sidebar.info(f"Detected Intent: `{intent}`")

            if intent == "records_management":
                response = run_rag_pipeline(user_text)
            elif intent == "appointment":
                # delegate to Groq or fallback
                prompt = f"You are a scheduling assistant. Help handle this appointment request: {user_text}"
                response = (llm.generate(prompt) if llm else "I can help schedule — please provide date/time.")
            elif intent == "insurance_billing":
                prompt = f"You are an insurance assistant. Clearly answer or guide: {user_text}"
                response = (llm.generate(prompt) if llm else "For insurance queries, please provide claim/policy details.")
            else:
                response = (llm.generate(user_text) if llm else "I can help — ask about records, appointments, or insurance.")

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
