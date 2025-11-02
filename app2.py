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
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "2")

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SYSTEM_POLICY = (
    "You are LoopCare, a specialized medical prescription assistant. "
    "You help users manage prescriptions, medication schedules, and health tracking. "
    "Always stay polite, helpful, and concise. "
    "Never diagnose — always recommend consulting a doctor for medical advice. "
    "Focus on prescription information, medication instructions, and health monitoring."
)

# --- 2. GROQ LLM WRAPPER ---
class GroqLLM:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GROQ_API_KEY") or "USE YOUR OWN API"
        if not api_key:
            st.error("GROQ_API_KEY not provided.")
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
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
            )
            choice = getattr(chat_completion, "choices", None)
            if choice and len(choice) > 0:
                content = choice[0].message.content
                return content if isinstance(content, str) else str(content)
            return str(chat_completion)
        except Exception as e:
            err = f"[Groq error: {e}]"
            print(err)
            return "I'm experiencing temporary issues. Please try again in a moment."

# --- 3. MODEL LOADING ---
@st.cache_resource(show_spinner=False)
def get_quantization_config():
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
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def load_vlm_model():
    hf_token = os.getenv("HF_TOKEN", "USE YOUR OWN API")
    if hf_token:
        try:
            login(token=hf_token)
        except Exception as e:
            print("HuggingFace login failed:", e)

    vlm_model_id = "google/paligemma-3b-mix-224"
    try:
        vlm_processor = PaliGemmaProcessor.from_pretrained(vlm_model_id)
        vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(
            vlm_model_id,
            quantization_config=get_quantization_config(),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        return vlm_processor, vlm_model
    except Exception as e:
        print("PaliGemma load failed:", e)
        return None, None

# --- APP INITIALIZATION ---
st.set_page_config(page_title="LoopCare - Prescription Logger", layout="wide")
st.title("🩺 Medical Prescription Logger & Assistant")
st.caption("Upload prescription images to translate & log. Ask questions about your medication records.")

# Initialize session state
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.text_db = []
    st.session_state.doc_ids = []
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! I'm your medical prescription assistant. Upload a prescription image to log it, or ask me questions about your medications."
    }]

# Load models
with st.spinner("Loading AI models..."):
    llm = load_groq_llm()
    if st.session_state.retriever is None:
        st.session_state.retriever = load_retriever_model()
    vlm_processor, vlm_model = load_vlm_model()

    # Initialize FAISS index after retriever is loaded
    if st.session_state.faiss_index is None and st.session_state.retriever is not None:
        d = st.session_state.retriever.get_sentence_embedding_dimension()
        st.session_state.faiss_index = faiss.IndexFlatL2(d)

# Helper function to add text to FAISS
def add_text_to_faiss(text: str):
    if not text or text.strip() == "":
        return False
    
    try:
        vec = st.session_state.retriever.encode([text])
        vec = np.asarray(vec).astype("float32")
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        st.session_state.faiss_index.add(vec)
        st.session_state.text_db.append(text)
        st.session_state.doc_ids.append(str(uuid.uuid4()))
        return True
    except Exception as e:
        print(f"Failed to add to FAISS: {e}")
        return False

# --- INTENT CLASSIFICATION ---
def classify_medical_intent(user_input: str) -> str:
    ui = user_input.lower()
    
    if any(w in ui for w in ["prescription", "medication", "medicine", "drug", "pill", "dose", "dosage", "logged", "record"]):
        return "prescription_query"
    if any(w in ui for w in ["side effect", "interaction", "take with", "empty stomach", "safe", "danger"]):
        return "medication_safety"
    if any(w in ui for w in ["schedule", "when to take", "how often", "times a day", "frequency"]):
        return "medication_schedule"
    if any(w in ui for w in ["what is", "what does", "explain", "describe", "information about"]):
        return "medication_info"
    return "general_medical"

# --- MEDICAL PIPELINES ---
def run_medical_rag_pipeline(query: str, k=3) -> str:
    if st.session_state.faiss_index is None or st.session_state.faiss_index.ntotal == 0:
        return "I haven't logged any prescriptions yet. Please upload a prescription image first."

    try:
        qvec = st.session_state.retriever.encode([query]).astype("float32")
        if qvec.ndim == 1:
            qvec = qvec.reshape(1, -1)

        k_results = min(k, st.session_state.faiss_index.ntotal)
        distances, indices = st.session_state.faiss_index.search(qvec, k_results)

        docs = []
        for idx in indices[0]:
            if 0 <= idx < len(st.session_state.text_db):
                docs.append(st.session_state.text_db[idx])

        if not docs:
            return "I searched my prescription records, but couldn't find anything matching your query."

        context_string = "\n---\n".join(docs)
        rag_prompt = f"""
        Based ONLY on these prescription records:
        {context_string}
        
        Answer: "{query}"
        
        If the information isn't in the records, say so clearly.
        """
        
        if llm:
            return llm.generate(rag_prompt)
        else:
            return f"Based on your records:\n\n{context_string}"
    
    except Exception as e:
        print(f"RAG error: {e}")
        return "Error searching records. Please try again."

def run_prescription_transcription(image_bytes: bytes) -> str:
    if vlm_processor is None or vlm_model is None:
        return "Medical transcription model not available. Please try again later."

    medical_prompt = """Transcribe all text from this medical prescription clearly and accurately.
Perform these medical abbreviations replacements:
- 'BID' or 'bid' -> 'twice a day'
- 'PO' or 'po' -> 'by mouth'
- 'TID' or 'tid' -> 'three times a day'
- 'QID' or 'qid' -> 'four times a day'
- 'PRN' or 'prn' -> 'as needed'
- 'OD' -> 'once daily'
- 'AC' -> 'before meals'
- 'PC' -> 'after meals'
- 'HS' -> 'at bedtime'

Extract: patient name, medication names, dosages, frequencies, and instructions.
"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
    except Exception as e:
        return f"Failed to process image: {e}"

    try:
        inputs = vlm_processor(text=medical_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(vlm_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            response = vlm_model.generate(**inputs, max_new_tokens=500)
        
        transcribed_text = vlm_processor.decode(response[0], skip_special_tokens=True)
        
        if medical_prompt.strip() in transcribed_text:
            transcribed_text = transcribed_text.split(medical_prompt.strip(), 1)[-1].strip()
        
        transcribed_text = transcribed_text.strip()
        
        if not transcribed_text:
            transcribed_text = "No text could be transcribed from this image"
            
    except Exception as e:
        print("Transcription error:", e)
        transcribed_text = f"Transcription failed: {e}"

    # Add to FAISS if transcription successful
    if transcribed_text and not transcribed_text.startswith("Transcription failed") and not transcribed_text.startswith("Failed to process"):
        success = add_text_to_faiss(transcribed_text)
        if not success:
            transcribed_text += "\n\n[Note: Could not save to database]"

    return transcribed_text

# --- SIDEBAR ---
st.sidebar.title("Medical Database")
if st.session_state.faiss_index is not None:
    st.sidebar.metric("Prescriptions Logged", st.session_state.faiss_index.ntotal)
else:
    st.sidebar.metric("Prescriptions Logged", 0)

# Model status
if vlm_processor is None or vlm_model is None:
    st.sidebar.error("❌ Transcription Model")
else:
    st.sidebar.success("✅ Transcription Model")

if llm is None:
    st.sidebar.warning("⚠️ Assistant Model")
else:
    st.sidebar.success("✅ Assistant Model")

# File upload in sidebar
with st.sidebar:
    st.subheader("Upload Prescription")
    uploaded_file = st.file_uploader("Choose prescription image", type=["png", "jpg", "jpeg"], key="file_uploader")

# --- CHAT DISPLAY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            try:
                st.image(message["image"], width=250, caption="Uploaded Prescription")
            except Exception:
                st.write("📷 Prescription Image")
        st.markdown(message["content"])

# --- HANDLE FILE UPLOAD ---
if uploaded_file is not None:
    # Check if this is a new file
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if "processed_file" not in st.session_state or st.session_state.processed_file != current_file_id:
        st.session_state.processed_file = current_file_id
        
        image_bytes = uploaded_file.getvalue()
        
        # Add user message with image
        st.session_state.messages.append({
            "role": "user",
            "image": image_bytes,
            "content": f"Uploaded prescription: {uploaded_file.name}"
        })
        
        # Display user message
        with st.chat_message("user"):
            st.image(image_bytes, width=250, caption=uploaded_file.name)
            st.markdown(f"Uploaded prescription: {uploaded_file.name}")
        
        # Process transcription
        with st.chat_message("assistant"):
            with st.spinner("Transcribing prescription..."):
                transcription = run_prescription_transcription(image_bytes)
                
                # Display transcription
                st.subheader("📋 Transcribed Prescription")
                st.text_area("Transcription", transcription, height=200, key=f"transcript_{uploaded_file.name}")
                
                # Generate confirmation
                if transcription and not transcription.startswith("Transcription failed") and not transcription.startswith("Failed to process"):
                    if llm:
                        confirm_prompt = f"Tell the user their prescription has been logged and summarize this information briefly: {transcription[:300]}"
                        llm_response = llm.generate(confirm_prompt)
                        st.markdown(llm_response)
                        st.session_state.messages.append({"role": "assistant", "content": llm_response})
                    else:
                        success_msg = "✅ Prescription logged successfully! I've added it to your medical records."
                        st.success(success_msg)
                        st.session_state.messages.append({"role": "assistant", "content": success_msg})
                else:
                    error_msg = f"❌ Failed to process prescription: {transcription}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()

# --- HANDLE TEXT CHAT INPUT ---
if prompt := st.chat_input("Ask about your prescriptions or medications..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent = classify_medical_intent(prompt)
            st.sidebar.info(f"Intent: {intent}")
            
            if intent == "prescription_query" and st.session_state.faiss_index is not None and st.session_state.faiss_index.ntotal > 0:
                response = run_medical_rag_pipeline(prompt)
            elif intent in ["medication_safety", "medication_schedule", "medication_info"]:
                if llm:
                    medical_prompt = f"As a medical assistant, provide helpful information about: {prompt}. Remember to advise consulting a doctor for medical decisions."
                    response = llm.generate(medical_prompt)
                else:
                    response = "I can help with medication information. For detailed medical advice, please consult your healthcare provider."
            else:
                if llm:
                    response = llm.generate(prompt)
                else:
                    response = "I'm here to help with your prescription management. You can upload prescription images or ask about medications."
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- DEBUG VIEWER ---
with st.sidebar.expander("View Stored Prescriptions"):
    if st.session_state.text_db:
        st.write(f"Total prescriptions: {len(st.session_state.text_db)}")
        for i, text in enumerate(st.session_state.text_db):
            with st.expander(f"Prescription {i+1}"):
                st.text(text[:500] + "..." if len(text) > 500 else text)
    else:
        st.write("No prescriptions stored yet.")