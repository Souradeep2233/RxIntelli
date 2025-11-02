import streamlit as st
import chromadb
import uuid
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    BitsAndBytesConfig
)
import torch
from PIL import Image
import io
from huggingface_hub import login

# --- 1. MODEL & DATA LOADING (Cached for speed) ---

@st.cache_resource
def get_quantization_config():
    """Load models in 4-bit to save memory."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


@st.cache_resource
def load_models():
    """Load all AI models once."""
    print("Loading models... This will take several minutes.")

    # Authenticate Hugging Face login
    login(token="USE YOUR OWN API")

    # --- RAG: Generator (LLM) Model ---
    print("Loading Gemma 2B...")
    llm_model_id = "google/gemma-2b-it"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        quantization_config=get_quantization_config(),
        device_map="auto",
    )

    # --- Vision: PaliGemma Model ---
    print("Loading PaliGemma...")
    vlm_model_id = "google/paligemma-3b-mix-224"
    vlm_processor = PaliGemmaProcessor.from_pretrained(vlm_model_id)
    vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(
        vlm_model_id,
        quantization_config=get_quantization_config(),
        device_map="auto",
    )

    print("All models loaded.")
    return llm_tokenizer, llm_model, vlm_processor, vlm_model


st.title("🩺 Live Prescription Logger & Search (RAG + VLM)")
st.caption("Upload a prescription to translate AND log it. Then, search for it!")

with st.spinner("Loading AI models... This may take a few minutes."):
    llm_tokenizer, llm_model, vlm_processor, vlm_model = load_models()


@st.cache_resource
def load_chroma_collection():
    """Connect to or create a persistent vector database."""
    print("Connecting to persistent ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="prescriptions")
    print("Connected to ChromaDB.")
    return collection


collection = load_chroma_collection()

# --- 2. PIPELINE FUNCTIONS ---

def run_rag_pipeline(query, k=3):
    """RAG: Retrieve + Generate from ChromaDB."""
    if collection.count() == 0:
        return "I haven't logged any prescriptions yet. Please upload an image of a prescription first."

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count())
    )

    if not results['documents'][0]:
        return "I searched my records, but couldn't find anything matching your query."

    context_chunks = results['documents'][0]
    context_string = "\n---\n".join(context_chunks)

    prompt_template = f"""
    <start_of_turn>user
    You are a helpful medical assistant. Answer the user's question based *only* on the following prescription records.
    If the records don't contain the answer, say so.

    **My Prescription Records:**
    {context_string}

    **User's Question:**
    {query}<end_of_turn>
    <start_of_turn>model
    """

    inputs = llm_tokenizer(prompt_template, return_tensors="pt").to(llm_model.device)
    response = llm_model.generate(**inputs, max_new_tokens=250)
    generated_text = llm_tokenizer.decode(response[0], skip_special_tokens=True)
    model_response = generated_text.split("<start_of_turn>model")[-1].strip()
    return model_response


def run_vision_and_log_pipeline(image_bytes, original_filename):
    """Use PaliGemma to extract text from prescription and log it to ChromaDB."""

    # --- THIS IS THE NEW, BETTER PROMPT ---
    prompt = """Transcribe all text from this image.
Perform these specific text replacements:
- Replace 'BID' with 'twice a day'
- Replace 'PO' with 'by mouth'
- Replace 'TID' with 'three times a day'
- Replace 'QID' with 'four times a day'
- Replace 'PRN' with 'as needed'
- Replace 'QD' with 'once a day'
- Replace 'AC' with 'before meals'
- Replace 'PC' with 'after meals'
"""
    # --- END OF PROMPT CHANGE ---

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = vlm_processor(text=prompt, images=image, return_tensors="pt").to(vlm_model.device)
    response = vlm_model.generate(**inputs, max_new_tokens=200)
    translated_text = vlm_processor.decode(response[0], skip_special_tokens=True)
    
    # Clean the output (it might repeat the prompt)
    translated_text = translated_text.split(prompt)[-1].strip()

    # --- LOGGING (no changes needed) ---
    doc_id = f"rx_{str(uuid.uuid4())}"
    collection.add(
        documents=[translated_text],
        metadatas=[{"source": "Image Upload", "filename": original_filename}],
        ids=[doc_id]
    )

    print(f"Successfully logged new document: {doc_id}")
    return translated_text

# --- 3. STREAMLIT CHAT UI ---

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! My database is ready. Upload a prescription image to log it, or ask me a question about your logged prescriptions."
    }]

# Sidebar info
st.sidebar.title("Database Status")
db_count = collection.count()
st.sidebar.metric("Prescriptions Logged", db_count)
if db_count > 0:
    st.sidebar.success("Database is online and contains data.")
else:
    st.sidebar.warning("Database is empty. Upload an image to start.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"], width=200)
        st.markdown(message["content"])

# --- INPUT SECTION (Fixed) ---

col1, col2 = st.columns([3, 1])

with col1:
    user_text = st.chat_input("Ask a question...")

with col2:
    uploaded_file = st.file_uploader("Upload a prescription", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# --- Handle image upload ---
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    st.session_state.messages.append({
        "role": "user",
        "image": image_bytes,
        "content": f"Please translate and log this: `{uploaded_file.name}`"
    })

    with st.chat_message("user"):
        st.image(image_bytes, width=200)
        st.markdown(f"Please translate and log this: `{uploaded_file.name}`")

    with st.chat_message("assistant"):
        with st.spinner("PaliGemma is translating and logging..."):
            translated_text = run_vision_and_log_pipeline(image_bytes, uploaded_file.name)
            response = f"**Translation complete and logged!**\n\n```\n{translated_text}\n```"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            # st.rerun()


# --- Handle text queries ---
elif user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("RAG pipeline is searching your logs..."):
            response = run_rag_pipeline(user_text)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
