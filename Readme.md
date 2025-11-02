# **🩺 LoopCare**

### ***Your Smart AI Prescription Logger & Intelli Health Search***

---
# Customer Friendly UI
![alt text](image-1.png)
# Semantic Understanding: 
![alt text](image.png)
# Retrive Information from any Prescription:
![alt text](image-2.png)
Even after uploading 3 prescriptions , it can understand every semantic information of every prescription and retrives intelligently.
## **🌍 The Problems We Try to Solve**

* Medical prescriptions are handwritten, messy, and hard to read.

* Tracking medicines, dosage, and history is confusing.

* Doctors and patients both waste time finding past prescriptions.  
* Semantic Search over prescription becomes almost impossible.

---

## **💡 The Vision**

To create a **smart healthcare assistant** that reads, organizes, and retrieves medical prescriptions —  
 bridging the gap between **handwritten chaos** and **digital clarity**.

LoopCare makes medical data:  
 ✅ Searchable  
 ✅ Understandable  
 ✅ Accessible

---

## **⚙️ The Flow**

![System Architecture](diagram-export-02-11-2025-13_26_18.svg)

**Step 1: Upload Prescription**  
 📸 Snap or upload an image of your prescription.

**Step 2: AI Reads It**  
 🧠 LoopCare’s **PaliGemma Vision-Language Model** transcribes the text and expands abbreviations like:

* “BID → twice a day”

* “PO → by mouth”

**Step 3: Auto Logging**  
 💾 The transcribed text is stored as **semantic vectors (MiniLM \+ FAISS)** for instant retrieval.

**Step 4: Ask Anything**  
 💬 Powered by **Groq \+ Llama 3.3 70B**, you can query naturally:

“Show me my last antibiotic prescription”  
 “What did my doctor prescribe for fever?”

---

## **🚀 Intelli Search for the Web**

✨ LoopCare’s **AI Intelli-Search** feature can be **embedded into any healthcare platform or portal**.  
 Plug it into:

* Hospital record systems 🏥

* Pharmacy management apps 💊

* Patient portals 👩‍⚕️

→ and instantly gain **semantic prescription search** & **AI-powered summaries**.

---

## **🤖 The Stack**

| Layer | Technology |
| ----- | ----- |
| 💬 Language Model | Groq (Llama 3.3 70B) |
| 🧠 Vision Model | PaliGemma 3B |
| 🔍 Retriever | MiniLM \+ FAISS |
| 🧩 Frontend | Streamlit |
| 🔐 Data Layer | Local FAISS Index / Persistent DB Ready |

---
## **🚀 How to Run Locally**

Follow these steps to get the application up and running on your machine.

1. Setup Your Environment

    First, it's highly recommended to create a virtual environment to keep your project dependencies separate.

2. Create a new virtual environment:

    * conda create -n my_env python=3.10

3. Activate the environment (on macOS/Linux):

    * conda activate my_env

4. Install all packages from the requirements file:
    
    * pip install -r requirements.txt

    Visit the Official PyTorch Website to find the correct command for your system (e.g., pip3 install torch torchvision torchaudio).

5. Run the App!

    Once everything is installed, use Streamlit to launch the app.

    Make sure you are in the project's root directory:

    * streamlit run app3.py


Your browser should automatically open to the application's local address, or manually give the port address!
## **❤️ Why LoopCare Wins**

* Real healthcare impact — automates messy, daily workflows.

* Fully local \+ private — no cloud dependency for sensitive data.

* Lightning fast with **Groq inference**.

* Beautiful UI.

* Ready-to-integrate API layer for real-world healthcare systems.

---

## **💫 Future Add-ons**

* 📊 Analytics Dashboard for doctors

* 🧾 Auto medication reminders

* 🔗 Integration with hospital EHR systems

* 🌐 Multi-language transcription & translation

---

## **👥 Team LoopCare**

Built by a team passionate about healthcare and AI-driven simplicity.  
 Every upload helps move us closer to **accessible, intelligent, patient-centered care.**

---

### **🌟 *“From paper to precision — one prescription at a time.”***

---

