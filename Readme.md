# **🩺 LoopCare**

### ***Your Smart AI Prescription Logger & Intelli Health Search***
* Powered By FAISS , Groq , Pali Gemma Architecture
---

## **🌍 The Problems We Try to Solve**

While exploring existing healthcare tools, we realized something critical —  
there’s still **no intelligent chatbot or system** capable of truly **understanding and retrieving** information from medical prescriptions.  
And that gap isn’t just inconvenient — it’s *life-impacting*.

Without a system like this:

* Patients struggle with **unclear, handwritten prescriptions**, risking confusion and medication errors.  
* Tracking **dosage, timing, and medical history** becomes a daily headache for families and caregivers.  
* **Doctors waste valuable minutes** searching old records instead of focusing on patient care.  
* **Pharmacies face misreads** and delays, leading to frustration and potential harm.  
* And most importantly — **semantic understanding of prescriptions** remains nearly *impossible*, keeping healthcare stuck in the paper age.  

In a world where seconds can save lives,  
not having a smart, reliable way to understand prescriptions isn’t just inefficient —  
it’s **unacceptable**.  
LoopCare exists to change that.


---

## **💡 The Vision**

To create a **smart healthcare assistant** that reads, organizes, and retrieves medical prescriptions —  
 bridging the gap between **handwritten chaos** and **digital clarity**.

LoopCare makes medical data:  
 ✅ Searchable  
 ✅ Understandable  
 ✅ Accessible

---

# 🩺 **Features Of Loopcare**

## 🖥️ Customer Friendly UI
![alt text](image-1.png)

## 🧠 Semantic Understanding
![alt text](image.png)

## 🔍 Retrieve Information from Any Prescription
![alt text](image-2.png)

Even after uploading multiple prescriptions, LoopCare can understand the **semantic meaning** of each and **retrieve information intelligently**.

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

## ❤️ **Why LoopCare Wins**

LoopCare isn’t just another healthcare app — it’s a complete **rethink of how prescriptions are handled**.  
Every feature is built with real impact, privacy, and intelligence at its core.

* **Real Healthcare Impact** — Automates the messy, everyday workflows that doctors and patients struggle with.  
* **Truly Smart** — Understands and retrieves prescriptions with semantic precision, not just OCR text.  
* **100% Local & Private** — Your medical data stays with you. No cloud. No leaks. Total control.  
* **Lightning Fast** — Powered by **Groq inference**, enabling real-time, low-latency intelligence.  
* **Beautiful, Human-Centered UI** — Clean, intuitive, and built for effortless use by anyone.  
* **API-Ready Integration** — Easily plugs into hospital systems, pharmacy apps, and EHR platforms.

> **LoopCare wins because it turns handwritten confusion into digital clarity — instantly, privately, and intelligently.**

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

