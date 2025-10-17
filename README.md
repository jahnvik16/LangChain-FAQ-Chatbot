# 🧠 LangChain FAQ Chatbot (RAG System)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![Colab](https://img.shields.io/badge/Google%20Colab-Compatible-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📘 Project Overview

This project demonstrates a **Retrieval-Augmented Generation (RAG)** chatbot built using **LangChain**, **Hugging Face models**, and a **vector database (Chroma)**.
It combines your uploaded **OpenAI PDF** with additional **AI/ML FAQs** to create a powerful **FAQ assistant** that retrieves contextually relevant answers using embeddings-based search and a local LLM.

### 🧩 Key Capabilities

* Extracts and cleans text from uploaded PDF documents
* Generates embeddings using Sentence Transformers (`all-MiniLM-L6-v2`)
* Stores and searches documents in a **Chroma** vector database
* Uses a **free Hugging Face LLM (`flan-t5-base`)** for natural language responses
* Implements **LangChain-style RAG pipeline** (Retriever + Generator)
* Monitors performance metrics (query latency, similarity score)
* Includes a **Streamlit Dashboard** for interactive querying and monitoring

---

## 🧠 What You’ll Learn

| Concept                                  | Description                                        |
| ---------------------------------------- | -------------------------------------------------- |
| **RAG (Retrieval-Augmented Generation)** | How LLMs use external knowledge sources            |
| **Vector Databases**                     | Storing embeddings for fast semantic search        |
| **Sentence Transformers**                | Generating text embeddings for semantic similarity |
| **LangChain Chains**                     | Building retrieval + generation pipelines          |
| **Monitoring & Evaluation**              | Logging query metrics and visualizing them         |
| **Streamlit UI**                         | Deploying your chatbot as an interactive app       |

---

## ⚙️ Tech Stack

| Component       | Library / Tool                             |
| --------------- | ------------------------------------------ |
| **Language**    | Python                                     |
| **Framework**   | LangChain                                  |
| **Embeddings**  | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector DB**   | Chroma                                     |
| **LLM Model**   | Hugging Face `google/flan-t5-base`         |
| **PDF Parsing** | `pdfplumber`                               |
| **Monitoring**  | Pandas, Matplotlib, Seaborn                |
| **Interface**   | Streamlit                                  |
| **Environment** | Google Colab / Streamlit Cloud             |

---

## 🧱 System Architecture

**1️⃣ Data Extraction**

* Extracts clean text from the provided `OpenAI PDF.pdf` using `pdfplumber`.

**2️⃣ Knowledge Base Building**

* Appends additional AI/ML FAQs to enrich responses.
* Splits documents into overlapping chunks for semantic retrieval.

**3️⃣ Embeddings & Vector Database**

* Generates text embeddings with Sentence Transformers.
* Stores embeddings in Chroma (in-memory or persistent).

**4️⃣ RAG Pipeline**

* On query, retrieves top-K most relevant chunks.
* Builds context prompt and feeds into the `flan-t5-base` LLM.
* Returns a concise, context-grounded answer.

**5️⃣ Monitoring**

* Logs metrics (latency, similarity scores) to `rag_metrics_log.csv`.
* Visualizes trends using Matplotlib and Seaborn.
* Displays summary charts in Streamlit sidebar.

---

## 🚀 How to Run the Project

### 🧩 Option 1: Run in Google Colab

1. Upload your notebook + this repo’s files to Colab.
2. Install dependencies:

   ```bash
   !pip install -r requirements.txt
   ```
3. Run each cell in order.
4. Ask questions using:

   ```python
   ask_question("What are the seven lessons for enterprise AI adoption?")
   ```
5. Visualize monitoring dashboard inside Colab.

---

### 💬 Option 2: Run as a Streamlit App

1. After generating `last_rag_answer.json`, run:

   ```bash
   streamlit run langchain_faq_app.py
   ```
2. In Colab, expose the app with `pyngrok`:

   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_TOKEN")
   public_url = ngrok.connect(8501)
   print("Access your app at:", public_url)
   ```
3. Or push to GitHub and deploy on **[Streamlit Cloud](https://streamlit.io/cloud)** (free hosting).

---

## 📊 Monitoring & Logging

### Logged Metrics (`rag_metrics_log.csv`)

| Metric              | Description                                  |
| ------------------- | -------------------------------------------- |
| `retrieval_time_s`  | Time to find relevant chunks                 |
| `generation_time_s` | Time to generate answer                      |
| `mean_distance`     | Average similarity distance (lower = better) |
| `total_time_s`      | Total query latency                          |
| `num_docs`          | Number of retrieved documents                |

### Visualizations

* Query latency trend over time
* Mean similarity distance trend
* Query count summary

You can extend this dashboard using:

* `Streamlit` charts
* `Evidently AI` for drift detection
* `Weights & Biases` for advanced logging

---

## 📁 Repository Structure

```
LangChain-FAQ-Chatbot/
│
├── langchain_faq_app.py          # Streamlit front-end
├── requirements.txt              # Dependencies list
├── LICENSE                       # MIT License
├── README.md                     # This file
├── rag_metrics_log.csv           # Generated at runtime
├── last_rag_answer.json          # Sample answer (for Streamlit demo)
└── your_notebook.ipynb           # Full Colab pipeline
```

---

## 🧰 Installation Requirements

All dependencies are free-tier and work on Colab or locally.

```bash
pip install sentence-transformers
pip install transformers
pip install chromadb
pip install pdfplumber
pip install tqdm
pip install streamlit
pip install pyngrok
pip install matplotlib seaborn pandas
```
