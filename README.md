GenAI Chatbot Backend (APIs)

## 🌐 Deployed API

**Base URL:**
👉 [https://chatbotapis.onrender.com](https://chatbotapis.onrender.com)

---

## 📌 Overview

This repository contains the **backend APIs** for the GenAI-powered chatbot inspired by GitLab’s *Build in Public* philosophy.

The backend implements a **Retrieval-Augmented Generation (RAG)** pipeline that retrieves relevant information from **GitLab’s Handbook and Direction pages**, performs semantic search using **vector embeddings**, and generates accurate, context-aware responses using **Google Gemini AI**.

---

## 🎯 Objective

* Enable semantic search over GitLab documentation
* Generate reliable answers grounded in source content
* Provide a scalable API for frontend integration

---

## 🛠️ Tech Stack

* **Python**
* **FastAPI** (API framework)
* **Google Gemini AI** (Embeddings + LLM)
* **Pinecone** (Vector database)
* **dotenv** (Environment configuration)

---

## 🧠 System Design (RAG Flow)

1. User submits a query from the UI
2. Query is converted into vector embeddings using **Gemini**
3. Similar documents are retrieved from **Pinecone**
4. Retrieved context is passed to the Gemini LLM
5. Grounded response is generated and returned to the UI

---

## 🔌 API Endpoints

### 2️⃣ Chat Query API

```
POST /rag/query
```

**Request Body**

```json
{
  "query": "What is GitLab's remote work policy?"
  "k":"3"
}
```

**Response**

```json
{
  "query": "What is GitLab's remote work policy?"
  "answer": "GitLab operates as an all-remote company...",
}
```

---

## ⚙️ Environment Variables

Create a `.env` file:

```env
GEMINIAI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

---

## ▶️ Running the API Locally

### Prerequisites

* Python 3.10+
* Pinecone account
* Google Gemini API access

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Server

```bash
uvicorn main:app --reload
```

API will run at:

```
http://localhost:5000
```

---

## 🌍 Deployment

* Deployed on **Render**
* Automatic rebuild on code updates
* Uses environment variables for secure key management

---

## 🛡️ Guardrails & Reliability

* Responses limited to retrieved documentation context
* Prevents hallucinations using RAG grounding
* Graceful error handling for empty or invalid queries

---

## 🚀 Innovation Highlights

* Gemini-based embeddings for semantic accuracy
* Pinecone vector search for fast retrieval
* Modular RAG pipeline for scalability
* Designed with employee learning in mind

---

## 📈 Evaluation Alignment

* **Innovation:** Advanced RAG architecture
* **Code Quality:** Clean, modular API design
* **Approach:** Efficient retrieval + fast inference

---

## 📜 Notes

* Uses only publicly available GitLab documentation
* No user data is stored or logged
* Intended for learning and demonstration purposes

---

## 🚧 Future Enhancements

* Source citation confidence scoring
* Streaming responses
* Authentication & rate limiting
* Multi-index document support

---

## ✅ Related Repositories

* **Frontend UI:** [https://rkchatbot.netlify.app/](https://rkchatbot.netlify.app/)
