# MediPulse AI – Smart Health Assistant 🩺🤖

MediPulse AI is an intelligent health assistant that analyzes user symptoms and provides preliminary health insights using **Retrieval-Augmented Generation (RAG)** and **Google Gemini AI**.

The system retrieves relevant medical information from a knowledge base using **FAISS vector search** and generates structured responses to help users understand possible conditions, severity levels, and recommended actions.

This project demonstrates the integration of **Artificial Intelligence, NLP, vector databases, and web development** to build a real-world healthcare support application.

---

## 🚀 Features

### 🧠 AI Symptom Analysis
Users can enter symptoms in natural language, and the AI analyzes them to suggest possible health conditions.

### 📚 RAG-based Medical Knowledge Retrieval
The system retrieves relevant information from a medical dataset using:
- Sentence Transformers
- FAISS Vector Search

### ⚠️ Severity Detection
AI predicts the seriousness of symptoms:
- Mild  
- Moderate  
- Critical  

### 🏥 Emergency Awareness
If critical symptoms are detected, the system alerts the user and recommends visiting nearby hospitals.

### 💬 Chat-Based Interface
A modern chat interface allows users to interact with the AI assistant easily.

### 📷 Image Upload Support
Users can upload images of visible symptoms for future AI visual analysis integration.

---

## 🏗️ System Architecture
User Input
    ↓
Frontend (HTML + TailwindCSS + JavaScript)
    ↓
Flask API
    ↓
RAG Pipeline
    ↓
FAISS Vector Search
    ↓
Relevant Medical Context
    ↓
Google Gemini AI
    ↓
Structured JSON Response
    ↓
Interactive UI Display

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- TailwindCSS
- JavaScript
- Lucide Icons

### Backend
- Python
- Flask

### AI / Machine Learning
- Google Gemini API
- Sentence Transformers
- FAISS Vector Database

### Data Processing
- PyPDF2
- NumPy

---

## 📂 Project Structure
MediPulse_AI
│
├── app.py # Flask backend
├── rag_core.py # RAG pipeline and AI inference
├── text_extract.py # Dataset processing and FAISS index creation
├── requirements.txt
├── README.md
│
├── templates
│ └── index.html # Main frontend UI
│
├── static
│
├── data
│ └── DATASET.pdf # Medical knowledge dataset
│
└── index
├── faiss.index
└── documents.npy
