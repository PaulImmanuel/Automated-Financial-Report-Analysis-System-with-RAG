# 📈 Automated Financial Report Analysis System

An **Retrieval-Augmented Generation (RAG)** system for analyzing financial reports using Large Language Models (LLMs). The application enables users to query financial documents in natural language, compare companies, detect qualitative risks, and visualize financial trends through an interactive dashboard.

---

## 🌟 Overview

This project was developed as an 'InternsElite' internship project to simplify financial document analysis by combining **Retrieval-Augmented Generation (RAG)** with **parallel AI agents**.

The system processes both:

- 📄 Unstructured financial documents (10-Ks, Annual Reports, Earnings Reports, Analyst Notes)
- 📊 Structured financial datasets (CSV financial statements)

It provides intelligent answers, peer comparisons, risk analysis, and automatic chart generation through an intuitive Streamlit interface.

### Target Users

- Equity Research Analysts
- Portfolio Managers
- Retail Investors
- Financial Analysts
- Business Journalists

---

# 🚀 Features

### 🔍 Dual-Agent RAG Architecture

The system uses two AI agents working simultaneously.

### 📊 Primary Analyst Agent

- Semantic document retrieval using **Maximal Marginal Relevance (MMR)**
- Context-aware financial question answering
- Financial trend explanations
- Company performance summaries

### ⚠️ Auditor / Risk Agent

Runs in parallel to identify:

- Management speak
- Vague explanations
- Revenue decline justifications
- Potential qualitative red flags
- Source verification

---

### 📄 Dynamic Data Ingestion

Supports both structured and unstructured data.

- PDF Financial Reports
- Annual Reports
- Earnings Reports
- CSV Financial Statements

---

### 📈 Automated Trend Visualization

Automatically extracts requested financial metrics such as:

- Revenue
- Net Income
- Operating Margin
- EPS
- Growth Rate

and generates interactive time-series charts.

---

### 🤝 Intelligent Peer Comparison

Automatically detects comparison questions such as:

> Compare Apple and Microsoft

and generates side-by-side comparisons including:

- Revenue
- Market Cap
- Profitability
- Growth
- Key observations

---

## 🏗️ System Architecture

```
   Financial Reports (PDF)
            │
            ▼
    Document Loader
            │
            ▼
Text Chunking (RecursiveCharacterTextSplitter)
            │
            ▼
 Sentence Embeddings(llama3)
            │
            ▼
    FAISS Vector Store
            │
            ▼
    Semantic Retrieval
            │
     ┌───────────────┐
     │               │
     ▼               ▼
Primary Agent   Auditor Agent
     │               │
     └──────┬────────┘
            ▼
     Final AI Response
            │
            ▼
 Streamlit Enterprise Dashboard
```

---

# 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| Language | Python |
| Frontend | Streamlit |
| LLM | Ollama (Llama 3) |
| Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | all-MiniLM-L6-v2 |
| PDF Processing | PyPDF2 |
| Data Processing | Pandas |
| Retrieval | Maximal Marginal Relevance (MMR) |

---

# 📸 Screenshots

## 📊 Data Processing & Analytics

### Processing Financial Data

> Add screenshot here

```
images/data_processing.png
```

### Trend Analysis

> Add screenshot here

```
images/trend_analysis.png
```

---

## 🤖 Dual-Agent Responses

### Primary Analyst

> Add screenshot here

```
images/primary_agent.png
```

### Auditor Agent

> Add screenshot here

```
images/auditor_agent.png
```

---

## 📈 Advanced Analytics

### Trend Detection

> Add screenshot here

```
images/trend_detection.png
```

### Backend Parallel Processing

> Add screenshot here

```
images/backend_processing.png
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/PaulImmanuel/Automated-Financial-Report-Analysis-System-with-RAG.git

cd Automated-Financial-Report-Analysis-System-with-RAG
```

---

## 2. Install Dependencies

```bash
pip install streamlit
langchain
langchain-community
langchain-core
langchain-text-splitters
faiss-cpu
sentence-transformers
PyPDF2
pandas
```

---

## 3. Install Ollama

Download and install Ollama from:

https://ollama.com/

---

## 4. Download the Llama 3 Model

```bash
ollama run llama3
```

---

## 5. Launch the Application

```bash
streamlit run financial_rag_app.py
```

---

# 💡 Example Queries

- Summarize Tesla's latest annual report.
- Compare Apple and Microsoft revenue growth.
- Why did operating margins decline?
- Show revenue trend over the last five years.
- Identify qualitative risks mentioned by management.
- Compare EPS and revenue between two companies.

---

# 🔮 Future Enhancements

- 🌐 Live financial market API integration (Yahoo Finance, Alpha Vantage)
- 📊 Interactive Plotly dashboards
- 📑 Improved extraction of financial tables embedded in PDFs
- 🧮 GAAP-based mathematical consistency checks
- ☁️ Cloud deployment (AWS/Azure/GCP)
- 💬 Multi-document conversational memory
- 📈 Portfolio-level financial analysis

---

# 📜 License

This project is intended for educational and research purposes.

---

# 👨‍💻 Author

**Paul Immanuel J**

B.Tech Computer Science & Engineering

InternsElite Internship Major Project
```
