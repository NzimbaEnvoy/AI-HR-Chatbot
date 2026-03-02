AI HR Chatbot
Agentic RAG System using LangChain + Ollama + ChromaDB
An intelligent, locally-hosted HR assistant that combines Retrieval-Augmented Generation (RAG) with agentic intent routing to answer HR policy questions, check leave balances, and simulate HR ticket creation.
This project demonstrates practical implementation of:
-Large Language Models (Llama 3.1 via Ollama)
-Hybrid Retrieval (FAQ + Policy search)
-Vector Databases (ChromaDB)
-Query Rewriting for conversational memory
-LLM-based Intent Classification
-Streamlit conversational UI
-Local-first AI (no external API dependency)

System Architecture
The chatbot follows an agentic routing pattern

User Input
    ↓
Intent Classification (LLM Router)
    ↓
 ┌───────────────────────┬───────────────────────┬───────────────────────┐
 │ LEAVE_BALANCE         │ CREATE_TICKET         │ POLICY_QA (RAG)       │
 └───────────────────────┴───────────────────────┴───────────────────────┘
                                                 ↓
                                   Hybrid Vector Retrieval
                                                 ↓
                                        Llama 3.1 (Ollama)
                                                 ↓
                                           Streamlit UI


Project Strcuture 

AI-HR-Chatbot/
│
├── app.py                  # Streamlit application + agent logic
├── ingest.py               # Document ingestion + embedding pipeline
├── requirements.txt
│
├── data/
│   └── hr_docs/
│       ├── benefits.txt
│       ├── disciplinary_policy.txt
│       ├── faq.txt
│       ├── leave_policy.txt
│       ├── onboarding.txt
│       ├── payroll_policy.txt
│       └── remote_work_policy.txt
│
└── db/                     # Auto-generated Chroma vector store



