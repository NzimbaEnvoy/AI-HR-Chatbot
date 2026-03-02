import os
import re
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional import; if it fails, app still works without multi-query.
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    HAS_MULTIQUERY = True
except Exception:
    HAS_MULTIQUERY = False

DB_DIR = "db"
COLLECTION = "hr_policies"

# ---------- Demo tools ----------
def get_leave_balance(employee_id: str) -> str:
    fake = {"E001": "12 days", "E002": "6 days"}
    return fake.get(employee_id, "Employee not found. Try E001 or E002 for demo.")

def create_hr_ticket(category: str, description: str) -> str:
    return f"Ticket created ✅ | Category: {category} | Ref: HR-{abs(hash(description)) % 10000}"


# ---------- LLM + Vector DB ----------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectordb = Chroma(
    collection_name=COLLECTION,
    persist_directory=DB_DIR,
    embedding_function=embeddings
)

llm = ChatOllama(model="llama3.1", temperature=0)
parser = StrOutputParser()

# ---------- Retrievers ----------
# FAQ retriever (precise)
faq_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5, "filter": {"doc_type": "faq"}},
)

# Policy retriever (broader)
policy_base_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 25, "lambda_mult": 0.5},
)

policy_retriever = (
    MultiQueryRetriever.from_llm(retriever=policy_base_retriever, llm=llm)
    if HAS_MULTIQUERY else policy_base_retriever
)

# ---------- Prompts ----------
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the user message into one of:\n"
     "LEAVE_BALANCE, CREATE_TICKET, POLICY_QA.\n"
     "Return ONLY the label."),
    ("human", "{text}")
])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR assistant. Use ONLY the provided HR policy context.\n"
     "If the answer is not clearly supported, say you are not sure and advise contacting HR.\n"
     "Always include citations like (Source 1) or (Source 2) after key claims.\n"
     "Be concise and practical."),
    ("human",
     "Question:\n{question}\n\n"
     "HR Policy Context:\n{context}\n\n"
     "Answer (concise, with citations):")
])

query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question into a standalone HR question using the chat history.\n"
     "Do NOT answer the question. Only return the rewritten question."),
    ("human",
     "Chat history:\n{history}\n\nLatest user question:\n{question}")
])

# ---------- Helpers ----------
def format_docs(docs) -> str:
    if not docs:
        return "No relevant policy text found."
    blocks = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown")
        src_name = os.path.basename(src)
        blocks.append(f"[Source {i+1}: {src_name}] {d.page_content}")
    return "\n\n".join(blocks)

def build_history_text(max_msgs: int = 6) -> str:
    msgs = st.session_state.messages[-max_msgs:]
    return "\n".join([f"{m['role']}: {m['content']}" for m in msgs])

def rewrite_query_with_history(user_text: str) -> str:
    history = build_history_text()
    chain = query_rewrite_prompt | llm | parser
    return chain.invoke({"history": history, "question": user_text}).strip()

def route_intent(text: str) -> str:
    return (router_prompt | llm | parser).invoke({"text": text}).strip().upper()

def rag_answer(user_text: str) -> str:
    # Make follow-ups work better
    standalone_q = rewrite_query_with_history(user_text)

    # Hybrid retrieval: FAQ first (high precision), then policy
    faq_docs = faq_retriever.invoke(standalone_q)
    policy_docs = policy_retriever.invoke(standalone_q)

    # Combine + lightly de-duplicate by source+content prefix
    seen = set()
    merged = []
    for d in (faq_docs + policy_docs):
        key = (d.metadata.get("source", ""), d.page_content[:120])
        if key not in seen:
            seen.add(key)
            merged.append(d)

    context = format_docs(merged)
    chain = rag_prompt | llm | parser
    answer = chain.invoke({"question": standalone_q, "context": context}).strip()

    # Guardrail: if retrieval failed, force a safe response
    if "No relevant policy text found" in context:
        return "I’m not sure based on the available HR documents. Please contact HR for confirmation."

    return answer

def route_and_respond(user_text: str) -> str:
    text = user_text.strip()
    intent = route_intent(text)

    if intent == "LEAVE_BALANCE":
        m = re.search(r"\b(E\d{3,})\b", text.upper())
        if m:
            return f"Your leave balance is: **{get_leave_balance(m.group(1))}**"
        return "Please provide your employee ID (e.g., **E001**) so I can check your leave balance."

    if intent == "CREATE_TICKET":
        cat_match = re.search(r"category\s*:\s*([^\n,]+)", text, re.IGNORECASE)
        desc_match = re.search(r"description\s*:\s*(.+)", text, re.IGNORECASE)
        category = cat_match.group(1).strip() if cat_match else "General"
        description = desc_match.group(1).strip() if desc_match else text
        return create_hr_ticket(category=category, description=description)

    # Default: policy Q&A via RAG
    return rag_answer(text)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="HR Chatbot (Ollama + RAG)", page_icon="💬")
st.title("Agentic AI HR-Chatbot Demo")
st.caption("Try: 'What is annual leave policy?' | 'Leave balance for E001' | 'Open a ticket for payroll issue'")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me about HR policies, benefits, leave…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    reply = route_and_respond(user_input)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)