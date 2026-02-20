import re
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DB_DIR = "db"

# ---------- Tools (demo actions) ----------
def get_leave_balance(employee_id: str) -> str:
    fake = {"E001": "12 days", "E002": "6 days"}
    return fake.get(employee_id, "Employee not found. Try E001 or E002 for demo.")

def create_hr_ticket(category: str, description: str) -> str:
    return f"Ticket created ✅ | Category: {category} | Ref: HR-{abs(hash(description)) % 10000}"

# ---------- RAG setup ----------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model="llama3.1", temperature=0)
parser = StrOutputParser()

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR assistant. Answer ONLY using the provided HR policy context. "
     "If the answer is not in the context, say you are not sure and advise contacting HR."),
    ("human",
     "Question: {question}\n\n"
     "HR Policy Context:\n{context}\n\n"
     "Answer (concise):")
])

def format_docs(docs) -> str:
    if not docs:
        return "No relevant policy text found."
    return "\n\n".join([f"[Source {i+1}] {d.page_content}" for i, d in enumerate(docs)])

def rag_answer(question: str) -> str:
    docs = retriever.invoke(question)  
    context = format_docs(docs)
    chain = rag_prompt | llm | parser
    return chain.invoke({"question": question, "context": context})

# ---------- Simple “agentic” router ----------
def route_and_respond(user_text: str) -> str:
    text = user_text.strip()

    # 1) Leave balance tool trigger
    # Examples: "leave balance for E001", "I'm E002, how many leave days left?"
    m = re.search(r"\b(E\d{3,})\b", text.upper())
    if ("leave balance" in text.lower()) or ("days left" in text.lower()) or ("leave remaining" in text.lower()):
        if m:
            return f"Your leave balance is: **{get_leave_balance(m.group(1))}**"
        return "Please provide your employee ID (e.g., **E001**) so I can check your leave balance."

    # 2) Ticket tool trigger
    # Examples: "open a ticket for payroll: salary wrong"
    if ("open a ticket" in text.lower()) or ("create a ticket" in text.lower()) or ("raise a ticket" in text.lower()):
        # Very simple extraction for demo
        # If user writes "category: X, description: Y" we parse it; otherwise default.
        cat_match = re.search(r"category\s*:\s*([^\n,]+)", text, re.IGNORECASE)
        desc_match = re.search(r"description\s*:\s*(.+)", text, re.IGNORECASE)

        category = cat_match.group(1).strip() if cat_match else "General"
        description = desc_match.group(1).strip() if desc_match else text

        return create_hr_ticket(category=category, description=description)

    # 3) Default: RAG policy answer
    return rag_answer(text)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="HR Chatbot (Ollama + RAG)", page_icon="💬")
st.title("AI HR Chatbot Demo-Envoy")
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