# pip install langgraph langchain-core typing_extensions faiss-cpu sentence-transformers

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from typing_extensions import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.llms import Ollama

# 1. College Info Documents
documents = [
    Document(page_content="Admissions close on July 31st."),
    Document(page_content="The library is open from 8AM to 8PM."),
    Document(page_content="KGISL Institute of Technology offers AI, ML, and Data Science degrees."),
    Document(page_content="Semester registration opens in June and closes in July."),
]

# 2. Embeddings + Vector Store
embed_model = 'all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embed_model)
vector_db = FAISS.from_documents(documents, embedding=embeddings)

# 3. LLM Setup (Ollama, replace as needed)
llm = Ollama(model='llama3')

# 4. Define LangGraph State
class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 5. Define retrieval and generation steps
def search(state: RAGState):
    retrieved_docs = vector_db.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: RAGState):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    system_prompt = (
        "You are a college admin assistant. Use only the context provided to answer. "
        "If not found, say 'I don't know.'"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONTEXT:\n{docs_content}\nQUESTION:\n{state['question']}"}
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}

# 6. Build LangGraph workflow
graph_builder = StateGraph(RAGState)
graph_builder.add_node("search", search)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "search")
graph_builder.add_edge("search", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()

# 7. Run a query
user_query = "Which degrees does KGISL Institute of Technology offer?"
result = graph.invoke({"question": user_query, "context": [], "answer": ""})
print(result['answer'])
