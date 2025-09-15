import Ollama  # Or direct Ollama Python API if available

# 1. Your college info documents
documents = [
    "Admissions close on July 31st.",
    "The library is open from 8AM to 8PM.",
    "KGISL Institute of Technology offers AI, ML, and Data Science degrees.",
    "Semester registration opens in June and closes in July.",
]

# 2. Simple retrieval based on keywords in query
def retrieve_docs(query, docs):
    query_words = set(query.lower().split())
    filtered_docs = [doc for doc in docs if query_words.intersection(doc.lower().split())]
    # Return up to 3 relevant docs or all if few
    return filtered_docs[:3] if filtered_docs else docs[:3]

# 3. Initialize Ollama LLM
llm = Ollama(model='llama3')

# 4. Create prompt using retrieved docs and question
def create_prompt(context_docs, question):
    context_text = "\n".join(context_docs)
    prompt = (
        "You are a helpful college admin assistant. Use only the information below to answer the question.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}\n\nANSWER:"
    )
    return prompt

# 5. Complete RAG function
def rag_answer(query):
    context = retrieve_docs(query, documents)
    prompt = create_prompt(context, query)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content

# Example usage
query = "What degrees are offered at KGISL Institute of Technology?"
answer = rag_answer(query)
print(answer)
