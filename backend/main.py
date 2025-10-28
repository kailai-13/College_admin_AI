import os
import re
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, storage, firestore

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# ============================================================================
# CONFIGURATION
# ============================================================================
app = FastAPI(title="College Admin Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase
try:
    if not firebase_admin._apps:
        firebase_config = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {
            'storageBucket': f"{firebase_config['project_id']}.firebasestorage.app"
        })
    
    bucket = storage.bucket()
    db = firestore.client()
    FIREBASE_INITIALIZED = True
    print("✓ Firebase initialized")
except Exception as e:
    print(f"✗ Firebase initialization failed: {e}")
    FIREBASE_INITIALIZED = False
    db = None

vectorstore = None
conversation_chain = None
user_sessions = {}

# ============================================================================
# MODELS
# ============================================================================
class ChatMessage(BaseModel):
    message: str
    user_id: str = "anonymous"
    user_type: str = "student"  # student, staff, admin

class ChatResponse(BaseModel):
    response: str
    timestamp: str

class UserSession:
    def __init__(self):
        self.context = {}
        self.last_activity = datetime.now()
    
    def is_active(self):
        return (datetime.now() - self.last_activity).seconds < 1800  # 30 min

def get_session(user_id: str):
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession()
    user_sessions[user_id].last_activity = datetime.now()
    return user_sessions[user_id]

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
def load_pdf(file_path: str):
    """Load PDF documents"""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"✓ Loaded PDF: {len(documents)} pages")
        return documents
    except Exception as e:
        print(f"✗ PDF loading failed: {e}")
        return []

def setup_vectorstore(documents):
    """Create FAISS vectorstore"""
    if not documents:
        return None
    
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✓ Vectorstore created")
    return vectorstore

def create_chain(vectorstore):
    """Create conversational chain"""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )
    
    memory = ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True
    )
    
    prompt_template = """You are a helpful AI assistant for a college administration.
    
Provide accurate information based on the context below. If information is not available,
politely guide the user to contact the appropriate department.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer professionally and concisely:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain

# ============================================================================
# FIREBASE FUNCTIONS
# ============================================================================
def upload_to_firebase(file_path: str, filename: str):
    """Upload file to Firebase Storage"""
    if not FIREBASE_INITIALIZED:
        return False, "Firebase not initialized"
    
    try:
        blob = bucket.blob(f"documents/{filename}")
        blob.upload_from_filename(file_path)
        print(f"✓ Uploaded: {filename}")
        return True, "Upload successful"
    except Exception as e:
        return False, str(e)

def list_firebase_documents():
    """List all documents in Firebase"""
    if not FIREBASE_INITIALIZED:
        return []
    
    try:
        blobs = bucket.list_blobs(prefix="documents/")
        return [{"name": blob.name.replace("documents/", ""), 
                 "size": blob.size} for blob in blobs if blob.name.endswith('.pdf')]
    except:
        return []

def download_from_firebase(filename: str):
    """Download file from Firebase"""
    if not FIREBASE_INITIALIZED:
        return None
    
    try:
        blob = bucket.blob(f"documents/{filename}")
        temp_path = f"/tmp/{filename}"
        blob.download_to_filename(temp_path)
        return temp_path
    except:
        return None

def reload_documents():
    """Reload all documents from Firebase"""
    global vectorstore, conversation_chain
    
    files = list_firebase_documents()
    if not files:
        return False, "No documents found"
    
    all_docs = []
    for file_info in files:
        temp_path = download_from_firebase(file_info['name'])
        if temp_path:
            docs = load_pdf(temp_path)
            all_docs.extend(docs)
            os.remove(temp_path)
    
    if all_docs:
        vectorstore = setup_vectorstore(all_docs)
        conversation_chain = create_chain(vectorstore)
        return True, f"Loaded {len(files)} documents"
    
    return False, "No documents processed"

def save_chat(user_id: str, user_type: str, message: str, response: str):
    """Save chat history to Firestore"""
    if not db:
        return
    
    try:
        db.collection('chats').add({
            'user_id': user_id,
            'user_type': user_type,
            'message': message,
            'response': response,
            'timestamp': firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"Chat save error: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    return {
        "message": "College Admin Chatbot API",
        "status": "running",
        "firebase": FIREBASE_INITIALIZED,
        "documents_loaded": vectorstore is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage):
    """Main chat endpoint"""
    try:
        session = get_session(msg.user_id)
        
        # Handle greetings
        if msg.message.lower().strip() in ['hi', 'hello', 'hey']:
            response_text = "Hello! I'm here to help with college information. How can I assist you today?"
            save_chat(msg.user_id, msg.user_type, msg.message, response_text)
            return ChatResponse(
                response=response_text,
                timestamp=datetime.now().isoformat()
            )
        
        # Use RAG chain if available
        if conversation_chain:
            result = conversation_chain.invoke({'question': msg.message})
            response_text = result.get('answer', '')
        else:
            response_text = "I'm currently processing documents. Please try again in a moment or contact the admin office."
        
        # Format response
        response_text = format_response(response_text)
        
        # Save to database
        save_chat(msg.user_id, msg.user_type, msg.message, response_text)
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        print(f"Chat error: {e}")
        return ChatResponse(
            response="I apologize for the technical issue. Please contact the admin office for assistance.",
            timestamp=datetime.now().isoformat()
        )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload PDF documents"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    temp_path = f"/tmp/{file.filename}"
    
    try:
        content = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        success, msg = upload_to_firebase(temp_path, file.filename)
        os.remove(temp_path)
        
        if success:
            reload_success, reload_msg = reload_documents()
            return {"message": reload_msg, "status": "success"}
        else:
            raise HTTPException(500, msg)
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(500, str(e))

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    docs = list_firebase_documents()
    return {"documents": docs, "count": len(docs)}

@app.post("/reload")
async def reload():
    """Reload all documents"""
    success, msg = reload_documents()
    if success:
        return {"message": msg, "status": "success"}
    raise HTTPException(500, msg)

# ============================================================================
# UTILITIES
# ============================================================================
def format_response(text: str) -> str:
    """Format response text"""
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add phone number markers
    text = re.sub(r'(\+?\d[\d\s-]{8,})', r'[TEL:\1]', text)
    
    # Add email markers
    text = re.sub(r'(\S+@\S+\.\S+)', r'[EMAIL:\1]', text)
    
    return text

# ============================================================================
# STARTUP
# ============================================================================
@app.on_event("startup")
async def startup():
    print("Starting College Admin Chatbot...")
    if FIREBASE_INITIALIZED:
        success, msg = reload_documents()
        print(msg)
    print("✓ API Ready!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)