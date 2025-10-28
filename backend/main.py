import os
import re
import tempfile
import uuid
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import shutil

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
app = FastAPI(title="College Admin Chatbot API", version="2.0.0")

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("documents", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
vectorstore = None
conversation_chain = None
user_sessions = {}
DOCUMENTS_DIR = "documents"
UPLOADS_DIR = "uploads"

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

class AdminLogin(BaseModel):
    email: str
    password: str

class DocumentInfo(BaseModel):
    name: str
    size: int
    created: str
    pages: int

class UserSession:
    def __init__(self):
        self.context = {}
        self.last_activity = datetime.now()
        self.chat_history = []
    
    def is_active(self):
        return (datetime.now() - self.last_activity).seconds < 1800  # 30 min
    
    def add_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content, "timestamp": datetime.now()})
        if len(self.chat_history) > 50:  # Keep last 50 messages
            self.chat_history = self.chat_history[-50:]

def get_session(user_id: str):
    if user_id not in user_sessions or not user_sessions[user_id].is_active():
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
        print(f"✓ Loaded PDF: {len(documents)} pages from {file_path}")
        return documents
    except Exception as e:
        print(f"✗ PDF loading failed for {file_path}: {e}")
        return []

def setup_vectorstore(documents):
    """Create FAISS vectorstore"""
    if not documents:
        print("✗ No documents to process for vectorstore")
        return None
    
    try:
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✓ Vectorstore created successfully")
        return vectorstore
    except Exception as e:
        print(f"✗ Vectorstore creation failed: {e}")
        return None

def create_chain(vectorstore):
    """Create conversational chain"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0,
            groq_api_key=groq_api_key
        )
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 20}
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        prompt_template = """You are a helpful AI assistant for a college administration system. Use the following context to answer the user's question.

Context: {context}

Chat History: {chat_history}

Question: {question}

If the information is not available in the context, politely guide the user to contact the appropriate college department. Be professional, concise, and helpful.

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        
        return chain
    except Exception as e:
        print(f"✗ Chain creation failed: {e}")
        return None

# ============================================================================
# LOCAL FILE MANAGEMENT
# ============================================================================
def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to local directory"""
    try:
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOADS_DIR, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Move to documents directory
        final_path = os.path.join(DOCUMENTS_DIR, unique_filename)
        shutil.move(file_path, final_path)
        
        print(f"✓ Saved file: {unique_filename}")
        return unique_filename
    except Exception as e:
        print(f"✗ File save failed: {e}")
        raise

def list_local_documents() -> List[DocumentInfo]:
    """List all documents in local directory"""
    documents = []
    try:
        for filename in os.listdir(DOCUMENTS_DIR):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(DOCUMENTS_DIR, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    # Count pages by loading the PDF
                    pages = 0
                    try:
                        docs = load_pdf(file_path)
                        pages = len(docs)
                    except:
                        pages = 0
                    
                    documents.append(DocumentInfo(
                        name=filename,
                        size=stat.st_size,
                        created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        pages=pages
                    ))
        return documents
    except Exception as e:
        print(f"✗ Document listing failed: {e}")
        return []

def delete_local_document(filename: str) -> bool:
    """Delete document from local directory"""
    try:
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Deleted file: {filename}")
            return True
        return False
    except Exception as e:
        print(f"✗ File deletion failed: {e}")
        return False

def reload_documents():
    """Reload all documents from local directory"""
    global vectorstore, conversation_chain
    
    try:
        files = list_local_documents()
        if not files:
            print("ℹ No documents found in local directory")
            return False, "No documents found"
        
        all_docs = []
        for file_info in files:
            file_path = os.path.join(DOCUMENTS_DIR, file_info.name)
            if os.path.exists(file_path):
                docs = load_pdf(file_path)
                all_docs.extend(docs)
        
        if all_docs:
            vectorstore = setup_vectorstore(all_docs)
            if vectorstore:
                conversation_chain = create_chain(vectorstore)
                if conversation_chain:
                    return True, f"Successfully loaded {len(files)} documents with {len(all_docs)} pages"
                else:
                    return False, "Failed to create conversation chain"
            else:
                return False, "Failed to create vectorstore"
        
        return False, "No documents could be processed"
    
    except Exception as e:
        print(f"✗ Document reload failed: {e}")
        return False, f"Reload failed: {str(e)}"

def save_chat(user_id: str, user_type: str, message: str, response: str):
    """Save chat history to local file (optional)"""
    try:
        chat_log_dir = "chat_logs"
        os.makedirs(chat_log_dir, exist_ok=True)
        
        log_file = os.path.join(chat_log_dir, f"chats_{datetime.now().strftime('%Y%m%d')}.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} | {user_id} | {user_type} | Q: {message} | A: {response}\n")
    except Exception as e:
        print(f"Chat save error: {e}")

# ============================================================================
# AUTHENTICATION
# ============================================================================
def verify_admin(email: str, password: str):
    """Verify admin credentials"""
    try:
        admin_emails = os.getenv("ADMIN_EMAILS", "admin@college.edu,administrator@college.edu").split(",")
        # Simple authentication - in production, use proper auth system
        return email.strip() in [e.strip() for e in admin_emails if e.strip()]
    except:
        return False

async def get_current_admin(token: str = Form(...)):
    """Dependency for admin authentication"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Simple token validation
    try:
        # For demo purposes, we'll use a simple validation
        return {"email": "admin@college.edu", "role": "admin"}
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main chat interface"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        # Fallback to embedded HTML
        return HTMLResponse(content=generate_html_interface())

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Serve the admin dashboard"""
    try:
        with open("static/admin.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        # Fallback to embedded HTML
        return HTMLResponse(content=generate_admin_interface())

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    groq_available = bool(os.getenv("GROQ_API_KEY"))
    documents = list_local_documents()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "groq_api_available": groq_available,
        "documents_loaded": vectorstore is not None,
        "documents_count": len(documents),
        "active_sessions": len(user_sessions),
        "system": "local_files"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage):
    """Main chat endpoint"""
    try:
        session = get_session(msg.user_id)
        session.add_message("user", msg.message)
        
        # Handle greetings and basic queries
        lower_msg = msg.message.lower().strip()
        if lower_msg in ['hi', 'hello', 'hey', 'hola']:
            response_text = "Hello! I'm your college administration assistant. How can I help you with college information today?"
        elif lower_msg in ['help', 'what can you do', '?']:
            response_text = "I can help you with college-related information including admissions, courses, faculty, schedules, and administrative procedures. What would you like to know?"
        elif lower_msg in ['thanks', 'thank you', 'thank']:
            response_text = "You're welcome! Is there anything else I can help you with?"
        elif not conversation_chain:
            response_text = "I'm currently processing the college documents. Please try again in a moment or contact the administration office directly for immediate assistance."
        else:
            # Use RAG chain for document-based responses
            result = conversation_chain.invoke({'question': msg.message})
            response_text = result.get('answer', 'I apologize, but I cannot provide an answer at the moment. Please contact the administration office for assistance.')
        
        # Format response
        response_text = format_response(response_text)
        session.add_message("assistant", response_text)
        
        # Save to local file
        save_chat(msg.user_id, msg.user_type, msg.message, response_text)
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        print(f"Chat error: {e}")
        error_response = "I apologize for the technical issue. Please contact the admin office for assistance or try again later."
        return ChatResponse(
            response=error_response,
            timestamp=datetime.now().isoformat()
        )

@app.post("/api/admin/upload")
async def upload_document(file: UploadFile = File(...), admin: dict = Depends(get_current_admin)):
    """Upload PDF documents (Admin only)"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")
    
    try:
        # Save file locally
        filename = save_uploaded_file(file)
        
        # Reload documents to update the knowledge base
        reload_success, reload_msg = reload_documents()
        
        return {
            "message": f"File uploaded successfully as {filename}",
            "reload_status": reload_msg,
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(500, f"Upload error: {str(e)}")

@app.get("/api/admin/documents")
async def list_documents(admin: dict = Depends(get_current_admin)):
    """List all uploaded documents (Admin only)"""
    docs = list_local_documents()
    return {"documents": docs, "count": len(docs)}

@app.delete("/api/admin/documents/{filename}")
async def delete_document(filename: str, admin: dict = Depends(get_current_admin)):
    """Delete a document (Admin only)"""
    success = delete_local_document(filename)
    if success:
        # Reload documents after deletion
        reload_documents()
        return {"message": "File deleted successfully", "status": "success"}
    else:
        raise HTTPException(500, "File deletion failed")

@app.post("/api/admin/reload")
async def reload_docs(admin: dict = Depends(get_current_admin)):
    """Reload all documents (Admin only)"""
    success, msg = reload_documents()
    if success:
        return {"message": msg, "status": "success"}
    raise HTTPException(500, msg)

@app.post("/api/admin/login")
async def admin_login(login_data: AdminLogin):
    """Admin login endpoint"""
    try:
        # Simple authentication
        if verify_admin(login_data.email, login_data.password):
            # Generate a simple token
            token = str(uuid.uuid4())
            return {
                "token": token, 
                "status": "success",
                "user": {"email": login_data.email, "role": "admin"}
            }
        else:
            raise HTTPException(401, "Invalid admin credentials")
    except Exception as e:
        raise HTTPException(500, f"Login error: {str(e)}")

# ============================================================================
# UTILITIES
# ============================================================================
def format_response(text: str) -> str:
    """Format response text"""
    if not text:
        return "I apologize, but I couldn't generate a response. Please try again."
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add formatting for contact information
    text = re.sub(r'(\+?\d[\d\s-]{8,}\d)', r'📞 \1', text)
    text = re.sub(r'(\S+@\S+\.\S+)', r'📧 \1', text)
    
    return text

def cleanup_sessions():
    """Clean up expired user sessions"""
    current_time = datetime.now()
    expired_users = []
    
    for user_id, session in user_sessions.items():
        if not session.is_active():
            expired_users.append(user_id)
    
    for user_id in expired_users:
        del user_sessions[user_id]
    
    if expired_users:
        print(f"Cleaned up {len(expired_users)} expired sessions")

def generate_html_interface():
    """Generate fallback HTML interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>College Admin Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .chat-messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #007bff; color: white; text-align: right; }
            .bot { background: #f8f9fa; color: #333; }
            .input-group { display: flex; gap: 10px; }
            input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>College Admin Chatbot</h1>
            <div class="chat-messages" id="chatMessages">
                <div class="message bot">Hello! How can I help you with college information today?</div>
            </div>
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Ask about college information...">
                <button onclick="sendMessage()">Send</button>
            </div>
            <p><a href="/admin">Admin Panel</a></p>
        </div>
        <script>
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;
                
                // Add user message
                const messages = document.getElementById('chatMessages');
                messages.innerHTML += `<div class="message user">${message}</div>`;
                input.value = '';
                
                // Send to API
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message, user_id: 'web_user' })
                    });
                    const data = await response.json();
                    messages.innerHTML += `<div class="message bot">${data.response}</div>`;
                    messages.scrollTop = messages.scrollHeight;
                } catch (error) {
                    messages.innerHTML += `<div class="message bot">Sorry, I encountered an error. Please try again.</div>`;
                }
            }
        </script>
    </body>
    </html>
    """

def generate_admin_interface():
    """Generate fallback admin interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Admin Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 10px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            input, select { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Admin Dashboard</h1>
            <p><a href="/">Back to Chat</a></p>
            
            <div class="card">
                <h3>Upload Documents</h3>
                <input type="file" id="fileInput" accept=".pdf">
                <button onclick="uploadDocument()">Upload PDF</button>
            </div>
            
            <div class="card">
                <h3>Document Management</h3>
                <button onclick="loadDocuments()">Refresh Documents</button>
                <button onclick="reloadKnowledge()">Reload AI Knowledge</button>
                <div id="documentsList">Loading...</div>
            </div>
            
            <div class="card">
                <h3>System Status</h3>
                <div id="systemStatus">Loading...</div>
            </div>
        </div>
        
        <script>
            async function uploadDocument() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) return alert('Please select a file');
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('token', 'demo-token');
                
                try {
                    const response = await fetch('/api/admin/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    alert(result.message);
                    loadDocuments();
                } catch (error) {
                    alert('Upload failed');
                }
            }
            
            async function loadDocuments() {
                try {
                    const response = await fetch('/api/admin/documents?token=demo-token');
                    const data = await response.json();
                    document.getElementById('documentsList').innerHTML = 
                        data.documents.map(doc => `<div>${doc.name} (${doc.size} bytes) <button onclick="deleteDocument('${doc.name}')">Delete</button></div>`).join('');
                } catch (error) {
                    document.getElementById('documentsList').innerHTML = 'Error loading documents';
                }
            }
            
            async function deleteDocument(filename) {
                if (!confirm('Delete ' + filename + '?')) return;
                try {
                    const response = await fetch(`/api/admin/documents/${filename}?token=demo-token`, {
                        method: 'DELETE'
                    });
                    const result = await response.json();
                    alert(result.message);
                    loadDocuments();
                } catch (error) {
                    alert('Delete failed');
                }
            }
            
            async function reloadKnowledge() {
                try {
                    const response = await fetch('/api/admin/reload?token=demo-token', {
                        method: 'POST'
                    });
                    const result = await response.json();
                    alert(result.message);
                } catch (error) {
                    alert('Reload failed');
                }
            }
            
            // Load documents on page load
            loadDocuments();
        </script>
    </body>
    </html>
    """

# ============================================================================
# STARTUP AND BACKGROUND TASKS
# ============================================================================
@app.on_event("startup")
async def startup():
    print("🚀 Starting College Admin Chatbot API...")
    print(f"🔑 Groq API available: {bool(os.getenv('GROQ_API_KEY'))}")
    print(f"📁 Documents directory: {DOCUMENTS_DIR}")
    
    # Load initial documents
    print("🔄 Loading documents from local directory...")
    success, msg = reload_documents()
    print(f"📚 {msg}")
    
    print("✅ API Ready and listening!")

@app.on_event("shutdown")
async def shutdown():
    print("🛑 Shutting down College Admin Chatbot API...")
    cleanup_sessions()

# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail, "status": "error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "status": "error"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)