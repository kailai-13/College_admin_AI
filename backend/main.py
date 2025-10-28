import os
import re
import tempfile
import uuid
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, storage, firestore, auth
from firebase_admin.exceptions import FirebaseError

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

# Create static directory for frontend files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        # Better environment variable handling for private key
        private_key = os.getenv("FIREBASE_PRIVATE_KEY", "")
        
        # Ensure proper newline handling
        if private_key:
            private_key = private_key.replace('\\n', '\n')
        
        firebase_config = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": private_key,
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        
        # Validate required fields
        required_fields = ['project_id', 'private_key', 'client_email']
        for field in required_fields:
            if not firebase_config.get(field):
                raise ValueError(f"Missing required Firebase config: {field}")
        
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {
            'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET", f"{firebase_config['project_id']}.appspot.com")
        })
    
    bucket = storage.bucket()
    db = firestore.client()
    FIREBASE_INITIALIZED = True
    print("✓ Firebase initialized successfully")
    
except Exception as e:
    print(f"✗ Firebase initialization failed: {e}")
    print(f"✗ Project ID: {os.getenv('FIREBASE_PROJECT_ID')}")
    print(f"✗ Client Email: {os.getenv('FIREBASE_CLIENT_EMAIL')}")
    FIREBASE_INITIALIZED = False
    db = None
    bucket = None

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

class AdminLogin(BaseModel):
    email: str
    password: str

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
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
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
# FIREBASE FUNCTIONS
# ============================================================================
def upload_to_firebase(file_path: str, filename: str):
    """Upload file to Firebase Storage"""
    if not FIREBASE_INITIALIZED:
        return False, "Firebase not initialized"
    
    try:
        # Generate unique filename
        file_extension = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        blob = bucket.blob(f"documents/{unique_filename}")
        
        blob.upload_from_filename(file_path)
        
        # Make the file publicly readable (optional)
        blob.make_public()
        
        print(f"✓ Uploaded: {filename} as {unique_filename}")
        return True, unique_filename
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return False, str(e)

def list_firebase_documents():
    """List all documents in Firebase"""
    if not FIREBASE_INITIALIZED:
        return []
    
    try:
        blobs = bucket.list_blobs(prefix="documents/")
        documents = []
        for blob in blobs:
            if blob.name.endswith('.pdf'):
                documents.append({
                    "name": blob.name.replace("documents/", ""),
                    "size": blob.size,
                    "url": blob.public_url,
                    "created": blob.time_created
                })
        return documents
    except Exception as e:
        print(f"✗ Document listing failed: {e}")
        return []

def download_from_firebase(filename: str):
    """Download file from Firebase"""
    if not FIREBASE_INITIALIZED:
        return None
    
    try:
        blob = bucket.blob(f"documents/{filename}")
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        blob.download_to_filename(temp_path)
        return temp_path
    except Exception as e:
        print(f"✗ Download failed for {filename}: {e}")
        return None

def delete_from_firebase(filename: str):
    """Delete file from Firebase"""
    if not FIREBASE_INITIALIZED:
        return False, "Firebase not initialized"
    
    try:
        blob = bucket.blob(f"documents/{filename}")
        blob.delete()
        print(f"✓ Deleted: {filename}")
        return True, "File deleted successfully"
    except Exception as e:
        print(f"✗ Delete failed: {e}")
        return False, str(e)

def reload_documents():
    """Reload all documents from Firebase"""
    global vectorstore, conversation_chain
    
    try:
        files = list_firebase_documents()
        if not files:
            print("ℹ No documents found in Firebase")
            return False, "No documents found"
        
        all_docs = []
        for file_info in files:
            temp_path = download_from_firebase(file_info['name'])
            if temp_path and os.path.exists(temp_path):
                docs = load_pdf(temp_path)
                all_docs.extend(docs)
                # Clean up temporary file
                try:
                    os.remove(temp_path)
                except:
                    pass
        
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
# AUTHENTICATION
# ============================================================================
def verify_admin(email: str, password: str):
    """Verify admin credentials"""
    try:
        admin_emails = os.getenv("ADMIN_EMAILS", "").split(",")
        # Simple authentication - in production, use proper auth system
        return email.strip() in [e.strip() for e in admin_emails if e.strip()]
    except:
        return False

async def get_current_admin(token: str = Form(...)):
    """Dependency for admin authentication"""
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Simple token validation - in production, use proper JWT validation
    try:
        # For demo purposes, we'll use a simple validation
        # In real application, implement proper token validation
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
        return HTMLResponse(content="<h1>College Chatbot</h1><p>Frontend files not found. Please check static files setup.</p>")

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard():
    """Serve the admin dashboard"""
    try:
        with open("static/admin.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Admin Dashboard</h1><p>Admin frontend files not found.</p>")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "firebase": FIREBASE_INITIALIZED,
        "documents_loaded": vectorstore is not None,
        "active_sessions": len(user_sessions),
        "groq_api_available": bool(os.getenv("GROQ_API_KEY"))
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
        
        # Save to database
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
    
    # Create temporary file
    temp_path = None
    try:
        # Generate secure temporary file
        file_extension = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Upload to Firebase
        success, result = upload_to_firebase(temp_path, file.filename)
        
        if success:
            # Reload documents to update the knowledge base
            reload_success, reload_msg = reload_documents()
            return {
                "message": f"File uploaded successfully as {result}",
                "reload_status": reload_msg,
                "status": "success"
            }
        else:
            raise HTTPException(500, f"Upload failed: {result}")
    
    except Exception as e:
        raise HTTPException(500, f"Upload error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/api/admin/documents")
async def list_documents(admin: dict = Depends(get_current_admin)):
    """List all uploaded documents (Admin only)"""
    docs = list_firebase_documents()
    return {"documents": docs, "count": len(docs)}

@app.delete("/api/admin/documents/{filename}")
async def delete_document(filename: str, admin: dict = Depends(get_current_admin)):
    """Delete a document (Admin only)"""
    success, message = delete_from_firebase(filename)
    if success:
        # Reload documents after deletion
        reload_documents()
        return {"message": message, "status": "success"}
    else:
        raise HTTPException(500, message)

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
        # Simple authentication - in production, use Firebase Auth or similar
        if verify_admin(login_data.email, login_data.password):
            # Generate a simple token (in production, use proper JWT)
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

# ============================================================================
# STARTUP AND BACKGROUND TASKS
# ============================================================================
@app.on_event("startup")
async def startup():
    print("🚀 Starting College Admin Chatbot API...")
    print(f"📁 Firebase initialized: {FIREBASE_INITIALIZED}")
    print(f"🔑 Groq API available: {bool(os.getenv('GROQ_API_KEY'))}")
    
    if FIREBASE_INITIALIZED:
        print("🔄 Loading documents from Firebase...")
        success, msg = reload_documents()
        print(f"📚 {msg}")
    else:
        print("⚠️  Firebase not initialized - running in limited mode")
    
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