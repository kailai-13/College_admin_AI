class AdminDashboard {
    constructor() {
        this.authToken = null;
        this.initializeEventListeners();
        this.checkAuthentication();
    }

    initializeEventListeners() {
        // Login form
        document.getElementById('loginForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleLogin();
        });

        // File upload
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            this.handleFileUpload(e.dataTransfer.files);
        });

        // Document management
        document.getElementById('refreshDocs').addEventListener('click', () => {
            this.loadDocuments();
        });

        document.getElementById('reloadKnowledge').addEventListener('click', () => {
            this.reloadKnowledgeBase();
        });

        // Periodic status updates
        setInterval(() => this.updateSystemStatus(), 30000);
    }

    async handleLogin() {
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        try {
            const response = await fetch('/api/admin/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.authToken = data.token;
                localStorage.setItem('adminToken', data.token);
                this.showDashboard();
                this.loadDocuments();
                this.updateSystemStatus();
            } else {
                alert('Login failed: ' + (data.message || 'Invalid credentials'));
            }
        } catch (error) {
            console.error('Login error:', error);
            alert('Login failed. Please check your connection.');
        }
    }

    checkAuthentication() {
        const token = localStorage.getItem('adminToken');
        if (token) {
            this.authToken = token;
            this.showDashboard();
            this.loadDocuments();
            this.updateSystemStatus();
        }
    }

    showDashboard() {
        document.getElementById('loginSection').classList.add('hidden');
        document.getElementById('dashboardSection').classList.remove('hidden');
    }

    async handleFileUpload(files) {
        if (!this.authToken) {
            alert('Please login first');
            return;
        }

        const pdfFiles = Array.from(files).filter(file => 
            file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
        );

        if (pdfFiles.length === 0) {
            alert('Please select PDF files only');
            return;
        }

        for (const file of pdfFiles) {
            await this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('token', this.authToken);

        const progressBar = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const uploadProgress = document.getElementById('uploadProgress');

        uploadProgress.classList.remove('hidden');

        try {
            // Simulate progress for better UX
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 5;
                if (progress <= 90) {
                    progressBar.style.width = `${progress}%`;
                    progressText.textContent = `${progress}%`;
                }
            }, 100);

            const response = await fetch('/api/admin/upload', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();

            progressBar.style.width = '100%';
            progressText.textContent = '100%';

            setTimeout(() => {
                uploadProgress.classList.add('hidden');
                progressBar.style.width = '0%';
                progressText.textContent = '0%';
            }, 1000);

            alert(`Upload successful: ${result.message}`);
            this.loadDocuments();

        } catch (error) {
            console.error('Upload error:', error);
            uploadProgress.classList.add('hidden');
            alert('Upload failed: ' + error.message);
        }
    }

    async loadDocuments() {
        if (!this.authToken) return;

        try {
            const response = await fetch('/api/admin/documents', {
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });

            if (!response.ok) {
                throw new Error('Failed to load documents');
            }

            const data = await response.json();
            this.displayDocuments(data.documents);

        } catch (error) {
            console.error('Error loading documents:', error);
            document.getElementById('documentsList').innerHTML = 
                '<div class="loading">Error loading documents</div>';
        }
    }

    displayDocuments(documents) {
        const container = document.getElementById('documentsList');
        
        if (documents.length === 0) {
            container.innerHTML = '<div class="loading">No documents uploaded yet</div>';
            return;
        }

        container.innerHTML = documents.map(doc => `
            <div class="document-item">
                <div class="document-info">
                    <div class="document-name">${this.escapeHtml(doc.name)}</div>
                    <div class="document-meta">
                        ${this.formatFileSize(doc.size)} • 
                        ${new Date(doc.created).toLocaleDateString()}
                    </div>
                </div>
                <button class="delete-btn" onclick="adminDashboard.deleteDocument('${doc.name}')">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </div>
        `).join('');
    }

    async deleteDocument(filename) {
        if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
            return;
        }

        if (!this.authToken) return;

        try {
            const response = await fetch(`/api/admin/documents/${encodeURIComponent(filename)}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });

            if (!response.ok) {
                throw new Error('Delete failed');
            }

            const result = await response.json();
            alert(result.message);
            this.loadDocuments();

        } catch (error) {
            console.error('Delete error:', error);
            alert('Delete failed: ' + error.message);
        }
    }

    async reloadKnowledgeBase() {
        if (!this.authToken) return;

        try {
            const response = await fetch('/api/admin/reload', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });

            const result = await response.json();
            alert(result.message);
            this.updateSystemStatus();

        } catch (error) {
            console.error('Reload error:', error);
            alert('Reload failed: ' + error.message);
        }
    }

    async updateSystemStatus() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();

            // Update status indicators
            this.updateStatus('firebaseStatus', data.firebase, 'Firebase');
            this.updateStatus('documentsStatus', data.documents_loaded, 'Documents');
            this.updateStatus('modelStatus', data.documents_loaded, 'AI Model');
            this.updateStatus('sessionsStatus', true, `${data.active_sessions} active`);

        } catch (error) {
            console.error('Status update error:', error);
            // Mark all as error
            ['firebaseStatus', 'documentsStatus', 'modelStatus', 'sessionsStatus'].forEach(id => {
                this.updateStatus(id, false, 'Error');
            });
        }
    }

    updateStatus(elementId, isHealthy, text) {
        const element = document.getElementById(elementId);
        element.textContent = text;
        element.className = 'status-value ' + (isHealthy ? 'healthy' : 'error');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
}

// Initialize admin dashboard
const adminDashboard = new AdminDashboard();