class ChatInterface {
    constructor() {
        this.userId = this.generateUserId();
        this.isLoading = false;
        this.initializeEventListeners();
        this.checkHealth();
    }

    generateUserId() {
        return 'user_' + Math.random().toString(36).substr(2, 9);
    }

    initializeEventListeners() {
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const charCount = document.getElementById('charCount');

        // Send message on button click
        sendButton.addEventListener('click', () => this.sendMessage());

        // Send message on Enter key
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Character count
        messageInput.addEventListener('input', () => {
            const length = messageInput.value.length;
            charCount.textContent = `${length}/500`;
            
            if (length > 500) {
                charCount.style.color = '#e53e3e';
            } else {
                charCount.style.color = '#718096';
            }
        });

        // Quick question buttons
        document.querySelectorAll('.quick-btn').forEach(button => {
            button.addEventListener('click', () => {
                const question = button.getAttribute('data-question');
                messageInput.value = question;
                this.sendMessage();
            });
        });

        // Auto-focus input
        messageInput.focus();
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();

        if (!message || this.isLoading) return;

        // Clear input
        messageInput.value = '';
        document.getElementById('charCount').textContent = '0/500';

        // Add user message to chat
        this.addMessage('user', message);

        // Show loading indicator
        this.setLoading(true);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    user_id: this.userId,
                    user_type: 'student'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.addMessage('bot', data.response);

        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('bot', 'Sorry, I encountered an error. Please try again or contact support.');
        } finally {
            this.setLoading(false);
        }
    }

    addMessage(sender, content) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        
        const avatarIcon = document.createElement('i');
        avatarIcon.className = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
        avatar.appendChild(avatarIcon);

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('p');
        messageText.textContent = content;
        messageContent.appendChild(messageText);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    setLoading(loading) {
        this.isLoading = loading;
        const sendButton = document.getElementById('sendButton');
        const messageInput = document.getElementById('messageInput');

        if (loading) {
            sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            sendButton.disabled = true;
            messageInput.disabled = true;
        } else {
            sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
            sendButton.disabled = false;
            messageInput.disabled = false;
            messageInput.focus();
        }
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            const statusIndicator = document.getElementById('statusIndicator');
            const statusDot = statusIndicator.querySelector('.status-dot');
            const statusText = statusIndicator.querySelector('.status-text');

            if (data.documents_loaded) {
                statusDot.className = 'status-dot';
                statusText.textContent = 'Online';
            } else {
                statusDot.className = 'status-dot offline';
                statusText.textContent = 'Learning...';
            }
        } catch (error) {
            console.error('Health check failed:', error);
            const statusIndicator = document.getElementById('statusIndicator');
            statusIndicator.querySelector('.status-dot').className = 'status-dot offline';
            statusIndicator.querySelector('.status-text').textContent = 'Offline';
        }
    }
}

// Initialize chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatInterface();
});