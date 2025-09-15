const chatBox = document.getElementById('chat-box');
const chatForm = document.getElementById('chat-form');

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userInput = document.getElementById('user-input').value;
    chatBox.innerHTML += `<div class="chat-msg user-msg">${userInput}</div>`;
    document.getElementById('user-input').value = '';

    // Call backend RAG API here (replace '/api/chat')
    fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({query: userInput})
    })
    .then(res => res.json())
    .then(data => {
        chatBox.innerHTML += `<div class="chat-msg ai-msg">${data.answer}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(() => {
        chatBox.innerHTML += `<div class="chat-msg ai-msg">Error processing your query.</div>`;
    });
});
