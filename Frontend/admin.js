const addDocForm = document.getElementById('add-doc-form');
const adminStatus = document.getElementById('admin-status');

addDocForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const title = document.getElementById('doc-title').value;
    const content = document.getElementById('doc-content').value;

    // Backend API to add doc (replace '/api/add_doc')
    fetch('/api/add_doc', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({title, content})
    })
    .then(res=>res.json())
    .then(data=>{
        adminStatus.textContent = data.message || "Document added.";
        addDocForm.reset();
    })
    .catch(()=>{
        adminStatus.textContent = "Error adding document.";
    });
});  
