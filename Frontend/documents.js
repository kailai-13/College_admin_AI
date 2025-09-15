const searchForm = document.getElementById('search-form');
const searchResults = document.getElementById('search-results');

searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('search-input').value;

    // Call backend documents retrieval API (replace '/api/search_docs')
    fetch('/api/search_docs', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({query})
    })
    .then(res=>res.json())
    .then(data=>{
        searchResults.innerHTML = data.results.map(
            doc=>`<div><strong>${doc.title}</strong><br>${doc.content}</div>`
        ).join('');
    })
    .catch(()=>{
        searchResults.innerHTML = `<div>Error retrieving documents.</div>`;
    });
});
