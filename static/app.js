// ─── DOM Elements ───
const apiKeyInput = document.getElementById('apiKeyInput');
const toggleApiKeyBtn = document.getElementById('toggleApiKey');
const queryInput = document.getElementById('queryInput');
const searchBtn = document.getElementById('searchBtn');
const statusSection = document.getElementById('statusSection');
const statusTimeline = document.getElementById('statusTimeline');
const resultSection = document.getElementById('resultSection');
const answerContent = document.getElementById('answerContent');
const sourcesCard = document.getElementById('sourcesCard');
const sourcesList = document.getElementById('sourcesList');
const reasoningCard = document.getElementById('reasoningCard');
const reasoningContent = document.getElementById('reasoningContent');
const reasoningToggle = document.getElementById('reasoningToggle');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// Server configuration state
let hasServerDefaultKey = false;

// Check server configuration on startup
async function checkServerConfig() {
    try {
        const res = await fetch('/api/config');
        if (res.ok) {
            const data = await res.json();
            hasServerDefaultKey = !!data.has_default_api_key;
            if (hasServerDefaultKey) {
                apiKeyInput.placeholder = "Server key active (or enter custom key)";
            } else {
                apiKeyInput.placeholder = "Enter Google AI API Key";
            }
        }
    } catch (e) {
        console.warn('Could not check server configuration:', e);
    }
}
checkServerConfig();

// ─── API Key persistence ───
const savedKey = localStorage.getItem('google_api_key');
if (savedKey) {
    apiKeyInput.value = savedKey;
}

apiKeyInput.addEventListener('change', () => {
    const val = apiKeyInput.value.trim();
    if (val) {
        localStorage.setItem('google_api_key', val);
    } else {
        localStorage.removeItem('google_api_key');
    }
});

toggleApiKeyBtn.addEventListener('click', () => {
    apiKeyInput.type = apiKeyInput.type === 'password' ? 'text' : 'password';
});

// ─── Markdown rendering ───
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true
});

// ─── Reasoning toggle ───
reasoningToggle.addEventListener('click', () => {
    reasoningToggle.classList.toggle('collapsed');
    reasoningContent.classList.toggle('collapsed');
});

// ─── Ctrl+Enter to search ───
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        searchBtn.click();
    }
});

// ─── Search ───
searchBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    const apiKey = apiKeyInput.value.trim();

    // If user didn't enter a key and the server has no default key configured
    if (!apiKey && !hasServerDefaultKey) {
        showError('Please enter your Google AI API key in the header.');
        apiKeyInput.focus();
        return;
    }
    if (!query) {
        showError('Please enter a search query.');
        queryInput.focus();
        return;
    }

    // Reset UI
    setLoading(true);
    hideAll();
    statusSection.hidden = false;
    statusTimeline.innerHTML = '';
    addStatus('Starting search agent...', true);

    try {
        const payload = {
            query: query,
            api_key: apiKey ? apiKey : null
        };

        const response = await fetch('/api/search/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || 'Search failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep the last incomplete line
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.substring(6).trim();
                    if (!dataStr) continue;
                    
                    let data;
                    try {
                        data = JSON.parse(dataStr);
                    } catch (e) {
                        // Ignore JSON parse errors for incomplete data
                        continue;
                    }

                    if (data.type === 'status') {
                        addStatus(data.message, true);
                    } else if (data.type === 'error') {
                        throw new Error(data.message);
                    } else if (data.type === 'final') {
                        displayResult(data);
                    }
                }
            }
        }
    } catch (err) {
        showError(err.message);
    } finally {
        setLoading(false);
    }
});

function setLoading(loading) {
    searchBtn.disabled = loading;
    searchBtn.querySelector('.btn-text').hidden = loading;
    searchBtn.querySelector('.btn-loading').hidden = !loading;
}

function hideAll() {
    resultSection.hidden = true;
    errorSection.hidden = true;
}

function addStatus(message, active = false) {
    // Mark previous active dots as done
    statusTimeline.querySelectorAll('.status-dot.active').forEach(dot => {
        dot.classList.remove('active');
        dot.classList.add('done');
    });

    const item = document.createElement('div');
    item.className = 'status-item';
    item.innerHTML = `
        <span class="status-dot ${active ? 'active' : 'done'}"></span>
        <span>${escapeHtml(message)}</span>
    `;
    statusTimeline.appendChild(item);
    statusTimeline.scrollTop = statusTimeline.scrollHeight;
}

function showError(msg) {
    errorSection.hidden = false;
    errorMessage.textContent = msg;
}

function displayResult(result) {
    // Finalize status
    addStatus('✓ Search complete!', false);

    // Answer
    resultSection.hidden = false;
    answerContent.innerHTML = marked.parse(result.answer || 'No answer generated.');

    // Syntax highlight code blocks
    answerContent.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });

    // Sources
    if (result.sources && result.sources.length > 0) {
        sourcesCard.hidden = false;
        sourcesList.innerHTML = result.sources.map(s => `
            <div class="source-item">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                    <polyline points="15 3 21 3 21 9"/>
                    <line x1="10" y1="14" x2="21" y2="3"/>
                </svg>
                <div>
                    <div class="source-title">${escapeHtml(s.title || 'Source')}</div>
                    <a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.url)}</a>
                </div>
            </div>
        `).join('');
    } else {
        sourcesCard.hidden = true;
    }

    // Reasoning steps
    if (result.reasoning_steps && result.reasoning_steps.length > 0) {
        reasoningCard.hidden = false;
        reasoningContent.innerHTML = result.reasoning_steps.map(step => `
            <div class="reasoning-step">${escapeHtml(step)}</div>
        `).join('');
    } else {
        reasoningCard.hidden = true;
    }
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
