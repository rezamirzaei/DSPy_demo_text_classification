/**
 * DSPy Text Classifier - Frontend JavaScript
 */

// State
let currentClassifier = 'sentiment';
let history = [];

// DOM Elements
const inputText = document.getElementById('input-text');
const classifyBtn = document.getElementById('classify-btn');
const resultsSection = document.getElementById('results-section');
const loading = document.getElementById('loading');
const resultsContent = document.getElementById('results-content');
const errorDisplay = document.getElementById('error-display');
const errorMessage = document.getElementById('error-message');
const historyList = document.getElementById('history-list');
const clearHistoryBtn = document.getElementById('clear-history');
const customOptions = document.getElementById('custom-options');
const topicOptions = document.getElementById('topic-options');
const intentOptions = document.getElementById('intent-options');

// Classifier tabs
const tabButtons = document.querySelectorAll('.tab-btn');
const resultTypes = {
    sentiment: document.getElementById('sentiment-results'),
    topic: document.getElementById('topic-results'),
    intent: document.getElementById('intent-results')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => switchClassifier(btn.dataset.classifier));
    });

    // Classify button
    classifyBtn.addEventListener('click', classify);

    // Enter key in textarea
    inputText.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            classify();
        }
    });

    // Example buttons
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            inputText.value = btn.dataset.text;
            switchClassifier(btn.dataset.type);
            classify();
        });
    });

    // Clear history
    clearHistoryBtn.addEventListener('click', clearHistory);

    // Load history from localStorage
    loadHistory();
});

/**
 * Switch active classifier
 */
function switchClassifier(type) {
    currentClassifier = type;

    // Update tabs
    tabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.classifier === type);
    });

    // Show/hide custom options
    topicOptions.classList.toggle('hidden', type !== 'topic');
    intentOptions.classList.toggle('hidden', type !== 'intent');
    customOptions.classList.toggle('hidden', type === 'sentiment');

    // Hide all result types
    Object.values(resultTypes).forEach(el => el.classList.add('hidden'));
}

/**
 * Perform classification
 */
async function classify() {
    const text = inputText.value.trim();

    if (!text) {
        showError('Please enter some text to classify.');
        return;
    }

    // Prepare request
    const payload = {
        text: text,
        classifier_type: currentClassifier
    };

    // Add custom options if applicable
    if (currentClassifier === 'topic') {
        const customCats = document.getElementById('custom-categories').value.trim();
        if (customCats) {
            payload.categories = customCats.split(',').map(c => c.trim()).filter(c => c);
        }
    } else if (currentClassifier === 'intent') {
        const customIntents = document.getElementById('custom-intents').value.trim();
        if (customIntents) {
            payload.intents = customIntents.split(',').map(i => i.trim()).filter(i => i);
        }
    }

    // Show loading
    resultsSection.classList.remove('hidden');
    loading.classList.remove('hidden');
    resultsContent.classList.add('hidden');
    errorDisplay.classList.add('hidden');
    classifyBtn.disabled = true;

    try {
        const response = await fetch('/api/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
            addToHistory(data);
        } else {
            showError(data.error || 'Classification failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        loading.classList.add('hidden');
        resultsContent.classList.remove('hidden');
        classifyBtn.disabled = false;
    }
}

/**
 * Display classification results
 */
function displayResults(data) {
    const type = data.classifier_type;
    const result = data.result;

    // Hide all result types first
    Object.values(resultTypes).forEach(el => el.classList.add('hidden'));
    errorDisplay.classList.add('hidden');

    if (type === 'sentiment') {
        displaySentimentResults(result);
    } else if (type === 'topic') {
        displayTopicResults(result);
    } else if (type === 'intent') {
        displayIntentResults(result);
    }
}

/**
 * Display sentiment results
 */
function displaySentimentResults(result) {
    const container = resultTypes.sentiment;
    container.classList.remove('hidden');

    const valueEl = document.getElementById('sentiment-value');
    valueEl.textContent = result.sentiment;
    valueEl.className = 'result-value sentiment-' + result.sentiment;

    const confidenceEl = document.getElementById('sentiment-confidence');
    confidenceEl.textContent = result.confidence;
    confidenceEl.className = 'confidence-badge confidence-' + result.confidence;

    document.getElementById('sentiment-reasoning').textContent = result.reasoning;
}

/**
 * Display topic results
 */
function displayTopicResults(result) {
    const container = resultTypes.topic;
    container.classList.remove('hidden');

    document.getElementById('topic-value').textContent = result.topic;

    const confidenceEl = document.getElementById('topic-confidence');
    confidenceEl.textContent = result.confidence;
    confidenceEl.className = 'confidence-badge confidence-' + result.confidence;

    // Display categories
    const categoriesEl = document.getElementById('topic-categories');
    categoriesEl.innerHTML = '';
    if (result.available_categories) {
        result.available_categories.forEach(cat => {
            const tag = document.createElement('span');
            tag.className = 'tag' + (cat === result.topic ? ' active' : '');
            tag.textContent = cat;
            categoriesEl.appendChild(tag);
        });
    }

    document.getElementById('topic-reasoning').textContent = result.reasoning;
}

/**
 * Display intent results
 */
function displayIntentResults(result) {
    const container = resultTypes.intent;
    container.classList.remove('hidden');

    document.getElementById('intent-value').textContent = result.intent;

    const confidenceEl = document.getElementById('intent-confidence');
    confidenceEl.textContent = result.confidence;
    confidenceEl.className = 'confidence-badge confidence-' + result.confidence;

    // Format entities
    let entitiesText = result.entities;
    try {
        // Try to parse and pretty-print JSON
        const parsed = JSON.parse(result.entities);
        entitiesText = JSON.stringify(parsed, null, 2);
    } catch (e) {
        // Keep original text if not valid JSON
    }
    document.getElementById('intent-entities').textContent = entitiesText;

    document.getElementById('intent-reasoning').textContent = result.reasoning;
}

/**
 * Show error message
 */
function showError(message) {
    resultsSection.classList.remove('hidden');
    loading.classList.add('hidden');
    resultsContent.classList.remove('hidden');

    Object.values(resultTypes).forEach(el => el.classList.add('hidden'));

    errorDisplay.classList.remove('hidden');
    errorMessage.textContent = message;
}

/**
 * Add classification to history
 */
function addToHistory(data) {
    const historyItem = {
        text: data.text,
        type: data.classifier_type,
        result: data.result,
        timestamp: new Date().toISOString()
    };

    history.unshift(historyItem);

    // Keep only last 20 items
    if (history.length > 20) {
        history = history.slice(0, 20);
    }

    saveHistory();
    renderHistory();
}

/**
 * Render history list
 */
function renderHistory() {
    if (history.length === 0) {
        historyList.innerHTML = '<p class="empty-history">No classifications yet. Try classifying some text!</p>';
        return;
    }

    historyList.innerHTML = history.map(item => {
        let resultValue = '';
        let resultClass = '';

        if (item.type === 'sentiment') {
            resultValue = item.result.sentiment;
            resultClass = 'sentiment-' + resultValue;
        } else if (item.type === 'topic') {
            resultValue = item.result.topic;
            resultClass = 'topic-value';
        } else if (item.type === 'intent') {
            resultValue = item.result.intent;
            resultClass = 'intent-value';
        }

        return `
            <div class="history-item">
                <span class="history-text" title="${escapeHtml(item.text)}">${escapeHtml(truncate(item.text, 50))}</span>
                <div class="history-result">
                    <span class="history-type">${item.type}</span>
                    <span class="history-value ${resultClass}">${escapeHtml(resultValue)}</span>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Clear history
 */
function clearHistory() {
    history = [];
    saveHistory();
    renderHistory();
}

/**
 * Save history to localStorage
 */
function saveHistory() {
    try {
        localStorage.setItem('classification_history', JSON.stringify(history));
    } catch (e) {
        console.error('Failed to save history:', e);
    }
}

/**
 * Load history from localStorage
 */
function loadHistory() {
    try {
        const saved = localStorage.getItem('classification_history');
        if (saved) {
            history = JSON.parse(saved);
            renderHistory();
        }
    } catch (e) {
        console.error('Failed to load history:', e);
        history = [];
    }
}

/**
 * Utility: Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Utility: Truncate text
 */
function truncate(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}
