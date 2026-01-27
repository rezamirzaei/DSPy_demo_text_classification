# 🧠 DSPy Text Classification Studio

> **Transform raw text into actionable insights using the power of Large Language Models — completely FREE with local AI!**

---

## 📖 The Story

In a world drowning in unstructured text data — customer reviews, social media posts, support tickets, news articles — businesses struggle to extract meaningful insights quickly and affordably.

**Enter DSPy Text Classification Studio.**

Built on Stanford's revolutionary [DSPy framework](https://github.com/stanfordnlp/dspy), this application brings enterprise-grade text classification to your local machine. No expensive API costs. No rate limits. No data leaving your computer.

Whether you're a startup analyzing customer feedback, a researcher categorizing documents, or a developer building intelligent applications — this tool gives you the power of GPT-class models running entirely on your hardware.

### Why DSPy?

Traditional prompt engineering is fragile. Small changes break everything. DSPy changes the game by treating LLM pipelines as **optimizable programs** rather than brittle prompt strings. Your classifiers get smarter over time, not harder to maintain.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Sentiment Analysis** | Detect positive, negative, or neutral sentiment with confidence scores |
| 📂 **Topic Classification** | Categorize text into Technology, Sports, Politics, Business & more |
| 🎯 **Intent Detection** | Understand user intent and extract key entities |
| 🖥️ **Beautiful Web UI** | Modern dark-themed interface for easy interaction |
| 🔌 **REST API** | Integrate classification into any application |
| 🏠 **100% Local** | Runs on your machine with Ollama — no cloud required |
| 💰 **Completely FREE** | No API costs, no rate limits, no surprises |

---

## 🏗️ Architecture

This application follows the **Model-View-Controller (MVC)** pattern for clean, maintainable code:

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Browser                          │
│                    http://localhost:8080                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                     VIEWS (Flask)                           │
│  • routes.py - HTTP endpoints                               │
│  • index.html - Web UI                                      │
│  • style.css / app.js - Frontend                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    CONTROLLERS                              │
│  • classification_controller.py                             │
│  • Business logic & orchestration                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      MODELS (DSPy)                          │
│  • classifier.py - DSPy Signatures & Modules                │
│  • schemas.py - Pydantic data models                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    LLM PROVIDER                             │
│  • Ollama (local) ← DEFAULT, FREE                           │
│  • Google Gemini (cloud)                                    │
│  • OpenAI (cloud)                                           │
│  • HuggingFace (cloud)                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- macOS (Intel or Apple Silicon)
- Python 3.11+
- [Homebrew](https://brew.sh) (for installing Ollama)

### Step 1: Install Ollama (FREE Local AI)

```bash
# Install Ollama
brew install ollama

# Start Ollama service
brew services start ollama

# Download a model (phi3:mini is fast & lightweight)
ollama pull phi3:mini
```

### Step 2: Install Dependencies

```bash
cd /path/to/PythonProject8
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python run.py --port 8080
```

### Step 4: Open Your Browser

Navigate to **http://localhost:8080** and start classifying!

---

## 🎮 Usage Examples

### Web Interface

1. Open http://localhost:8080
2. Select a classifier (Sentiment, Topic, or Intent)
3. Enter your text
4. Click "Classify" and see results with confidence scores

### REST API

**Sentiment Analysis:**
```bash
curl -X POST http://localhost:8080/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product exceeded all my expectations! Absolutely love it!",
    "classifier_type": "sentiment"
  }'
```

**Response:**
```json
{
  "success": true,
  "text": "This product exceeded all my expectations! Absolutely love it!",
  "classifier_type": "sentiment",
  "result": {
    "sentiment": "positive",
    "confidence": "high",
    "reasoning": "Strong positive language with words like 'exceeded', 'love', and 'absolutely'"
  }
}
```

**Topic Classification:**
```bash
curl -X POST http://localhost:8080/api/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple unveiled the new M4 chip with breakthrough AI capabilities",
    "classifier_type": "topic"
  }'
```

**Batch Classification:**
```bash
curl -X POST http://localhost:8080/api/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I hate waiting in long lines",
      "The sunset was beautiful today",
      "The meeting is scheduled for 3pm"
    ],
    "classifier_type": "sentiment"
  }'
```

---

## ⚙️ Configuration

All settings are in `.env`:

```env
# ============================================
# Choose your AI provider
# ============================================
PROVIDER=ollama              # FREE local AI (recommended)
# PROVIDER=gemini            # Google Gemini (has rate limits)
# PROVIDER=openai            # OpenAI (paid)
# PROVIDER=huggingface       # HuggingFace (free tier)

# ============================================
# Ollama Settings (FREE - No limits!)
# ============================================
OLLAMA_MODEL=phi3:mini       # Fast & lightweight
OLLAMA_BASE_URL=http://localhost:11434

# ============================================
# Cloud Provider Settings (if needed)
# ============================================
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
HF_TOKEN=your_token_here

# ============================================
# Server Settings
# ============================================
HOST=0.0.0.0
PORT=8080
DEBUG=false
```

### Recommended Ollama Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `phi3:mini` | 2.2 GB | ⚡ Fast | Good | Quick classification |
| `llama3.2` | 4.7 GB | Medium | Better | Balanced performance |
| `mistral` | 4.1 GB | Medium | Better | General purpose |
| `llama3.1:8b` | 8 GB | Slower | Best | Highest accuracy |

---

## 🧠 How DSPy Works

### Traditional Prompting (Fragile)
```python
prompt = f"Classify the sentiment of: {text}\nAnswer: positive, negative, or neutral"
# 😰 Breaks with edge cases, hard to improve
```

### DSPy Approach (Robust)
```python
class SentimentSignature(dspy.Signature):
    """Classify the sentiment of text as positive, negative, or neutral."""
    text: str = dspy.InputField(desc="The text to analyze")
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")
    confidence: str = dspy.OutputField(desc="high, medium, or low")
    reasoning: str = dspy.OutputField(desc="Brief explanation")

class SentimentClassifier(dspy.Module):
    def __init__(self):
        self.classifier = dspy.ChainOfThought(SentimentSignature)
    
    def forward(self, text: str):
        return self.classifier(text=text)
```

**Benefits:**
- ✅ Structured inputs and outputs
- ✅ Automatic prompt optimization
- ✅ Chain-of-thought reasoning
- ✅ Easy to extend and modify

---

## 📁 Project Structure

```
PythonProject8/
├── app/                              # MVC Application
│   ├── models/                       # M - Data & DSPy
│   │   ├── classifier.py             # DSPy Signatures & Modules
│   │   └── schemas.py                # Pydantic models
│   ├── views/                        # V - Presentation
│   │   ├── routes.py                 # Flask endpoints
│   │   └── __init__.py
│   ├── controllers/                  # C - Business Logic
│   │   └── classification_controller.py
│   ├── templates/
│   │   └── index.html                # Web UI
│   └── static/
│       ├── css/style.css             # Styling
│       └── js/app.js                 # Frontend logic
├── run.py                            # 🚀 Entry point
├── quick_test.py                     # ✅ Verify setup
├── diagnose.py                       # 🔍 Debug providers
├── requirements.txt                  # Dependencies
├── Dockerfile                        # Container build
├── docker-compose.yml                # Container orchestration
└── .env                              # Configuration
```

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8080
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check & status |
| `/api/classify` | POST | Classify single text |
| `/api/classify/batch` | POST | Classify multiple texts |
| `/api/classifiers` | GET | List available classifiers |

---

## 🛠️ Troubleshooting

### "Ollama not running"
```bash
brew services start ollama
ollama list  # Verify models are installed
```

### "No models found"
```bash
ollama pull phi3:mini
```

### "Rate limit exceeded" (Gemini)
Switch to Ollama in `.env`:
```env
PROVIDER=ollama
```

### Check system status
```bash
python diagnose.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - Use freely for personal and commercial projects.

---

## 🙏 Acknowledgments

- [DSPy](https://github.com/stanfordnlp/dspy) - Stanford NLP's revolutionary framework
- [Ollama](https://ollama.ai) - Making local AI accessible
- [Flask](https://flask.palletsprojects.com) - Lightweight Python web framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM interface

---

<div align="center">

**Built with ❤️ using DSPy**

[Report Bug](../../issues) · [Request Feature](../../issues)

</div>
