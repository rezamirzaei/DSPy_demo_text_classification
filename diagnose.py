#!/usr/bin/env python3
"""
Diagnostic script to check provider availability and fix configuration.
"""
import urllib.request
import json
import os
import sys

def load_env():
    """Load .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

def check_ollama():
    """Check if Ollama is running"""
    try:
        response = urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)
        data = json.loads(response.read())
        models = [m['name'] for m in data.get('models', [])]
        return True, models
    except Exception as e:
        return False, str(e)

def check_gemini():
    """Check Gemini API key"""
    api_key = os.environ.get('GOOGLE_API_KEY', '')
    return bool(api_key), api_key

def main():
    load_env()

    print("=" * 50)
    print("DSPy Provider Diagnostic")
    print("=" * 50)
    print()

    # Check Ollama
    print("1. Checking Ollama (FREE local)...")
    ollama_ok, ollama_info = check_ollama()
    if ollama_ok:
        if ollama_info:
            print(f"   ✅ Ollama running with models: {ollama_info}")
            print("   → Recommended: Use PROVIDER=ollama")
        else:
            print("   ⚠️  Ollama running but no models installed")
            print("   → Run: ollama pull llama3.2")
    else:
        print(f"   ❌ Ollama not available: {ollama_info}")
        print("   → Install from: https://ollama.ai")

    print()

    # Check Gemini
    print("2. Checking Google Gemini...")
    gemini_ok, gemini_key = check_gemini()
    if gemini_ok:
        print(f"   ✅ API key found: {gemini_key[:15]}...")
        print("   ⚠️  Note: Free tier has rate limits")
    else:
        print("   ❌ No GOOGLE_API_KEY in .env")

    print()
    print("=" * 50)

    # Determine best provider
    current_provider = os.environ.get('PROVIDER', 'ollama')
    print(f"Current PROVIDER setting: {current_provider}")
    print()

    if ollama_ok and ollama_info:
        print("✅ RECOMMENDATION: Use Ollama (FREE, no rate limits)")
        print("   Set PROVIDER=ollama in .env")
        return "ollama"
    elif gemini_ok:
        print("✅ RECOMMENDATION: Use Gemini (has rate limits)")
        print("   Set PROVIDER=gemini in .env")
        return "gemini"
    else:
        print("❌ No working provider found!")
        print()
        print("To fix, choose one option:")
        print()
        print("Option A - Install Ollama (FREE):")
        print("   1. Download from https://ollama.ai")
        print("   2. Run: ollama pull llama3.2")
        print("   3. Set PROVIDER=ollama in .env")
        print()
        print("Option B - Use Gemini:")
        print("   1. Get API key from https://aistudio.google.com/app/apikey")
        print("   2. Set GOOGLE_API_KEY=your_key in .env")
        print("   3. Set PROVIDER=gemini in .env")
        return None

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
