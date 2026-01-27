#!/bin/bash
# DSPy Text Classifier - Run Script
# This script uses the correct Python installation

PYTHON="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$PROJECT_DIR"

echo "=============================================="
echo "🧠 DSPy Text Classifier"
echo "=============================================="
echo ""
echo "Python: $PYTHON"
echo "Project: $PROJECT_DIR"
echo ""

# Check Python exists
if [ ! -f "$PYTHON" ]; then
    echo "❌ Python not found at $PYTHON"
    echo "   Trying system python3..."
    PYTHON="python3"
fi

# Verify imports work
echo "Checking dependencies..."
$PYTHON -c "import dspy; import flask; print('✅ Dependencies OK')" 2>&1 || {
    echo "❌ Missing dependencies. Installing..."
    $PYTHON -m pip install dspy flask pydantic
}

echo ""
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

$PYTHON run.py --port 8000
