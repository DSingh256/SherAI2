#!/bin/bash
# VanRakshak AI - Phase 1 Quick Start Script

echo "🚀 VanRakshak AI - Phase 1 Setup & Test"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if in correct directory
if [ ! -f "backend/main.py" ]; then
    echo "✗ Error: Please run this script from project root directory"
    exit 1
fi

echo "✓ In correct directory"
echo ""

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r backend/requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check PostgreSQL
echo "🗄️  Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    echo "✓ PostgreSQL is installed"
    if psql -U postgres -c "SELECT 1" &> /dev/null; then
        echo "✓ PostgreSQL is running"
    else
        echo "⚠ PostgreSQL not running. Start with: brew services start postgresql"
    fi
else
    echo "⚠ PostgreSQL not found. Install or use Docker: docker-compose up postgres"
fi
echo ""

# Run tests
echo "🧪 Running Phase 1 Tests..."
echo "========================================"
cd backend
python -m pytest tests_phase1.py -v --tb=short

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All Phase 1 tests PASSED!"
    echo ""
    echo "📚 Next Steps:"
    echo "  1. Start backend: python -m uvicorn main:app --reload"
    echo "  2. Visit: http://localhost:8000/docs"
    echo "  3. Test upload endpoint"
    echo ""
else
    echo ""
    echo "❌ Some tests FAILED. Check output above."
    exit 1
fi
