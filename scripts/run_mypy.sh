#!/bin/bash
# mypy Type Check Script
# Run this after: pip install -r requirements.txt

set -e

echo "🔍 Running mypy type checks..."
echo "================================"

# Install mypy plugins if not already installed
pip install -q mypy sqlalchemy[mypy] types-redis 2>/dev/null || true

# Run mypy on app directory
echo ""
echo "📦 Checking app/ directory..."
mypy app/ --config-file pyproject.toml || {
    echo ""
    echo "⚠️  Found type errors in app/"
    echo "This is expected on first run. Let's categorize them."
}

# Run mypy on tests directory (less strict)
echo ""
echo "🧪 Checking tests/ directory..."
mypy tests/ --config-file pyproject.toml --no-strict-optional || {
    echo ""
    echo "⚠️  Found type errors in tests/"
}

echo ""
echo "✅ mypy scan complete!"
echo ""
echo "Next steps:"
echo "1. Review errors above"
echo "2. Fix critical errors (missing return types, wrong types)"
echo "3. Add '# type: ignore' comments for false positives"
echo "4. Gradually enable stricter checks"
