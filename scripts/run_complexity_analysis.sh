#!/bin/bash
# Code Complexity Analysis with radon
# Run this after: pip install -r requirements.txt

set -e

echo "📊 Code Complexity Analysis with radon"
echo "======================================="
echo ""

# Install radon if not already installed
pip install -q radon 2>/dev/null || true

# 1. Cyclomatic Complexity (CC) - identify complex functions
echo "🔍 1. Cyclomatic Complexity Analysis"
echo "--------------------------------------"
echo "Legend: A=1-5 (simple), B=6-10 (ok), C=11-20 (complex), D=21-50 (very complex), F=50+ (unmaintainable)"
echo ""

# Show all files with their average complexity
echo "📁 Average Complexity per Module:"
radon cc app/ -a -s

echo ""
echo "⚠️  Functions with Complexity > 10 (C-F rating):"
radon cc app/ -nc --min C

echo ""
echo ""

# 2. Maintainability Index (MI)
echo "📊 2. Maintainability Index"
echo "--------------------------------------"
echo "Legend: A=100-20 (very maintainable), B=19-10 (maintainable), C=9-0 (needs work)"
echo ""

# Show files with low maintainability
echo "⚠️  Modules with MI < 20 (B-C rating):"
radon mi app/ -s --min B

echo ""
echo ""

# 3. Raw Metrics (LOC, LLOC, Comments)
echo "📏 3. Raw Metrics Summary"
echo "--------------------------------------"
radon raw app/ -s | head -30

echo ""
echo ""

# 4. Halstead Metrics (optional, detailed)
echo "🧮 4. Halstead Complexity (top 10 most complex)"
echo "--------------------------------------"
radon hal app/ -f | head -50

echo ""
echo ""
echo "✅ Analysis complete!"
echo ""
echo "Next steps:"
echo "1. Review functions with CC > 10 (C-F rating)"
echo "2. Review modules with MI < 20"
echo "3. Refactor complex functions to reduce CC"
echo "4. Add this check to CI pipeline"
echo ""
echo "Save results:"
echo "  ./scripts/run_complexity_analysis.sh > docs/complexity_report.txt"
