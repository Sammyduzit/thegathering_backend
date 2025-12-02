#!/bin/bash
set -e  # Exit on any error

echo "🚀 Starting deployment..."

# Configuration
PROJECT_DIR="/home/ubuntu/the_gathering"
BRANCH="main"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Backup current version
echo "📦 Creating backup..."
BACKUP_DIR="/home/ubuntu/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r "$PROJECT_DIR" "$BACKUP_DIR/" || echo "⚠️  Backup failed (non-critical)"

# Fetch latest changes
echo "📥 Fetching latest code..."
git fetch origin

# Check if there are changes
CURRENT_COMMIT=$(git rev-parse HEAD)
LATEST_COMMIT=$(git rev-parse origin/$BRANCH)

if [ "$CURRENT_COMMIT" == "$LATEST_COMMIT" ]; then
    echo "✅ Already up to date!"
    exit 0
fi

echo "📝 Updating from $CURRENT_COMMIT to $LATEST_COMMIT"

# Pull latest code
git pull origin $BRANCH

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run database migrations (if needed)
# echo "🗄️  Running database migrations..."
# alembic upgrade head

# Restart services
echo "🔄 Restarting services..."
sudo systemctl restart gathering-api
sudo systemctl restart gathering-worker

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 5

# Health check
echo "🏥 Running health check..."
MAX_RETRIES=5
RETRY_COUNT=0
HEALTH_URL="http://localhost:8000/health"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f -s "$HEALTH_URL" > /dev/null; then
        echo -e "${GREEN}✅ Health check passed!${NC}"
        echo "🎉 Deployment successful!"

        # Show service status
        echo ""
        echo "📊 Service Status:"
        sudo systemctl status gathering-api --no-pager -l | head -n 5
        sudo systemctl status gathering-worker --no-pager -l | head -n 5

        exit 0
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "⏳ Health check attempt $RETRY_COUNT/$MAX_RETRIES..."
    sleep 3
done

# Health check failed - rollback
echo -e "${RED}❌ Health check failed!${NC}"
echo "🔙 Rolling back to previous version..."

git reset --hard "$CURRENT_COMMIT"
sudo systemctl restart gathering-api
sudo systemctl restart gathering-worker

echo -e "${RED}❌ Deployment failed and rolled back!${NC}"
exit 1
