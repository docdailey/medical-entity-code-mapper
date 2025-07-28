#!/bin/bash
#
# Start all Medical Entity Code Mapper TCP servers
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${GREEN}🏥 Medical Entity Code Mapper - Starting Servers${NC}"
echo "================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "   Please create it first: python -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if models are downloaded
echo -e "\n${YELLOW}Checking models...${NC}"
python scripts/download_models.py --verify-only
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Models not found! Run setup first.${NC}"
    exit 1
fi

# Function to start a server
start_server() {
    local name=$1
    local script=$2
    local port=$3
    local log_file="data/logs/${name}_server.log"
    
    # Check if already running
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  ${name} server already running on port ${port}${NC}"
        return
    fi
    
    echo -e "${GREEN}Starting ${name} server on port ${port}...${NC}"
    python $script > "$log_file" 2>&1 &
    
    # Wait a moment and check if started
    sleep 2
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${name} server started successfully${NC}"
    else
        echo -e "${RED}❌ ${name} server failed to start. Check ${log_file}${NC}"
    fi
}

# Create logs directory
mkdir -p data/logs

# Start all servers
echo -e "\n${YELLOW}Starting TCP servers...${NC}"
start_server "ICD-10" "src/servers/icd10_server.py" 8901
start_server "SNOMED" "src/servers/snomed_server.py" 8902
start_server "LOINC" "src/servers/loinc_server.py" 8903
start_server "RxNorm" "src/servers/rxnorm_server.py" 8904

# Show status
echo -e "\n${GREEN}Server Status:${NC}"
echo "=============="
echo "ICD-10:  http://localhost:8901 (TCP)"
echo "SNOMED:  http://localhost:8902 (TCP)"
echo "LOINC:   http://localhost:8903 (TCP)"
echo "RxNorm:  http://localhost:8904 (TCP)"

echo -e "\n${YELLOW}Logs location: ${PROJECT_ROOT}/data/logs/${NC}"
echo -e "${YELLOW}To stop all servers: pkill -f 'python.*server.py'${NC}"

# Keep script running to show logs
echo -e "\n${GREEN}Servers are running. Press Ctrl+C to stop all servers.${NC}"

# Trap Ctrl+C and stop all servers
trap 'echo -e "\n${YELLOW}Stopping all servers...${NC}"; pkill -f "python.*server.py"; exit' INT

# Show combined logs
tail -f data/logs/*_server.log