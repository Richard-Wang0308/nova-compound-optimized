#!/bin/bash

echo "Setting up optimized miner with caching..."

# Create cache directory
mkdir -p cache

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download database if not exists
if [ ! -f "combinatorial_db/molecules.sqlite" ]; then
    echo "Downloading molecular database..."
    mkdir -p combinatorial_db
    
    # Check if wget is available, otherwise use curl
    if command -v wget &> /dev/null; then
        wget -O combinatorial_db/molecules.sqlite \
            https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite
    elif command -v curl &> /dev/null; then
        curl -L -o combinatorial_db/molecules.sqlite \
            https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite
    else
        echo "Error: Neither wget nor curl found. Please install one of them."
        echo "Then manually download the database from:"
        echo "https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite"
        exit 1
    fi
    
    echo "Database downloaded successfully"
else
    echo "Database already exists, skipping download"
fi

# Pre-load cache (optional but recommended)
read -p "Do you want to pre-load the molecule pool cache? (Recommended, y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Pre-loading molecule pools into cache..."
    python -c "
from cache_manager import MoleculePoolCache
import sqlite3

print('Loading molecule pools...')
conn = sqlite3.connect('combinatorial_db/molecules.sqlite')
cache = MoleculePoolCache(conn)
cache.preload_all_pools()
print('Cache pre-loaded successfully!')
conn.close()
"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To run the optimized miner:"
echo "    python miner/miner_optimized.py --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY"
echo ""
echo "Cache directory: ./cache"
echo "Database location: ./combinatorial_db/molecules.sqlite"
echo ""
