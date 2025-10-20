#!/bin/bash

echo "🚀 Démarrage Backend 3D sur RunPod"
echo "=================================="

# Vérifier CUDA
echo "📊 Vérification GPU..."
nvidia-smi

# Installer les dépendances si nécessaire
if [ ! -d "venv" ]; then
    echo "📦 Installation des dépendances..."
    pip3 install -r requirements.txt
fi

# Créer les dossiers nécessaires
mkdir -p uploads outputs colmap_workspace

# Variables d'environnement pour mode headless
export QT_QPA_PLATFORM=offscreen
export RUNPOD_MODE=true

# Lancer le serveur
echo "🚀 Lancement du serveur..."
python3 main.py
