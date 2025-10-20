#!/bin/bash

echo "🚀 Démarrage Backend 3D - Gaussian Splatting (RTX 5090)"
echo "======================================================="

# Vérifier GPU
echo "📊 Vérification GPU RTX 5090..."
nvidia-smi

# Créer les dossiers
mkdir -p uploads outputs gaussian_workspace

# Variables d'environnement
export QT_QPA_PLATFORM=offscreen
export CUDA_VISIBLE_DEVICES=0

# Lancer le serveur
echo ""
echo "🚀 Lancement du serveur Gaussian Splatting..."
echo "⚡ Performance: 1-2 minutes par génération"
echo "🔥 GPU RTX 5090 utilisé à 100%"
echo ""

python3 main_gaussian.py
