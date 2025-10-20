#!/bin/bash

echo "🚀 Installation COLMAP pour RTX 4090"
echo "====================================="

# Installer COLMAP depuis les dépôts Ubuntu (version stable)
echo "📦 Installation COLMAP..."
apt-get update
apt-get install -y colmap

# Vérifier l'installation
echo ""
echo "✅ Vérification de l'installation..."
colmap -h > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ COLMAP installé avec succès !"
    colmap --version
else
    echo "❌ Erreur lors de l'installation de COLMAP"
    exit 1
fi

# Installer les dépendances Python
echo ""
echo "📦 Installation des dépendances Python..."
pip install opencv-python pillow rembg[gpu] open3d trimesh numpy torch

echo ""
echo "🎉 Installation terminée !"
echo "⚠️  Note: COLMAP MVS (dense) nécessite CUDA, mais le fallback sparse fonctionne"
echo "🚀 Vous pouvez maintenant lancer le serveur"
