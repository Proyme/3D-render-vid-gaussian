#!/bin/bash

echo "📦 Installation Backend 3D sur RunPod"
echo "======================================"

# Mettre à jour apt
echo "🔄 Mise à jour des paquets..."
apt-get update

# Installer les dépendances système
echo "📦 Installation des dépendances système..."
apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libxcb-xinerama0 \
    colmap

# Installer PyTorch avec CUDA
echo "🔥 Installation de PyTorch avec CUDA..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Installer les dépendances Python
echo "🐍 Installation des dépendances Python..."
pip3 install -r requirements.txt

# Installer onnxruntime-gpu pour rembg
echo "🚀 Installation de onnxruntime-gpu..."
pip3 install onnxruntime-gpu

# Créer les dossiers
echo "📁 Création des dossiers..."
mkdir -p uploads outputs colmap_workspace

# Vérifier CUDA
echo "✅ Vérification de CUDA..."
python3 -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun')"

echo ""
echo "✅ Installation terminée !"
echo ""
echo "Pour lancer le serveur:"
echo "  export QT_QPA_PLATFORM=offscreen"
echo "  export RUNPOD_MODE=true"
echo "  python3 main.py"
echo ""
echo "Ou utilisez:"
echo "  ./start_runpod.sh"
