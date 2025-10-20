#!/bin/bash

echo "🚀 Installation Backend 3D - Gaussian Splatting (RTX 5090)"
echo "=========================================================="

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
    colmap \
    git \
    build-essential

# Installer PyTorch Nightly avec CUDA 12.4 (support RTX 5090)
echo "🔥 Installation de PyTorch Nightly (CUDA 12.4 - RTX 5090)..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip3 install --pre \
    torch==2.7.0.dev20250226+cu124 \
    torchvision==0.22.0.dev20250226+cu124 \
    torchaudio==2.6.0.dev20250226+cu124 \
    --index-url https://download.pytorch.org/whl/nightly/cu124 \
    --no-cache-dir

# Installer onnxruntime-gpu pour rembg
echo "🚀 Installation de onnxruntime-gpu..."
pip3 install onnxruntime-gpu

# Installer les dépendances Python
echo "🐍 Installation des dépendances Python..."
pip3 install -r requirements_gaussian.txt

# Cloner et installer Gaussian Splatting
echo "⭐ Installation de Gaussian Splatting..."
if [ ! -d "/workspace/gaussian-splatting" ]; then
    cd /workspace
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
    cd gaussian-splatting
    
    # Installer dépendances
    pip3 install -r requirements.txt
    
    # Compiler les CUDA kernels
    echo "🔥 Compilation des CUDA kernels (RTX 5090)..."
    pip3 install submodules/diff-gaussian-rasterization
    pip3 install submodules/simple-knn
    
    cd /workspace/backend-3d-gaussian
else
    echo "  ✓ Gaussian Splatting déjà installé"
fi

# Installer Open3D pour conversion
echo "📦 Installation de Open3D..."
pip3 install open3d

# Créer les dossiers
echo "📁 Création des dossiers..."
mkdir -p uploads outputs gaussian_workspace

# Vérifier CUDA et GPU
echo ""
echo "✅ Vérification de CUDA et GPU..."
python3 -c "
import torch
print('CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute Capability:', torch.cuda.get_device_capability(0))
else:
    print('⚠️  CUDA non disponible')
"

echo ""
echo "✅ Installation terminée !"
echo ""
echo "🚀 Backend 3D - Gaussian Splatting (RTX 5090)"
echo "⚡ Performance estimée: 1-2 minutes par génération"
echo "🔥 GPU utilisé à 100%"
echo ""
echo "Pour lancer le serveur:"
echo "  ./start_gaussian.sh"
echo ""
