#!/bin/bash

echo "🔥 Installation Backend 3D - Gaussian Splatting RTX 5090"
echo "=========================================================="
echo "Version corrigée et testée"
echo ""

# Vérifier GPU
echo "📊 Vérification GPU..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "❌ GPU non détecté - Redémarrez le Pod"
    exit 1
fi

echo ""
echo "✅ GPU détecté !"
echo ""

# Mettre à jour apt
echo "🔄 Mise à jour des paquets..."
apt-get update

# Installer dépendances système (sans libgl1-mesa-glx qui n'existe plus sur Ubuntu 24.04)
echo "📦 Installation des dépendances système..."
apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libxcb-xinerama0 \
    colmap \
    git \
    build-essential \
    wget

# Installer PyTorch STABLE (pas nightly) avec CUDA 12.1 (compatible RTX 5090)
echo ""
echo "🔥 Installation PyTorch STABLE (CUDA 12.1)..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# Vérifier PyTorch
echo ""
echo "✅ Vérification PyTorch..."
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute: {torch.cuda.get_device_capability(0)}")
else:
    print("❌ CUDA non disponible")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ PyTorch ne fonctionne pas"
    exit 1
fi

# Installer onnxruntime-gpu
echo ""
echo "🚀 Installation onnxruntime-gpu..."
pip3 install onnxruntime-gpu --no-cache-dir

# Installer dépendances Python
echo ""
echo "🐍 Installation des dépendances Python..."
pip3 install -r requirements_gaussian.txt --no-cache-dir

# Cloner Gaussian Splatting
echo ""
echo "⭐ Installation Gaussian Splatting..."
if [ ! -d "/workspace/gaussian-splatting" ]; then
    cd /workspace
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
    cd gaussian-splatting
    
    # Compiler les kernels CUDA
    echo "🔥 Compilation des CUDA kernels..."
    pip3 install submodules/diff-gaussian-rasterization --no-cache-dir
    pip3 install submodules/simple-knn --no-cache-dir
    
    cd /workspace/3D-render-vid-gaussian
else
    echo "  ✓ Gaussian Splatting déjà installé"
fi

# Installer Open3D (ignorer blinker)
echo ""
echo "📦 Installation Open3D..."
pip3 install open3d --no-cache-dir --ignore-installed blinker

# Créer dossiers
mkdir -p uploads outputs gaussian_workspace

# Vérification finale
echo ""
echo "=========================================================="
echo "✅ VÉRIFICATION FINALE"
echo "=========================================================="

python3 << 'EOF'
import torch
print(f"\n✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.version.cuda}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Compute: {torch.cuda.get_device_capability(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("\n🔥 Prêt pour Gaussian Splatting !")
EOF

echo ""
echo "=========================================================="
echo "✅ Installation terminée !"
echo "=========================================================="
echo ""
echo "Pour lancer:"
echo "  ./start_gaussian.sh"
echo ""
