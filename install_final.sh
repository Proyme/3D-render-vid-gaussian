#!/bin/bash

echo "🔥 Installation Backend 3D - Version FINALE"
echo "=========================================================="
echo "Installation avec versions PyTorch compatibles"
echo ""

# Vérifier GPU
nvidia-smi
if [ $? -ne 0 ]; then
    echo "❌ GPU non détecté"
    exit 1
fi

echo ""
echo "✅ GPU détecté !"
echo ""

# Mettre à jour apt
apt-get update

# Installer dépendances système
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

# Désinstaller tout PyTorch
echo ""
echo "🧹 Nettoyage PyTorch..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Installer PyTorch avec versions EXACTES qui matchent
echo ""
echo "🔥 Installation PyTorch Nightly (versions compatibles)..."
pip3 install \
    torch==2.7.0.dev20250226+cu124 \
    torchvision==0.22.0.dev20250226+cu124 \
    torchaudio==2.6.0.dev20250226+cu124 \
    --index-url https://download.pytorch.org/whl/nightly/cu124 \
    --no-cache-dir \
    --force-reinstall

# Vérifier
echo ""
python3 << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ Compute: {torch.cuda.get_device_capability(0)}")
EOF

if [ $? -ne 0 ]; then
    echo "❌ PyTorch échoué"
    exit 1
fi

# Installer le reste
echo ""
echo "🚀 Installation onnxruntime-gpu..."
pip3 install onnxruntime-gpu --no-cache-dir

echo ""
echo "🐍 Installation dépendances..."
pip3 install -r requirements_gaussian.txt --no-cache-dir

# Gaussian Splatting
echo ""
echo "⭐ Installation Gaussian Splatting..."
if [ ! -d "/workspace/gaussian-splatting" ]; then
    cd /workspace
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
    cd gaussian-splatting
    
    echo "🔥 Compilation CUDA kernels..."
    export TORCH_CUDA_ARCH_LIST="12.0"
    
    pip3 install submodules/diff-gaussian-rasterization --no-cache-dir
    pip3 install submodules/simple-knn --no-cache-dir
    
    cd /workspace/3D-render-vid-gaussian
fi

# Open3D
echo ""
echo "📦 Installation Open3D..."
pip3 install open3d --no-cache-dir --ignore-installed blinker

# Créer dossiers
mkdir -p uploads outputs gaussian_workspace

# Vérification finale
echo ""
echo "=========================================================="
python3 << 'EOF'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.version.cuda}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Compute: {torch.cuda.get_device_capability(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("\n🚀 Installation terminée !")
EOF

echo "=========================================================="
echo ""
echo "Pour lancer: ./start_gaussian.sh"
echo ""
