#!/bin/bash

echo "🚀 Installation Backend 3D - Gaussian Splatting RTX 4090"
echo "=========================================================="
echo "Version simplifiée - Utilise PyTorch déjà installé"
echo ""

# Vérifier GPU
echo "📊 Vérification GPU..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Vérifier PyTorch existant
echo ""
echo "✅ Vérification PyTorch existant..."
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute: {torch.cuda.get_device_capability(0)}")
EOF

# Installer onnxruntime-gpu
echo ""
echo "🚀 Installation onnxruntime-gpu..."
pip3 install onnxruntime-gpu --no-cache-dir -q

# Installer dépendances Python
echo ""
echo "🐍 Installation dépendances..."
pip3 install -r requirements_gaussian.txt --no-cache-dir -q

# Gaussian Splatting
echo ""
echo "⭐ Installation Gaussian Splatting..."
if [ ! -d "/workspace/gaussian-splatting" ]; then
    cd /workspace
    echo "   Clonage..."
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting --quiet
    cd gaussian-splatting
    
    echo "🔥 Compilation CUDA kernels (RTX 4090)..."
    export TORCH_CUDA_ARCH_LIST="8.9"
    
    pip3 install submodules/diff-gaussian-rasterization --no-cache-dir -q
    pip3 install submodules/simple-knn --no-cache-dir -q
    
    cd /workspace/3D-render-vid-gaussian
else
    echo "  ✓ Déjà installé"
fi

# Open3D
echo ""
echo "📦 Installation Open3D..."
pip3 install open3d --no-cache-dir --ignore-installed blinker -q

# Dossiers
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
print("\n🚀 Prêt pour Gaussian Splatting !")
print("⚡ Performance estimée: 2-3 minutes/génération")
EOF

echo "=========================================================="
echo ""
echo "Pour lancer: ./start_gaussian.sh"
echo ""
