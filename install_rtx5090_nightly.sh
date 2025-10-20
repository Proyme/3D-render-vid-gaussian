#!/bin/bash

echo "🔥 Installation Backend 3D - Gaussian Splatting RTX 5090"
echo "=========================================================="
echo "Version PyTorch Nightly (seule compatible RTX 5090 sm_120)"
echo ""

# Vérifier GPU
echo "📊 Vérification GPU..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "❌ GPU non détecté - Redémarrez le Pod"
    exit 1
fi

echo ""
echo "✅ GPU RTX 5090 détecté !"
echo ""

# Mettre à jour apt
echo "🔄 Mise à jour des paquets..."
apt-get update

# Installer dépendances système
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

# Installer PyTorch Nightly (DERNIÈRE VERSION DISPONIBLE)
echo ""
echo "🔥 Installation PyTorch Nightly (support sm_120)..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Installer la dernière version nightly sans spécifier de version exacte
pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu124 \
    --no-cache-dir

# Vérifier PyTorch
echo ""
echo "✅ Vérification PyTorch..."
python3 << 'EOF'
import torch
import sys

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA non disponible")
    sys.exit(1)

print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

# Vérifier si RTX 5090 est supporté
compute_cap = torch.cuda.get_device_capability(0)
if compute_cap[0] >= 12:
    print("✅ RTX 5090 (sm_120) supporté par cette version de PyTorch !")
else:
    print(f"⚠️  Compute Capability: {compute_cap}")

sys.exit(0)
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
    
    # Compiler les kernels CUDA pour RTX 5090
    echo "🔥 Compilation des CUDA kernels (sm_120)..."
    
    # Forcer la compilation pour Compute Capability 12.0
    export TORCH_CUDA_ARCH_LIST="12.0"
    
    pip3 install submodules/diff-gaussian-rasterization --no-cache-dir
    pip3 install submodules/simple-knn --no-cache-dir
    
    cd /workspace/3D-render-vid-gaussian
else
    echo "  ✓ Gaussian Splatting déjà installé"
fi

# Installer Open3D
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
import sys

print(f"\n✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.version.cuda}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

compute = torch.cuda.get_device_capability(0)
print(f"✅ Compute: {compute}")

vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"✅ VRAM: {vram:.1f} GB")

if compute[0] >= 12:
    print("\n🔥 RTX 5090 (Compute 12.0) SUPPORTÉ !")
    print("🚀 Prêt pour Gaussian Splatting ultra-rapide !")
    print("⚡ Performance estimée: 1-2 minutes/génération")
else:
    print(f"\n⚠️  Compute Capability {compute} détecté")

sys.exit(0)
EOF

echo ""
echo "=========================================================="
echo "✅ Installation terminée !"
echo "=========================================================="
echo ""
echo "Pour lancer:"
echo "  ./start_gaussian.sh"
echo ""
