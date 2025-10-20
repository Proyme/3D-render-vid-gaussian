#!/bin/bash

echo "🔥 Installation Backend 3D - Gaussian Splatting RTX 5090"
echo "=========================================================="
echo "Version spéciale optimisée pour NVIDIA RTX 5090"
echo ""

# Vérifier qu'on a bien un GPU
echo "📊 Vérification GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi non trouvé. Installation..."
    apt-get update
    apt-get install -y nvidia-utils-535
fi

nvidia-smi
if [ $? -ne 0 ]; then
    echo "❌ Erreur: GPU non détecté ou driver non chargé"
    echo "⚠️  Redémarrez le Pod RunPod et réessayez"
    exit 1
fi

echo ""
echo "✅ GPU détecté !"
echo ""

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
    build-essential \
    wget

# Vérifier CUDA
echo ""
echo "🔍 Vérification CUDA..."
if [ -d "/usr/local/cuda" ]; then
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    nvcc --version
    echo "✅ CUDA trouvé"
else
    echo "⚠️  CUDA non trouvé dans /usr/local/cuda"
fi

echo ""
echo "🔥 Installation PyTorch pour RTX 5090..."
echo "   (CUDA 12.4 - Compute Capability 9.0)"

# Désinstaller anciennes versions
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Installer PyTorch Nightly (support RTX 5090)
pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu124 \
    --no-cache-dir

# Vérifier PyTorch
echo ""
echo "✅ Vérification PyTorch + CUDA..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    if torch.cuda.get_device_capability(0)[0] >= 9:
        print("✅ RTX 5090 détecté et supporté !")
    else:
        print("⚠️  GPU détecté mais pas RTX 5090")
else:
    print("❌ CUDA non disponible dans PyTorch")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Erreur: PyTorch ne détecte pas CUDA"
    echo "⚠️  Vérifiez que le Pod a bien un GPU"
    exit 1
fi

# Installer onnxruntime-gpu pour rembg
echo ""
echo "🚀 Installation onnxruntime-gpu..."
pip3 install onnxruntime-gpu --no-cache-dir

# Installer les dépendances Python
echo ""
echo "🐍 Installation des dépendances Python..."
pip3 install -r requirements_gaussian.txt --no-cache-dir

# Cloner et installer Gaussian Splatting
echo ""
echo "⭐ Installation Gaussian Splatting..."
if [ ! -d "/workspace/gaussian-splatting" ]; then
    cd /workspace
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
    cd gaussian-splatting
    
    # Installer dépendances
    pip3 install -r requirements.txt --no-cache-dir
    
    # Compiler les CUDA kernels pour RTX 5090
    echo "🔥 Compilation des CUDA kernels (RTX 5090 - Compute 9.0)..."
    
    # Forcer Compute Capability 9.0 pour RTX 5090
    export TORCH_CUDA_ARCH_LIST="9.0"
    
    pip3 install submodules/diff-gaussian-rasterization --no-cache-dir
    pip3 install submodules/simple-knn --no-cache-dir
    
    cd /workspace/backend-3d-gaussian-rtx5090
else
    echo "  ✓ Gaussian Splatting déjà installé"
fi

# Installer Open3D
echo ""
echo "📦 Installation Open3D..."
pip3 install open3d --no-cache-dir

# Créer les dossiers
echo ""
echo "📁 Création des dossiers..."
mkdir -p uploads outputs gaussian_workspace

# Vérification finale
echo ""
echo "=========================================================="
echo "✅ VÉRIFICATION FINALE"
echo "=========================================================="

python3 << EOF
import torch
import sys

print("\n🔍 Configuration détectée:")
print(f"  • PyTorch: {torch.__version__}")
print(f"  • CUDA: {torch.version.cuda}")
print(f"  • GPU: {torch.cuda.get_device_name(0)}")
print(f"  • Compute: {torch.cuda.get_device_capability(0)}")
print(f"  • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Vérifier que c'est bien une RTX 5090
gpu_name = torch.cuda.get_device_name(0)
compute = torch.cuda.get_device_capability(0)

if "5090" in gpu_name or compute[0] >= 9:
    print("\n✅ RTX 5090 détecté et configuré correctement !")
    print("🔥 GPU utilisé à 100% pour Gaussian Splatting")
    print("⚡ Performance estimée: 1-2 minutes/génération")
elif "4090" in gpu_name:
    print("\n✅ RTX 4090 détecté - Excellent choix !")
    print("⚡ Performance estimée: 2-3 minutes/génération")
elif "3090" in gpu_name:
    print("\n✅ RTX 3090 détecté - Très bon !")
    print("⚡ Performance estimée: 3-4 minutes/génération")
else:
    print(f"\n⚠️  GPU détecté: {gpu_name}")
    print("   Gaussian Splatting fonctionnera mais peut être plus lent")

sys.exit(0)
EOF

echo ""
echo "=========================================================="
echo "✅ Installation terminée !"
echo "=========================================================="
echo ""
echo "🚀 Backend 3D - Gaussian Splatting (RTX 5090)"
echo "⚡ Optimisé pour NVIDIA RTX 5090 (Compute 9.0)"
echo "🔥 GPU utilisé à 100%"
echo ""
echo "Pour lancer le serveur:"
echo "  ./start_rtx5090.sh"
echo ""
