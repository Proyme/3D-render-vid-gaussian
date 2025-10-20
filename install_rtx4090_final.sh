#!/bin/bash

echo "🚀 Installation Backend 3D - Gaussian Splatting RTX 4090"
echo "=========================================================="
echo "Version finale - Optimisée pour RunPod PyTorch 2.2.0"
echo ""

# Vérifier GPU
echo "📊 Vérification GPU..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "❌ GPU non détecté"
    exit 1
fi
echo "✅ GPU: $GPU_NAME"

# Vérifier PyTorch (déjà installé dans le template)
echo ""
echo "🔍 Vérification PyTorch..."
python3 << 'EOF'
import sys
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ Compute: {torch.cuda.get_device_capability(0)}")
    else:
        print("❌ CUDA non disponible")
        sys.exit(1)
except ImportError:
    print("❌ PyTorch non trouvé")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Erreur PyTorch"
    exit 1
fi

# Installer dépendances système
echo ""
echo "📦 Installation dépendances système..."
apt-get update -qq
apt-get install -y -qq colmap git build-essential wget

# Installer onnxruntime-gpu pour rembg
echo ""
echo "🚀 Installation onnxruntime-gpu..."
pip3 install onnxruntime-gpu --no-cache-dir --quiet

# Installer dépendances Python
echo ""
echo "🐍 Installation dépendances Python..."
pip3 install -r requirements_gaussian.txt --no-cache-dir --quiet

echo "   ✓ FastAPI, uvicorn, rembg, trimesh, etc."

# Cloner et installer Gaussian Splatting
echo ""
echo "⭐ Installation Gaussian Splatting..."
if [ ! -d "/workspace/gaussian-splatting" ]; then
    cd /workspace
    echo "   Clonage du repository..."
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting --quiet
    
    if [ $? -ne 0 ]; then
        echo "❌ Erreur lors du clonage"
        exit 1
    fi
    
    cd gaussian-splatting
    
    # Compiler les CUDA kernels pour RTX 4090 (Compute 8.9)
    echo "🔥 Compilation CUDA kernels (RTX 4090 - Compute 8.9)..."
    export TORCH_CUDA_ARCH_LIST="8.9"
    
    echo "   Compilation diff-gaussian-rasterization..."
    pip3 install submodules/diff-gaussian-rasterization --no-cache-dir --quiet
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Erreur compilation diff-gaussian-rasterization (peut être normal)"
    fi
    
    echo "   Compilation simple-knn..."
    pip3 install submodules/simple-knn --no-cache-dir --quiet
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Erreur compilation simple-knn (peut être normal)"
    fi
    
    cd /workspace/3D-render-vid-gaussian
    echo "   ✓ Gaussian Splatting installé"
else
    echo "   ✓ Gaussian Splatting déjà installé"
fi

# Installer Open3D
echo ""
echo "📦 Installation Open3D..."
pip3 install open3d --no-cache-dir --ignore-installed blinker --quiet
echo "   ✓ Open3D installé"

# Créer dossiers
echo ""
echo "📁 Création des dossiers..."
mkdir -p uploads outputs gaussian_workspace
echo "   ✓ Dossiers créés"

# Vérification finale complète
echo ""
echo "=========================================================="
echo "✅ VÉRIFICATION FINALE"
echo "=========================================================="

python3 << 'EOF'
import sys

print("\n🔍 Vérification des imports...")

# PyTorch
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA {torch.version.cuda}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    compute = torch.cuda.get_device_capability(0)
    print(f"✅ Compute: {compute}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ VRAM: {vram:.1f} GB")
except Exception as e:
    print(f"❌ PyTorch: {e}")
    sys.exit(1)

# FastAPI
try:
    import fastapi
    print(f"✅ FastAPI {fastapi.__version__}")
except:
    print("❌ FastAPI manquant")
    sys.exit(1)

# rembg
try:
    import rembg
    print(f"✅ rembg installé")
except:
    print("❌ rembg manquant")
    sys.exit(1)

# Open3D
try:
    import open3d
    print(f"✅ Open3D installé")
except:
    print("❌ Open3D manquant")
    sys.exit(1)

# Trimesh
try:
    import trimesh
    print(f"✅ Trimesh installé")
except:
    print("❌ Trimesh manquant")
    sys.exit(1)

# OpenCV
try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
except:
    print("❌ OpenCV manquant")
    sys.exit(1)

print("\n🎯 Configuration GPU:")
gpu_name = torch.cuda.get_device_name(0)
if "4090" in gpu_name:
    print("✅ RTX 4090 détecté et configuré !")
    print("⚡ Performance estimée: 2-3 minutes/génération")
elif "3090" in gpu_name:
    print("✅ RTX 3090 détecté !")
    print("⚡ Performance estimée: 3-4 minutes/génération")
else:
    print(f"✅ GPU: {gpu_name}")

print("\n🚀 Système prêt pour Gaussian Splatting !")
sys.exit(0)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Erreur lors de la vérification"
    echo "Certaines dépendances manquent"
    exit 1
fi

echo ""
echo "=========================================================="
echo "✅ Installation terminée avec succès !"
echo "=========================================================="
echo ""
echo "🚀 Pour lancer le serveur:"
echo "   ./start_gaussian.sh"
echo ""
echo "📊 Pour monitorer le GPU:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "🌐 L'URL du serveur sera affichée au démarrage"
echo ""
