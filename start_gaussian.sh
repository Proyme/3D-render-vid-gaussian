#!/bin/bash

echo "🚀 Démarrage Backend 3D - TripoSR"
echo "=================================="

# Vérifier GPU
echo "📊 Vérification GPU..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

echo ""
echo "✅ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "✅ CUDA: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo "✅ Compute: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)"
echo "⚡ Performance estimée: 30-60 secondes/génération"

# Créer les dossiers
mkdir -p uploads outputs gaussian_workspace

# Variables d'environnement
export QT_QPA_PLATFORM=offscreen
export CUDA_VISIBLE_DEVICES=0

# Vérifier PyTorch + CUDA
echo ""
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ CUDA: {torch.version.cuda}")
    print(f"✅ Compute: {torch.cuda.get_device_capability(0)}")
    
    if "4090" in gpu_name:
        print("⚡ Performance estimée: 2-3 minutes/génération")
    elif "3090" in gpu_name:
        print("⚡ Performance estimée: 3-4 minutes/génération")
    else:
        print("⚡ Performance estimée: variable selon GPU")
else:
    print("❌ CUDA non disponible !")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Erreur: GPU non détecté"
    echo "⚠️  Redémarrez le Pod et réessayez"
    exit 1
fi

# Lancer le serveur
echo ""
echo "🚀 Lancement du serveur Gaussian Splatting..."
echo "🔥 GPU utilisé à 100%"
echo ""

python3 main_gaussian.py
