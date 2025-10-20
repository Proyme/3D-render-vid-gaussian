#!/bin/bash

echo "🔥 Démarrage Backend 3D - Gaussian Splatting RTX 5090"
echo "======================================================"

# Vérifier GPU
echo "📊 Vérification GPU RTX 5090..."
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# Créer les dossiers
mkdir -p uploads outputs gaussian_workspace

# Variables d'environnement
export QT_QPA_PLATFORM=offscreen
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="9.0"

# Vérifier PyTorch + CUDA
echo ""
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA: {torch.version.cuda}")
    print(f"✅ Compute: {torch.cuda.get_device_capability(0)}")
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
echo "⚡ Performance RTX 5090: 1-2 minutes par génération"
echo "🔥 GPU utilisé à 100%"
echo ""

python3 main_gaussian.py
