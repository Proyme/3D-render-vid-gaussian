#!/bin/bash

echo "🚀 Installation TripoSR pour RTX 4090"
echo "====================================="

# Installer les dépendances
echo "📦 Installation des dépendances..."
pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install \
    trimesh \
    rembg[gpu] \
    pillow \
    opencv-python \
    numpy \
    einops \
    omegaconf \
    transformers \
    diffusers

# Installer TripoSR
echo "📥 Installation TripoSR..."
cd /workspace
rm -rf TripoSR
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR

pip install -e .

# Télécharger le modèle (automatique au premier lancement)
echo "📥 Le modèle sera téléchargé automatiquement au premier usage"

# Vérifier l'installation
echo ""
echo "✅ Vérification de l'installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"

echo ""
echo "🎉 Installation terminée !"
echo "🚀 TripoSR est prêt à générer des meshes 3D"
