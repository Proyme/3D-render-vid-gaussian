#!/bin/bash

echo "🔧 Installation COLMAP avec support CUDA"
echo "=========================================="

# Désinstaller COLMAP actuel
echo "🗑️  Désinstallation COLMAP existant..."
apt-get remove -y colmap 2>/dev/null || true

# Installer les dépendances
echo "📦 Installation des dépendances..."
apt-get update
apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

# Cloner COLMAP
echo "📥 Clonage COLMAP..."
cd /workspace
rm -rf colmap
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout 3.8

# Compiler avec CUDA
echo "🔨 Compilation COLMAP avec CUDA (RTX 4090)..."
echo "⏱️  Cela prendra ~15-20 minutes..."
mkdir -p build
cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ENABLED=ON \
    -DGUI_ENABLED=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local

make -j$(nproc)
make install

# Vérifier l'installation
echo ""
echo "✅ Vérification de l'installation..."
colmap -h > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ COLMAP avec CUDA installé avec succès !"
    colmap --version
else
    echo "❌ Erreur lors de l'installation de COLMAP"
    exit 1
fi

echo ""
echo "🎉 Installation terminée !"
echo "🚀 Vous pouvez maintenant relancer le serveur"
