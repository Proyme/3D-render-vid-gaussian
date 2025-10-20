FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Installer Python et dépendances système
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Créer répertoire de travail
WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installer dépendances Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Installer COLMAP
RUN apt-get update && apt-get install -y \
    colmap \
    && rm -rf /var/lib/apt/lists/*

# Copier le code
COPY . .

# Créer dossiers nécessaires
RUN mkdir -p uploads outputs colmap_workspace

# Exposer le port
EXPOSE 8000

# Variable d'environnement pour RunPod
ENV RUNPOD_MODE=true

# Lancer l'application
CMD ["python3", "main.py"]
