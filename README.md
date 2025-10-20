# 🎨 Backend 3D - Génération de Modèles 3D

Backend Python pour générer des modèles 3D à partir de photos de plats.

## 🚀 Installation

### 1. Créer un environnement virtuel

```bash
cd "D:\application plats en 3d\backend-3d"
python -m venv venv
venv\Scripts\activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Installer Wonder3D (optionnel, pour la vraie génération 3D)

```bash
# Cloner le repo Wonder3D
git clone https://github.com/xxlong0/Wonder3D.git
cd Wonder3D

# Installer
pip install -e .
```

## 🎯 Démarrage

```bash
# Activer l'environnement virtuel
venv\Scripts\activate

# Lancer le serveur
python main.py
```

Le serveur démarre sur **http://localhost:8000**

## 📡 API Endpoints

### POST /generate-3d
Génère un modèle 3D à partir d'une photo

**Request:**
```bash
curl -X POST "http://localhost:8000/generate-3d" \
  -F "file=@plat.jpg"
```

**Response:**
```json
{
  "success": true,
  "file_id": "uuid-123",
  "download_url": "/download/uuid-123.glb",
  "message": "Modèle 3D généré avec succès"
}
```

### GET /download/{filename}
Télécharge un modèle 3D généré

**Request:**
```bash
curl "http://localhost:8000/download/uuid-123.glb" -O
```

### DELETE /cleanup
Nettoie les fichiers temporaires

## 🔧 Configuration

### GPU NVIDIA
Le backend détecte automatiquement si un GPU NVIDIA est disponible.

Pour vérifier :
```python
import torch
print(torch.cuda.is_available())  # True si GPU disponible
print(torch.cuda.get_device_name(0))  # Nom du GPU
```

### CPU Only
Si pas de GPU, le backend fonctionne sur CPU (plus lent).

## 📊 Performance

| Hardware | Temps par plat | Qualité |
|----------|----------------|---------|
| RTX 3060 | ~2 minutes | ⭐⭐⭐⭐⭐ |
| GTX 1060 | ~5 minutes | ⭐⭐⭐⭐ |
| CPU | ~15 minutes | ⭐⭐⭐ |

## 🎨 Intégration avec l'App Mobile

L'app mobile envoie la photo au backend :

```javascript
const formData = new FormData();
formData.append('file', {
  uri: photoUri,
  type: 'image/jpeg',
  name: 'plat.jpg',
});

const response = await fetch('http://YOUR_IP:8000/generate-3d', {
  method: 'POST',
  body: formData,
});

const data = await response.json();
console.log('Modèle 3D:', data.download_url);
```

## 🔄 Workflow Complet

```
1. Restaurateur prend une photo (App Mobile)
   ↓
2. Photo envoyée au Backend 3D (FastAPI)
   ↓
3. Wonder3D génère le modèle 3D
   ↓
4. Modèle .glb sauvegardé
   ↓
5. URL retournée à l'app
   ↓
6. Upload sur Firebase Storage
   ↓
7. Client voit le modèle 3D (Three.js)
```

## 🐛 Dépannage

### Erreur CUDA
```bash
# Vérifier CUDA
nvidia-smi

# Réinstaller PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Mémoire insuffisante
Réduire la résolution des images dans `main.py` :
```python
image = image.resize((256, 256))  # Au lieu de 512x512
```

## 📝 TODO

- [ ] Intégrer le vrai Wonder3D
- [ ] Optimiser les textures
- [ ] Ajouter un cache pour les modèles
- [ ] Support des vidéos 360°
- [ ] Compression des modèles .glb
