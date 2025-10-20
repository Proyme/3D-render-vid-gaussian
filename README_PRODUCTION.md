# 🍽️ Backend 3D - Plats 3D

Backend de reconstruction 3D optimisé pour scanner des plats et objets.

## 🎯 Fonctionnalités

- ✅ **Reconstruction 3D à partir de vidéo** (10-15 secondes)
- ✅ **Reconstruction 3D à partir de 5 photos** (mode guidé)
- ✅ **Segmentation automatique** avec AI (rembg)
- ✅ **COLMAP** pour reconstruction dense
- ✅ **Poisson Surface Reconstruction** pour mesh solide
- ✅ **Post-traitement** (nettoyage, lissage, optimisation)

## 📦 Installation

### Prérequis
- Python 3.10+
- COLMAP installé dans `./COLMAP/`

### Installation des dépendances

```powershell
# Créer l'environnement virtuel
python -m venv venv

# Activer l'environnement
.\venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Démarrage

```powershell
# Activer l'environnement
.\venv\Scripts\activate

# Lancer le serveur
python main.py
```

Le serveur démarre sur `http://0.0.0.0:8000`

## 📡 API Endpoints

### 1. Mode Vidéo
**POST** `/generate-3d`

Upload une vidéo (10-15 sec) pour générer un modèle 3D.

```bash
curl -X POST http://localhost:8000/generate-3d \
  -F "file=@video.mp4"
```

**Réponse:**
```json
{
  "success": true,
  "file_id": "uuid",
  "download_url": "/download/uuid.glb",
  "message": "Modèle 3D généré avec succès"
}
```

### 2. Mode 5 Photos
**POST** `/generate-3d-multi`

Upload 5 photos (différents angles) pour générer un modèle 3D.

```bash
curl -X POST http://localhost:8000/generate-3d-multi \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "files=@photo3.jpg" \
  -F "files=@photo4.jpg" \
  -F "files=@photo5.jpg"
```

### 3. Téléchargement
**GET** `/download/{filename}`

Télécharge le modèle 3D généré.

```bash
curl -O http://localhost:8000/download/uuid.glb
```

## ⚙️ Configuration

### Paramètres COLMAP
- **Frames extraites (vidéo):** 12 (sélection intelligente par netteté)
- **Reconstruction:** Dense MVS avec PatchMatch
- **Profondeur Poisson:** 10 (haute qualité)

### Segmentation
- **Alpha matting:** Activé
- **Dilatation du masque:** 20px (garde contexte pour COLMAP)

### Post-traitement
- **Isolation:** Plus grand cluster uniquement
- **Cropping:** 60% central (élimine table/arrière-plan)
- **Lissage:** 5 + 2 itérations
- **Simplification:** 80% des triangles

## 📊 Performances

| Mode | Temps | Qualité | Recommandé pour |
|------|-------|---------|-----------------|
| Vidéo (12 frames) | ~10 min | ⭐⭐⭐⭐ | Capture rapide |
| 5 Photos | ~5-6 min | ⭐⭐⭐⭐⭐ | Meilleure qualité |

## 🎨 Workflow Complet

### Mode Vidéo
1. Upload vidéo (10-15 sec)
2. Extraction de 12 meilleures frames
3. Segmentation automatique
4. COLMAP reconstruction (SfM + MVS)
5. Poisson Surface Reconstruction
6. Post-traitement et optimisation
7. Export GLB

### Mode 5 Photos
1. Upload 5 photos (angles variés)
2. Segmentation automatique
3. COLMAP reconstruction directe
4. Poisson + optimisation
5. Export GLB

## 📝 Recommandations pour les Utilisateurs

### Pour de Meilleurs Résultats

**Vidéo:**
- ✅ Filmer en tournant autour de l'objet (360°)
- ✅ Garder une distance constante (30-50cm)
- ✅ Mouvement fluide et lent
- ✅ Bon éclairage uniforme
- ✅ Durée: 10-15 secondes

**Photos (Mode 5):**
- ✅ Photo 1: Face avant
- ✅ Photo 2: Face arrière (180°)
- ✅ Photo 3: Côté gauche
- ✅ Photo 4: Côté droit
- ✅ Photo 5: Vue du dessus
- ✅ Distance constante pour toutes les photos
- ✅ Bon éclairage

## 🔧 Dépannage

### COLMAP échoue
- Vérifier que COLMAP est installé dans `./COLMAP/`
- S'assurer que les images ont assez de features communes
- Augmenter le nombre de photos/frames

### Segmentation trop agressive
- Ajuster `alpha_matting_foreground_threshold` (ligne 130 dans main.py)
- Augmenter la dilatation du masque

### Modèle 3D incomplet
- Prendre plus de photos/vidéo plus longue
- Améliorer l'éclairage
- S'assurer de couvrir tous les angles

## 📁 Structure

```
backend-3d/
├── main.py                    # API FastAPI
├── colmap_reconstruction.py   # Pipeline COLMAP
├── requirements.txt           # Dépendances Python
├── COLMAP/                    # Binaires COLMAP
├── uploads/                   # Fichiers temporaires
└── outputs/                   # Modèles 3D générés
```

## 🏆 Production Ready

Ce backend est optimisé pour:
- ✅ Performance (temps de traitement minimisé)
- ✅ Qualité (reconstruction dense + post-traitement)
- ✅ Robustesse (gestion d'erreurs complète)
- ✅ Scalabilité (prêt pour déploiement cloud)

## 📄 Licence

MIT License - Voir LICENSE pour plus de détails
