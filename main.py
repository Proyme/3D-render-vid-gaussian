from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uuid
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image as PILImage
from rembg import remove
from colmap_reconstruction import reconstruct_3d_from_video, reconstruct_3d_from_images

app = FastAPI(
    title="Plats 3D - Backend de Génération 3D",
    description="API de reconstruction 3D à partir de vidéos ou photos",
    version="1.0.0"
)

# CORS pour permettre les requêtes depuis l'app mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossiers
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configuration
print("🚀 Backend 3D - COLMAP + Segmentation AI")
print("✅ Système prêt pour reconstruction 3D")

@app.get("/")
def read_root():
    return {
        "message": "Backend 3D - Plats 3D",
        "status": "running",
        "method": "COLMAP + Segmentation AI",
        "endpoints": {
            "video": "POST /generate-3d (vidéo)",
            "multi_photos": "POST /generate-3d-multi (5 photos)",
            "download": "GET /download/{filename}"
        }
    }

@app.post("/generate-3d")
async def generate_3d(file: UploadFile = File(...)):
    """
    Génère un modèle 3D à partir d'une vidéo
    Utilise COLMAP pour reconstruction 3D dense
    """
    try:
        # Vérifier que c'est une vidéo
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Le fichier doit être une vidéo")
        
        # Générer un ID unique
        file_id = str(uuid.uuid4())
        
        # Sauvegarder la vidéo
        input_path = UPLOAD_DIR / f"{file_id}.mp4"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"🎥 Vidéo reçue: {file.filename}")
        print(f"🔄 Reconstruction 3D avec COLMAP...")
        
        output_path = OUTPUT_DIR / f"{file_id}.glb"
        success = reconstruct_3d_from_video(str(input_path), str(output_path))
        
        if not success:
            raise HTTPException(status_code=500, detail="Échec de la reconstruction 3D")
        
        print(f"✅ Modèle 3D généré: {output_path}")
        
        # Nettoyer la vidéo temporaire
        input_path.unlink()
        
        return {
            "success": True,
            "file_id": file_id,
            "download_url": f"/download/{file_id}.glb",
            "message": "Modèle 3D généré avec succès"
        }
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {str(e)}")

@app.post("/generate-3d-multi")
async def generate_3d_multi(files: list[UploadFile] = File(...)):
    """
    Génère un modèle 3D à partir de plusieurs photos (5 recommandé)
    """
    try:
        if len(files) < 3:
            raise HTTPException(status_code=400, detail="Au moins 3 photos sont nécessaires")
        
        print(f"📸 {len(files)} photos reçues")
        
        # Générer un ID unique
        file_id = str(uuid.uuid4())
        
        # Créer un dossier pour les images
        images_dir = UPLOAD_DIR / file_id
        images_dir.mkdir(exist_ok=True)
        
        # Sauvegarder toutes les images
        for i, file in enumerate(files):
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Tous les fichiers doivent être des images")
            
            image_path = images_dir / f"image_{i:04d}.jpg"
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        print(f"✓ {len(files)} images sauvegardées")
        
        # Appliquer la segmentation sur chaque image
        print("🔄 Segmentation des images...")
        for image_file in images_dir.glob("*.jpg"):
            img = PILImage.open(image_file)
            
            # Segmentation avec alpha matting pour garder plus de contexte
            output = remove(img, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10)
            
            # Créer un masque avec padding pour garder contexte
            if output.mode == 'RGBA':
                alpha = output.split()[3]
                # Dilater le masque pour garder plus de contexte
                alpha_np = np.array(alpha)
                kernel = np.ones((20, 20), np.uint8)
                alpha_dilated = cv2.dilate(alpha_np, kernel, iterations=1)
                alpha = PILImage.fromarray(alpha_dilated)
                
                # Fond blanc
                white_bg = PILImage.new('RGB', output.size, (255, 255, 255))
                white_bg.paste(output, mask=alpha)
            else:
                white_bg = output.convert('RGB')
            
            # Sauvegarder
            white_bg.save(image_file, 'JPEG', quality=95)
        
        print("✓ Segmentation terminée")
        
        # Utiliser COLMAP directement sur les images
        output_path = OUTPUT_DIR / f"{file_id}.glb"
        
        from colmap_reconstruction import reconstruct_3d_from_images
        success = reconstruct_3d_from_images(str(images_dir), str(output_path))
        
        if not success:
            raise HTTPException(status_code=500, detail="Échec de la reconstruction 3D")
        
        print(f"✅ Modèle 3D généré: {output_path}")
        
        # Nettoyer les images temporaires
        shutil.rmtree(images_dir)
        
        return {
            "success": True,
            "file_id": file_id,
            "download_url": f"/download/{file_id}.glb",
            "message": f"Modèle 3D généré avec succès à partir de {len(files)} photos"
        }
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Télécharge un modèle 3D généré
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return FileResponse(
        path=file_path,
        media_type="model/gltf-binary",
        filename=filename
    )

@app.delete("/cleanup")
async def cleanup():
    """
    Nettoie les fichiers temporaires (pour libérer de l'espace)
    """
    try:
        for folder in [UPLOAD_DIR, OUTPUT_DIR]:
            for file in folder.glob("*"):
                if file.is_file():
                    file.unlink()
        
        return {"success": True, "message": "Fichiers nettoyés"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
