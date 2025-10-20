from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uuid
from pathlib import Path
import shutil
from gaussian_reconstruction import reconstruct_3d_gaussian

app = FastAPI(
    title="Plats 3D - Gaussian Splatting (RTX 5090)",
    description="API de reconstruction 3D ultra-rapide avec Gaussian Splatting",
    version="3.0.0"
)

# CORS
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
print("🚀 Backend 3D - Gaussian Splatting (RTX 5090)")
print("⚡ Ultra-rapide: 1-2 minutes par génération")
print("🔥 GPU RTX 5090 utilisé à 100%")
print("✅ Système prêt")

@app.get("/")
def read_root():
    import torch
    
    gpu_info = "Non disponible"
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)
    
    return {
        "message": "Backend 3D - Gaussian Splatting (RTX 5090)",
        "status": "running",
        "method": "Gaussian Splatting + AI Segmentation",
        "gpu": gpu_info,
        "gpu_optimization": "100% GPU (Segmentation + Gaussian Splatting)",
        "estimated_time": "1-2 minutes",
        "endpoints": {
            "video": "POST /generate-3d (vidéo)",
            "download": "GET /download/{filename}"
        }
    }

@app.post("/generate-3d")
async def generate_3d(file: UploadFile = File(...)):
    """
    Génère un modèle 3D ultra-rapide avec Gaussian Splatting
    RTX 5090 - 1-2 minutes
    """
    try:
        # Sauvegarder la vidéo
        video_id = str(uuid.uuid4())
        video_path = UPLOAD_DIR / f"{video_id}.mp4"
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"🎥 Vidéo reçue: {file.filename}")
        
        # Générer le modèle 3D avec Gaussian Splatting
        output_glb = OUTPUT_DIR / f"{video_id}.glb"
        
        success = reconstruct_3d_gaussian(str(video_path), str(output_glb))
        
        if not success or not output_glb.exists():
            raise HTTPException(status_code=500, detail="Échec de la reconstruction 3D")
        
        # Nettoyer
        video_path.unlink()
        
        return {
            "success": True,
            "model_url": f"/download/{video_id}.glb",
            "message": "Modèle 3D généré avec Gaussian Splatting",
            "method": "Gaussian Splatting (RTX 5090)",
            "estimated_time": "1-2 minutes"
        }
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_model(filename: str):
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
    Nettoie les fichiers temporaires
    """
    try:
        for folder in [UPLOAD_DIR, OUTPUT_DIR, Path("gaussian_workspace")]:
            if folder.exists():
                for file in folder.glob("*"):
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        shutil.rmtree(file)
        
        return {"success": True, "message": "Fichiers nettoyés"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    HOST = "0.0.0.0"
    PORT = int(os.getenv('PORT', 8000))
    
    print(f"🚀 Démarrage sur {HOST}:{PORT}")
    
    uvicorn.run(app, host=HOST, port=PORT)
