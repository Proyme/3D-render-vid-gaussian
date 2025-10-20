from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uuid
from pathlib import Path
import shutil
from gaussian_reconstruction import reconstruct_3d_gaussian
from typing import Dict
import threading

app = FastAPI(
    title="Plats 3D - Gaussian Splatting (RTX 5090)",
    description="API de reconstruction 3D ultra-rapide avec Gaussian Splatting",
    version="3.0.0"
)

# Stockage des jobs en mémoire
jobs: Dict[str, dict] = {}

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

def process_video_background(job_id: str, video_path: str, output_ply: str):
    """
    Traite la vidéo en arrière-plan
    """
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Génération en cours..."
        
        success = reconstruct_3d_gaussian(video_path, output_ply)
        
        if success and Path(output_ply).exists():
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["download_url"] = f"/download/{job_id}.ply"
            jobs[job_id]["message"] = "Modèle 3D généré avec succès"
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = "Échec de la reconstruction 3D"
        
        # Nettoyer la vidéo
        Path(video_path).unlink(missing_ok=True)
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)
        print(f"❌ Erreur job {job_id}: {e}")
        import traceback
        traceback.print_exc()

@app.post("/generate-3d")
async def generate_3d(file: UploadFile = File(...)):
    """
    Démarre la génération 3D en arrière-plan et retourne immédiatement un job_id
    """
    try:
        # Créer un job
        job_id = str(uuid.uuid4())
        video_path = UPLOAD_DIR / f"{job_id}.mp4"
        output_ply = OUTPUT_DIR / f"{job_id}.ply"
        
        # Sauvegarder la vidéo
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"🎥 Vidéo reçue: {file.filename} (Job: {job_id})")
        
        # Initialiser le job
        jobs[job_id] = {
            "status": "queued",
            "message": "En attente de traitement...",
            "download_url": None
        }
        
        # Lancer le traitement en arrière-plan
        thread = threading.Thread(
            target=process_video_background,
            args=(job_id, str(video_path), str(output_ply))
        )
        thread.daemon = True
        thread.start()
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Génération démarrée",
            "estimated_time": "2-3 minutes"
        }
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "message": "Impossible de démarrer la génération"
        }

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Vérifie l'état d'un job
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    return {
        "success": True,
        "job_id": job_id,
        **jobs[job_id]
    }

@app.get("/download/{filename}")
async def download_model(filename: str):
    """
    Télécharge un modèle 3D généré (PLY Gaussian Splatting)
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
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
