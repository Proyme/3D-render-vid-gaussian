"""
Reconstruction 3D avec TripoSR
Optimisé pour RTX 4090 - Mesh texturé direct
Ultra-rapide: 30-60 secondes par objet
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from rembg import remove, new_session

def reconstruct_3d_triposr(video_path: str, output_glb: str):
    """
    Reconstruction 3D ultra-rapide avec TripoSR
    Génère un mesh texturé directement
    """
    print("🚀 Reconstruction 3D avec TripoSR (RTX 4090)")
    
    workspace = Path("triposr_workspace") / Path(video_path).stem
    workspace.mkdir(parents=True, exist_ok=True)
    
    try:
        # Étape 1: Extraire la meilleure frame + segmentation
        print("  1/3 Extraction de la meilleure frame...")
        best_frame_path = extract_best_frame(video_path, workspace)
        
        if not best_frame_path:
            print("  ❌ Échec extraction frame")
            return False
        
        # Étape 2: Segmentation GPU (fond transparent)
        print("  2/3 Segmentation GPU...")
        segmented_path = segment_image_gpu(best_frame_path, workspace)
        
        if not segmented_path:
            print("  ❌ Échec segmentation")
            return False
        
        # Étape 3: TripoSR - Génération mesh
        print("  3/3 Génération mesh 3D (TripoSR)...")
        success = run_triposr(segmented_path, output_glb)
        
        if not success:
            print("  ❌ Échec TripoSR")
            return False
        
        print(f"  ✅ Mesh texturé exporté: {output_glb}")
        print("✅ Reconstruction terminée !")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_best_frame(video_path: str, workspace: Path):
    """
    Extrait la frame la plus nette de la vidéo
    """
    cap = cv2.VideoCapture(video_path)
    
    best_frame = None
    best_sharpness = 0
    frame_count = 0
    
    print("    🔍 Analyse des frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculer netteté (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame.copy()
        
        frame_count += 1
    
    cap.release()
    
    if best_frame is None:
        return None
    
    # Sauvegarder la meilleure frame
    output_path = workspace / "best_frame.jpg"
    cv2.imwrite(str(output_path), best_frame)
    
    print(f"    ✓ Meilleure frame extraite (netteté: {best_sharpness:.0f})")
    return output_path

def segment_image_gpu(image_path: Path, workspace: Path):
    """
    Segmentation GPU avec rembg (fond transparent)
    """
    print("    🔥 Initialisation GPU...")
    
    # Session GPU
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"    ✅ GPU détecté: {device}")
        session = new_session("u2net", providers=['CUDAExecutionProvider'])
    else:
        print("    ⚠️  GPU non disponible, utilisation CPU")
        session = new_session("u2net")
    
    # Charger image
    input_image = Image.open(image_path)
    
    # Segmentation
    print("    🎨 Segmentation en cours...")
    output_image = remove(
        input_image,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=270,
        alpha_matting_background_threshold=5,
        alpha_matting_erode_size=15
    )
    
    # Sauvegarder
    output_path = workspace / "segmented.png"
    output_image.save(output_path)
    
    print(f"    ✅ Segmentation terminée")
    return output_path

def run_triposr(image_path: Path, output_glb: str):
    """
    TripoSR: Image → Mesh 3D texturé
    """
    try:
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground
        
        print("    🔥 Chargement du modèle TripoSR...")
        
        # Charger le modèle sur GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.to(device)
        
        print(f"    ✅ Modèle chargé sur {device}")
        
        # Charger et préparer l'image
        print("    📸 Préparation de l'image...")
        image = Image.open(image_path).convert('RGBA')
        
        # Redimensionner pour TripoSR (optimal: 512x512)
        image = resize_foreground(image, 0.85)
        image = image.resize((512, 512), Image.LANCZOS)
        
        # Génération du mesh
        print("    🎨 Génération du mesh 3D...")
        with torch.no_grad():
            scene_codes = model([image], device=device)
        
        # Extraire le mesh
        print("    🔧 Extraction du mesh...")
        meshes = model.extract_mesh(scene_codes, resolution=256)
        mesh = meshes[0]
        
        # Exporter en GLB
        print("    💾 Export GLB...")
        mesh.export(output_glb)
        
        print(f"    ✅ Mesh créé: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        return True
        
    except Exception as e:
        print(f"    ❌ Erreur TripoSR: {e}")
        import traceback
        traceback.print_exc()
        return False
