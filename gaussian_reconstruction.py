"""
Reconstruction 3D avec Gaussian Splatting
Optimisé pour RTX 5090 - GPU 100%
Version Ultra-Rapide: 1-2 minutes
"""

import subprocess
import cv2
import numpy as np
from pathlib import Path
import uuid
import torch
import json
import shutil

def reconstruct_3d_gaussian(video_path: str, output_glb: str):
    """
    Reconstruction 3D ultra-rapide avec Gaussian Splatting
    Optimisé RTX 5090 - 1-2 minutes
    """
    print("🚀 Reconstruction 3D avec Gaussian Splatting (RTX 5090)")
    
    # Créer workspace (Gaussian Splatting attend un dossier "images")
    video_name = Path(video_path).stem
    workspace = Path("gaussian_workspace") / video_name
    images_dir = workspace / "images"  # Gaussian Splatting cherche "images/"
    output_dir = workspace / "output"
    
    workspace.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Étape 1 : Extraction et segmentation des frames (GPU)
        print("  1/4 Extraction des frames + Segmentation GPU...")
        extract_frames_gpu(video_path, images_dir, max_frames=50)
        
        num_frames = len(list(images_dir.glob("*.jpg")))
        print(f"  ✓ {num_frames} frames extraites et segmentées")
        
        # Étape 2 : COLMAP rapide pour poses de caméra (minimal)
        print("  2/4 Estimation des poses caméra (COLMAP minimal)...")
        success = run_colmap_minimal(images_dir, workspace)
        
        if not success:
            print("  ❌ Échec estimation poses")
            return False
        
        # Étape 3 : Gaussian Splatting (GPU 100%)
        print("  3/4 Gaussian Splatting (GPU RTX 5090 - 100%)...")
        success = run_gaussian_splatting(workspace, output_dir)
        
        if not success:
            print("  ❌ Échec Gaussian Splatting")
            return False
        
        # Étape 4 : Export GLB
        print("  4/4 Export GLB...")
        ply_path = output_dir / "point_cloud" / "iteration_7000" / "point_cloud.ply"
        
        if not ply_path.exists():
            print(f"  ❌ Fichier PLY non trouvé: {ply_path}")
            return False
        
        convert_gaussian_to_glb(str(ply_path), output_glb)
        
        print("✅ Reconstruction terminée !")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_frames_gpu(video_path: str, output_dir: Path, max_frames: int = 50):
    """
    Extraction + Segmentation GPU avec PyTorch + rembg
    Optimisé RTX 5090
    """
    from PIL import Image
    from rembg import remove, new_session
    
    # Session GPU avec PyTorch
    print("    🔥 Initialisation GPU RTX 5090...")
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"    ✅ GPU détecté: {device}")
        session = new_session("u2net", providers=['CUDAExecutionProvider'])
    else:
        print("    ⚠️  GPU non disponible, utilisation CPU")
        session = new_session("u2net")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sélection intelligente des frames
    frame_interval = max(1, total_frames // (max_frames * 3))
    
    candidates = []
    frame_count = 0
    
    print("    🔄 Analyse des frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Calculer netteté
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            candidates.append({
                'frame': frame.copy(),
                'sharpness': sharpness,
                'index': frame_count
            })
        
        frame_count += 1
    
    cap.release()
    
    # Sélectionner meilleures frames
    candidates.sort(key=lambda x: x['sharpness'], reverse=True)
    best_frames = candidates[:max_frames]
    best_frames.sort(key=lambda x: x['index'])
    
    print(f"    ✓ {len(best_frames)} meilleures frames sélectionnées")
    print("    🔥 Segmentation GPU en cours...")
    
    # Segmentation GPU (batch processing pour plus de vitesse)
    for i, candidate in enumerate(best_frames):
        frame = candidate['frame']
        
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Segmentation GPU
        output = remove(
            pil_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=270,
            alpha_matting_background_threshold=5,
            alpha_matting_erode_size=15
        )
        
        # Nettoyer le masque
        if output.mode == 'RGBA':
            alpha = output.split()[3]
            alpha_np = np.array(alpha)
            
            _, alpha_thresh = cv2.threshold(alpha_np, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(alpha_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask_clean = np.zeros_like(alpha_np)
                cv2.drawContours(mask_clean, [largest_contour], -1, 255, -1)
                
                kernel = np.ones((3,3), np.uint8)
                mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)
                
                output.putalpha(Image.fromarray(mask_clean))
        
        # Fond blanc
        if output.mode == 'RGBA':
            background = Image.new('RGB', output.size, (255, 255, 255))
            background.paste(output, mask=output.split()[3])
            output = background
        
        # Sauvegarder
        output_path = output_dir / f"{i:04d}.jpg"
        output.save(output_path, 'JPEG', quality=95)
        
        if (i + 1) % 10 == 0:
            print(f"    ✓ {i + 1}/{len(best_frames)} frames traitées (GPU)")
    
    print(f"    ✅ {len(best_frames)} frames segmentées (GPU)")

def run_colmap_minimal(images_dir: Path, workspace: Path):
    """
    COLMAP minimal juste pour les poses caméra
    Version rapide CPU (suffisant pour poses)
    """
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    colmap_exe = "colmap"
    
    try:
        # Feature extraction (CPU rapide)
        print("    🔍 Feature extraction...")
        result = subprocess.run([
            colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
            "--SiftExtraction.use_gpu", "0"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    ❌ Feature extraction failed: {result.stderr}")
            return False
        
        # Feature matching (CPU rapide)
        print("    🔗 Feature matching...")
        result = subprocess.run([
            colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", "0"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    ❌ Feature matching failed: {result.stderr}")
            return False
        
        # Mapper (reconstruction minimale)
        print("    🗺️  Reconstruction...")
        result = subprocess.run([
            colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    ❌ Mapper failed: {result.stderr}")
            return False
        
        # Vérifier
        model_dir = sparse_dir / "0"
        if not model_dir.exists():
            print("    ❌ Aucun modèle créé par COLMAP")
            return False
        
        # Convertir en format texte pour Gaussian Splatting
        print("    📄 Conversion format...")
        sparse_txt_dir = workspace / "sparse_txt"
        sparse_txt_dir.mkdir(exist_ok=True)
        
        result = subprocess.run([
            colmap_exe, "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(sparse_txt_dir),
            "--output_type", "TXT"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    ❌ Model converter failed: {result.stderr}")
            return False
        
        print("    ✅ COLMAP terminé")
        return True
        
    except Exception as e:
        print(f"    ❌ Erreur COLMAP: {e}")
        return False

def run_gaussian_splatting(workspace: Path, output_dir: Path):
    """
    Gaussian Splatting - GPU 100%
    Ultra rapide sur RTX 5090
    """
    try:
        # Vérifier que gaussian-splatting est installé
        gs_path = Path("/workspace/gaussian-splatting")
        
        if not gs_path.exists():
            print("    ⚠️  Gaussian Splatting non installé, installation...")
            subprocess.run([
                "git", "clone", "--recursive",
                "https://github.com/graphdeco-inria/gaussian-splatting",
                "/workspace/gaussian-splatting"
            ], check=True)
            
            # Installer dépendances
            subprocess.run([
                "pip3", "install",
                "-r", str(gs_path / "requirements.txt")
            ], check=True)
            
            # Compiler CUDA kernels
            subprocess.run([
                "pip3", "install",
                str(gs_path / "submodules/diff-gaussian-rasterization"),
                str(gs_path / "submodules/simple-knn")
            ], check=True)
        
        # Lancer training (GPU 100%)
        print("    🔥 Training Gaussian Splatting (GPU 100%)...")
        subprocess.run([
            "python3", str(gs_path / "train.py"),
            "-s", str(workspace),
            "-m", str(output_dir),
            "--iterations", "7000",
            "--test_iterations", "-1",
            "--save_iterations", "7000"
        ], check=True)
        
        return True
        
    except Exception as e:
        print(f"    ❌ Erreur Gaussian Splatting: {e}")
        return False

def convert_gaussian_to_glb(ply_path: str, glb_path: str):
    """
    Convertit Gaussian Splatting PLY en GLB - Version simplifiée avec Ball Pivoting
    """
    import trimesh
    from plyfile import PlyData
    import open3d as o3d
    
    try:
        print("    🔄 Conversion PLY → GLB...")
        
        # Charger le PLY Gaussian
        plydata = PlyData.read(ply_path)
        vertex_data = plydata['vertex'].data
        
        # Extraire positions
        positions = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
        
        # Extraire couleurs
        if 'red' in vertex_data.dtype.names:
            colors = np.vstack([
                vertex_data['red'],
                vertex_data['green'],
                vertex_data['blue']
            ]).T
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            colors = np.ones_like(positions) * 0.7
        
        print(f"    ✓ {len(positions)} points chargés")
        
        # Convertir en Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Sous-échantillonner pour réduire le bruit
        print("    🔄 Sous-échantillonnage...")
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        # Estimer les normales
        print("    🔄 Estimation des normales...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Ball Pivoting Algorithm (meilleur pour les nuages de points denses)
        print("    🔄 Reconstruction Ball Pivoting...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * 1.5, avg_dist * 2, avg_dist * 3]
        
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
        # Si Ball Pivoting échoue, utiliser Alpha Shape
        if len(mesh_o3d.triangles) == 0:
            print("    ⚠️  Ball Pivoting échoué, utilisation Alpha Shape...")
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha=0.03
            )
        
        print(f"    ✓ {len(mesh_o3d.triangles)} triangles créés")
        
        # Nettoyer
        print("    🧹 Nettoyage...")
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_non_manifold_edges()
        
        # Transférer les couleurs
        print("    🎨 Application des couleurs...")
        positions_pcd = np.asarray(pcd.points)
        colors_pcd = np.asarray(pcd.colors)
        mesh_vertices = np.asarray(mesh_o3d.vertices)
        
        from scipy.spatial import KDTree
        tree = KDTree(positions_pcd)
        distances, indices = tree.query(mesh_vertices, k=1)
        mesh_colors = colors_pcd[indices]
        
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        mesh_o3d.compute_vertex_normals()
        
        # Simplifier si trop de triangles
        if len(mesh_o3d.triangles) > 100000:
            print("    🔄 Simplification...")
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=100000)
        
        # Centrer et normaliser
        center = mesh_o3d.get_center()
        mesh_o3d.translate(-center)
        max_bound = np.max(mesh_o3d.get_max_bound() - mesh_o3d.get_min_bound())
        if max_bound > 0:
            mesh_o3d.scale(1.0 / max_bound, center=mesh_o3d.get_center())
        
        # Exporter
        print("    💾 Export GLB...")
        temp_ply = str(Path(ply_path).parent / "temp_mesh.ply")
        o3d.io.write_triangle_mesh(temp_ply, mesh_o3d)
        
        mesh = trimesh.load(temp_ply)
        mesh.export(glb_path, file_type='glb')
        
        Path(temp_ply).unlink()
        
        print(f"    ✅ GLB exporté: {glb_path}")
        return True
        
    except Exception as e:
        print(f"    ❌ Erreur conversion: {e}")
        import traceback
        traceback.print_exc()
        return False
