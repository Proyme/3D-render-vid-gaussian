"""
Reconstruction 3D avec COLMAP à partir d'une vidéo
"""
import cv2
import numpy as np
import subprocess
from pathlib import Path
import shutil
import os
import uuid

def reconstruct_3d_from_images(images_dir: str, output_glb_path: str) -> bool:
    """
    Reconstruit un modèle 3D à partir d'un dossier d'images
    """
    print("📸 Reconstruction 3D à partir de photos...")
    
    images_path = Path(images_dir)
    workspace = Path("colmap_workspace") / str(uuid.uuid4())
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Dossiers COLMAP
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    dense_dir = workspace / "dense"
    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)
    
    # Chemin vers COLMAP (Linux pour RunPod, Windows pour local)
    import platform
    if platform.system() == "Windows":
        colmap_exe = r"D:\application plats en 3d\backend-3d\COLMAP\COLMAP.bat"
    else:
        colmap_exe = "colmap"  # Linux/RunPod
    
    try:
        # Étape 1 : Feature extraction
        print("  1/5 Extraction des features...")
        subprocess.run([
            colmap_exe, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_path)
        ], check=True)
        
        # Étape 2 : Feature matching
        print("  2/5 Matching des features...")
        subprocess.run([
            colmap_exe, "exhaustive_matcher",
            "--database_path", str(database_path)
        ], check=True)
        
        # Étape 3 : Reconstruction sparse (Structure from Motion)
        print("  3/5 Reconstruction sparse (SfM)...")
        subprocess.run([
            colmap_exe, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_path),
            "--output_path", str(sparse_dir)
        ], check=True)
        
        # Vérifier que la reconstruction a réussi
        model_dir = sparse_dir / "0"
        if not model_dir.exists():
            print("❌ Échec de la reconstruction sparse")
            return False
        
        # Étape 4 : Reconstruction dense (MVS)
        print("  4/5 Reconstruction dense (MVS)...")
        use_dense = False
        try:
            subprocess.run([
                colmap_exe, "image_undistorter",
                "--image_path", str(images_path),
                "--input_path", str(model_dir),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP"
            ], check=True)
            
            subprocess.run([
                colmap_exe, "patch_match_stereo",
                "--workspace_path", str(dense_dir)
            ], check=True)
            
            fused_ply = dense_dir / "fused.ply"
            subprocess.run([
                colmap_exe, "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(fused_ply)
            ], check=True)
            
            print("  ✅ MVS réussi ! Utilisation du mesh dense")
            use_dense = True
            
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  MVS échoué, utilisation du sparse")
            use_dense = False
        
        # Étape 5 : Convertir en GLB
        print("  5/5 Conversion en GLB...")
        if use_dense and (dense_dir / "fused.ply").exists():
            convert_ply_to_glb(dense_dir / "fused.ply", output_glb_path)
        else:
            sparse_ply = sparse_dir / "points3D.ply"
            subprocess.run([
                colmap_exe, "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(sparse_ply),
                "--output_type", "PLY"
            ], check=True)
            convert_ply_to_glb(sparse_ply, output_glb_path)
        
        # Nettoyer
        shutil.rmtree(workspace)
        
        print("✅ Reconstruction 3D terminée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        if workspace.exists():
            shutil.rmtree(workspace)
        return False

def reconstruct_3d_from_video(video_path: str, output_glb_path: str) -> bool:
    """
    Reconstruit un modèle 3D à partir d'une vidéo avec COLMAP
    
    Args:
        video_path: Chemin vers la vidéo
        output_glb_path: Chemin de sortie pour le fichier .glb
    
    Returns:
        True si succès, False sinon
    """
    print("🎥 Reconstruction 3D avec COLMAP...")
    
    # Créer un dossier temporaire pour les frames
    video_name = Path(video_path).stem
    workspace = Path("colmap_workspace") / video_name
    images_dir = workspace / "images"
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    dense_dir = workspace / "dense"
    
    # Nettoyer et créer les dossiers
    if workspace.exists():
        shutil.rmtree(workspace)
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)
    
    # Étape 1 : Extraire les frames de la vidéo
    print("  1/6 Extraction des frames...")
    # Segmentation activée pour isoler l'objet et améliorer la qualité
    # 50 frames pour une meilleure couverture (surtout pour objets complexes)
    extract_frames_from_video(video_path, images_dir, max_frames=50, use_segmentation=True)
    
    # Vérifier combien de frames ont été extraites
    num_extracted = len(list(images_dir.glob("*.jpg")))
    print(f"  ✓ {num_extracted} frames extraites et sauvegardées")
    
    # Chemin vers COLMAP
    colmap_exe = r"D:\application plats en 3d\backend-3d\COLMAP\COLMAP.bat"
    
    # Étape 2 : Feature extraction
    print("  2/6 Extraction des features...")
    subprocess.run([
        colmap_exe, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir)
    ], check=True)
    
    # Étape 3 : Feature matching
    print("  3/6 Matching des features...")
    subprocess.run([
        colmap_exe, "exhaustive_matcher",
        "--database_path", str(database_path)
    ], check=True)
    
    # Étape 4 : Reconstruction sparse (Structure from Motion)
    print("  4/6 Reconstruction sparse (SfM)...")
    subprocess.run([
        colmap_exe, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir)
    ], check=True)
    
    # Vérifier que la reconstruction a réussi
    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        print("❌ Échec de la reconstruction sparse")
        return False
    
    # Étape 5 : Reconstruction dense (MVS) - TEST
    print("  5/6 Reconstruction dense (MVS)...")
    try:
        # Undistort images
        subprocess.run([
            colmap_exe, "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(model_dir),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP"
        ], check=True)
        
        # Patch match stereo
        subprocess.run([
            colmap_exe, "patch_match_stereo",
            "--workspace_path", str(dense_dir)
        ], check=True)
        
        # Stereo fusion
        fused_ply = dense_dir / "fused.ply"
        subprocess.run([
            colmap_exe, "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--output_path", str(fused_ply)
        ], check=True)
        
        print("  ✅ MVS réussi ! Utilisation du mesh dense")
        use_dense = True
        
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  MVS échoué (pas de CUDA MVS), utilisation du sparse")
        use_dense = False
    
    # Étape 6 : Convertir en GLB
    print("  6/6 Conversion en GLB...")
    if use_dense and (dense_dir / "fused.ply").exists():
        convert_ply_to_glb(dense_dir / "fused.ply", output_glb_path)
    else:
        # Fallback sur sparse
        sparse_ply = sparse_dir / "points3D.ply"
        subprocess.run([
            colmap_exe, "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(sparse_ply),
            "--output_type", "PLY"
        ], check=True)
        convert_ply_to_glb(sparse_ply, output_glb_path)
    
    # Nettoyer
    shutil.rmtree(workspace)
    
    print("✅ Reconstruction 3D terminée")
    return True

def extract_frames_from_video(video_path: str, output_dir: Path, max_frames: int = 50, use_segmentation: bool = True):
    """
    Extrait les meilleures frames de la vidéo basées sur la nettete
    Et applique la segmentation automatique pour isoler l'objet (optionnel)
    
    Args:
        use_segmentation: Si True, applique la segmentation pour isoler l'objet
    """
    from PIL import Image
    import io
    
    if use_segmentation:
        from rembg import remove
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("    🔄 Analyse des frames pour sélection intelligente...")
    
    # Étape 1 : Extraire des candidats (3x plus que nécessaire)
    candidate_count = max_frames * 3
    frame_interval = max(1, total_frames // candidate_count)
    
    candidates = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Calculer la nettete (variance du Laplacien)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            candidates.append({
                'frame': frame,
                'sharpness': sharpness,
                'index': frame_count
            })
        
        frame_count += 1
    
    cap.release()
    
    # Étape 2 : Sélectionner les meilleures frames
    candidates.sort(key=lambda x: x['sharpness'], reverse=True)
    best_frames = candidates[:max_frames]
    best_frames.sort(key=lambda x: x['index'])  # Re-trier par ordre chronologique
    
    print(f"    ✓ {len(best_frames)} meilleures frames sélectionnées sur {len(candidates)} candidats")
    print(f"    📊 Vidéo totale: {total_frames} frames, intervalle: {frame_interval}")
    
    if use_segmentation:
        print("    🔄 Segmentation des frames...")
    else:
        print("    🔄 Sauvegarde des frames (sans segmentation)...")
    
    # Étape 3 : Segmenter (optionnel) et sauvegarder
    for i, candidate in enumerate(best_frames):
        frame = candidate['frame']
        
        # Convertir BGR (OpenCV) en RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        if use_segmentation:
            # Supprimer l'arrière-plan avec rembg (paramètres agressifs pour isoler l'objet)
            output = remove(
                pil_image,
                alpha_matting=True,
                alpha_matting_foreground_threshold=270,  # Plus agressif
                alpha_matting_background_threshold=5,     # Plus strict
                alpha_matting_erode_size=15               # Plus d'érosion
            )
            
            # Créer un masque strict pour isoler uniquement l'objet
            if output.mode == 'RGBA':
                alpha = output.split()[3]
                alpha_np = np.array(alpha)
                
                # Appliquer un seuillage pour ne garder que les pixels vraiment opaques
                _, alpha_thresh = cv2.threshold(alpha_np, 200, 255, cv2.THRESH_BINARY)
                
                # Trouver le plus grand contour (l'objet principal)
                contours, _ = cv2.findContours(alpha_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Garder seulement le plus grand contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    mask = np.zeros_like(alpha_np)
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                    
                    # Dilater légèrement pour ne pas couper les bords
                    kernel = np.ones((10, 10), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    alpha = Image.fromarray(mask)
                else:
                    alpha = Image.fromarray(alpha_thresh)
                
                # Remplacer le fond par du blanc pur
                white_bg = Image.new('RGB', output.size, (255, 255, 255))
                white_bg.paste(output, mask=alpha)
            else:
                white_bg = output.convert('RGB')
        else:
            # Pas de segmentation, garder l'image originale
            white_bg = pil_image
        
        # Sauvegarder
        frame_path = output_dir / f"frame_{i:04d}.jpg"
        white_bg.save(str(frame_path), 'JPEG', quality=95)
        
        if (i + 1) % 4 == 0:
            print(f"      ✓ {i + 1}/{max_frames} frames traitées")
    
    print(f"    ✓ {len(best_frames)} frames extraites et segmentées")

def convert_ply_to_glb(ply_path: Path, glb_path: str, aggressive_cleanup: bool = True):
    """
    Convertit un fichier PLY (nuage de points) en GLB (mesh solide)
    Utilise Poisson Surface Reconstruction
    """
    import trimesh
    import open3d as o3d
    
    print("    🔄 Conversion nuage de points → mesh solide (Poisson)...")
    
    try:
        # Charger le nuage de points
        pcd = o3d.io.read_point_cloud(str(ply_path))
        
        num_points = len(pcd.points)
        print(f"    ✓ Nuage de points chargé: {num_points} points")
        
        # Estimer les normales
        print("    🔄 Estimation des normales...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Reconstruction de surface avec Poisson (depth=10 pour plus de détails)
        print("    🔄 Reconstruction de surface (Poisson)...")
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10, width=0, scale=1.1, linear_fit=False
        )
        
        print(f"    ✓ Mesh créé: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} faces")
        
        # Projeter les couleurs du nuage de points sur le mesh (avec moyenne pour éviter aberrations)
        if pcd.has_colors():
            print("    🔄 Projection des couleurs sur le mesh...")
            # Créer un KDTree pour trouver les points les plus proches
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            
            # Pour chaque vertex du mesh, moyenner les couleurs des 5 points les plus proches
            mesh_colors = []
            for vertex in mesh_o3d.vertices:
                [k, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 5)  # 5 voisins au lieu de 1
                colors = np.asarray(pcd.colors)[idx]
                # Moyenne des couleurs pour éviter les aberrations
                avg_color = np.mean(colors, axis=0)
                mesh_colors.append(avg_color)
            
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))
            print("    ✓ Couleurs projetées sur le mesh (avec moyenne)")
        
        # Nettoyer le mesh (enlever les vertices à très faible densité)
        vertices_to_remove = densities < np.quantile(densities, 0.005)  # Plus conservateur (0.5% au lieu de 1%)
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"    ✓ Mesh nettoyé: {len(mesh_o3d.vertices)} vertices")
        
        # Enlever seulement les très petits clusters isolés (bruit)
        print("    🔄 Nettoyage des petits clusters...")
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_o3d.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # Garder le plus grand cluster + tous les clusters avec au moins 2% de sa taille
        # Plus agressif pour enlever les "gouttes" (artefacts)
        largest_cluster_idx = cluster_n_triangles.argmax()
        largest_cluster_size = cluster_n_triangles[largest_cluster_idx]
        min_cluster_size = largest_cluster_size * 0.02  # 2% du plus grand (plus agressif)
        
        clusters_to_keep = cluster_n_triangles >= min_cluster_size
        triangles_to_remove = ~np.isin(triangle_clusters, np.where(clusters_to_keep)[0])
        mesh_o3d.remove_triangles_by_mask(triangles_to_remove)
        mesh_o3d.remove_unreferenced_vertices()
        
        print(f"    ✓ Petits clusters enlevés: {len(mesh_o3d.vertices)} vertices")
        
        # Lissage plus agressif pour réduire les "gouttes" (artefacts)
        print("    🔄 Lissage du mesh (réduction des artefacts)...")
        mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=5)  # 5 itérations pour lisser les gouttes
        mesh_o3d.compute_vertex_normals()
        
        # Simplification modérée pour réduire la taille sans perdre les détails
        target_triangles = int(len(mesh_o3d.triangles) * 0.85)  # Garde 85%
        if target_triangles > 10000:  # Minimum 10k triangles pour garder les détails
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
            mesh_o3d.compute_vertex_normals()
        
        print(f"    ✓ Mesh optimisé: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} triangles")
        
        # Centrer et normaliser le mesh
        mesh_o3d.translate(-mesh_o3d.get_center())
        mesh_o3d.scale(1.0 / np.max(mesh_o3d.get_max_bound() - mesh_o3d.get_min_bound()), center=mesh_o3d.get_center())
        
        print("    ✓ Mesh centré et normalisé")
        
        # Sauvegarder temporairement en PLY
        temp_mesh_ply = str(ply_path).replace('.ply', '_mesh.ply')
        o3d.io.write_triangle_mesh(temp_mesh_ply, mesh_o3d)
        
        # Charger avec trimesh et corriger les normales
        print("    🔄 Correction des normales...")
        mesh = trimesh.load(temp_mesh_ply)
        
        # Inverser les normales si nécessaire (fix pour textures inversées)
        # Vérifier l'orientation des normales
        mesh.fix_normals()  # Corrige automatiquement les normales
        
        # Si les normales sont toujours inversées, les inverser manuellement
        if not mesh.is_winding_consistent:
            print("    ⚠️  Normales inconsistantes détectées, inversion...")
            mesh.invert()
        
        print("    ✓ Normales corrigées")
        
        # Exporter en GLB
        mesh.export(glb_path, file_type='glb')
        
        # Nettoyer
        Path(temp_mesh_ply).unlink()
        
        print(f"    ✓ Mesh solide exporté: {glb_path}")
        
    except Exception as e:
        print(f"    ❌ Erreur conversion: {e}")
        import traceback
        traceback.print_exc()
        raise
