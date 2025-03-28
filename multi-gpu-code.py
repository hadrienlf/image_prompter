import os
import cv2
import torch
import numpy as np
import open_clip
import torch.multiprocessing as mp
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import time
from concurrent.futures import ThreadPoolExecutor

# Configuration des modèles
SAM_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam_vit_h_4b8939.pth")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SELECTED_GPU = "0"  
os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU

torch.cuda.set_per_process_memory_fraction(0.8)  # Utiliser 80% max de la mémoire disponible
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Détection des GPUs disponibles
NUM_GPUS = torch.cuda.device_count()

def load_models_once(device_id):
    """Charge les modèles une seule fois par GPU"""
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    # Chargement de SAM
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    sam_mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Chargement de CLIP
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    
    return {
        'device': device,
        'sam_mask_generator': sam_mask_generator,
        'clip_model': model,
        'tokenizer': tokenizer,
        'preprocess': preprocess
    }

def get_image_embeddings(image_array, model, preprocess, device):
    image_pil = Image.fromarray(image_array)
    image = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image).squeeze().cpu().numpy()

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(
        torch.tensor(embedding1).unsqueeze(0),
        torch.tensor(embedding2).unsqueeze(0)
    ).item()

def process_segments_batch(segments, source_img, query_embedding, models, batch_size=4):
    """Traite les segments par lots pour une meilleure efficacité"""
    device = models['device']
    clip_model = models['clip_model']
    preprocess = models['preprocess']
    
    best_match, best_score, best_mask = None, -1, None
    
    # Traitement par lots
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        batch_scores = []
        batch_info = []
        
        for seg in batch:
            X, Y, W, H = seg["bbox"]
            binary_mask = seg["segmentation"].astype(int)
            cropped_img = np.where(binary_mask[..., None] > 0, source_img, 255)[Y:Y + H, X:X + W]
            segment_embedding = get_image_embeddings(cropped_img, clip_model, preprocess, device)
            
            similarity_score = cosine_similarity(query_embedding, segment_embedding)
            batch_scores.append(similarity_score)
            batch_info.append((X, Y, W, H, seg["segmentation"]))
        
        # Trouver le meilleur score dans ce lot
        max_idx = np.argmax(batch_scores)
        if batch_scores[max_idx] > best_score:
            best_score = batch_scores[max_idx]
            X, Y, W, H, mask = batch_info[max_idx]
            best_match = (X, Y, W, H)
            best_mask = mask
    
    return best_match, best_score, best_mask

def process_image_with_loaded_models(img_path, query, models, device_id):
    """Traite une image avec les modèles déjà chargés"""
    start_time = time.time()
    
    device = models['device']
    sam_mask_generator = models['sam_mask_generator']
    clip_model = models['clip_model']
    tokenizer = models['tokenizer']
    preprocess = models['preprocess']

    img_name = os.path.basename(img_path)
    img_stem = os.path.splitext(img_name)[0]
    source_img = cv2.imread(img_path)
    
    if source_img is None:
        print(f"❌ Impossible de charger l'image {img_path}")
        return None

    # Préparation de la requête une seule fois
    tokenized_query = tokenizer(query).to(device)
    query_embedding = clip_model.encode_text(tokenized_query).squeeze().detach().cpu().numpy()

    # Génération des segmentations
    segmentations = sam_mask_generator.generate(source_img)
    
    # Traitement par lots des segments
    best_match, best_score, best_mask = process_segments_batch(
        segmentations, source_img, query_embedding, models
    )

    if best_match is None:
        print(f"❌ Aucun segment trouvé pour {img_name}.")
        return None

    # Mise en surbrillance du segment trouvé - optimisé
    X, Y, W, H = best_match
    highlighted_image = source_img.copy()
    
    # # Utiliser une opération de masque plus efficace
    # mask_bg = cv2.GaussianBlur((best_mask * 255).astype(np.uint8), (15, 15), 5)
    # mask_idx = mask_bg > 50
    # # Application du masque en une seule opération
    # highlighted_image[mask_idx] = [253, 218, 13]
    
    mask_uint8 = best_mask.astype(np.uint8) * 255
    # Trouver les contours du masque
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dessiner les contours sur l'image
    cv2.drawContours(highlighted_image, contours, -1, (253, 218, 13), 10)

    # Sauvegarde des résultats
    img_output_path = os.path.join(RESULTS_DIR, img_name)
    highlighted_output_path = os.path.join(RESULTS_DIR, f"{img_stem}_h.jpg")
    cv2.imwrite(img_output_path, source_img)
    cv2.imwrite(highlighted_output_path, highlighted_image)

    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✅ Image {img_name} traitée sur GPU {device_id} en {processing_time:.2f}s, pour un score de : {best_score}")
    return img_output_path, highlighted_output_path, best_score

def worker_process(gpu_id, image_paths, query):
    """Fonction de travail pour chaque processus GPU"""
    # Charge les modèles une seule fois par processus
    models = load_models_once(gpu_id)
    
    # Sélectionne les images à traiter par ce GPU
    gpu_images = [img for i, img in enumerate(image_paths) if i % NUM_GPUS == gpu_id]
    
    # Traite toutes les images assignées à ce GPU
    for img_path in gpu_images:
        process_image_with_loaded_models(img_path, query, models, gpu_id)

def process_images_in_folder(folder_path, query):
    """Distribue les images entre les GPUs disponibles"""
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if NUM_GPUS > 1:
        # Utilisation du multiprocessing pour paralléliser sur plusieurs GPUs
        processes = []
        for gpu_id in range(NUM_GPUS):
            p = mp.Process(target=worker_process, args=(gpu_id, image_paths, query))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        # Un seul GPU disponible
        worker_process(0, image_paths, query)

if __name__ == "__main__":
    images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")
    if not os.path.exists(images_folder):
        print("📂 Le dossier 'Images' n'existe pas. Utilisation du répertoire courant.")
        folder_path = os.path.dirname(os.path.abspath(__file__))
        query = "Un zebre"
    else:
        folder_path = images_folder
        query = "Une etiquette ronde rouge et noire"

    total_cpu_mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    print(f"🖥️ Mémoire système: {total_cpu_mem:.2f} GB")
    for i in range(NUM_GPUS):
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"🎮 GPU {i}: {torch.cuda.get_device_name(i)} avec {gpu_mem:.2f} GB")
    
    mp.set_start_method('spawn')  # Important pour éviter les erreurs sous Windows
    process_images_in_folder(folder_path, query)