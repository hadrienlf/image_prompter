import os
import cv2
import torch
import numpy as np
import open_clip
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import time
import psutil
import urllib.request

# Configuration des modèles
SAM_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam_vit_h_4b8939.pth")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration pour utiliser un seul GPU spécifique (0 par défaut)
# Changez cette valeur pour utiliser un autre GPU (ex: 1) ou laissez vide pour le CPU
SELECTED_GPU = ""  
os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU

# Vérifiez si CUDA est disponible avant d'exécuter des instructions spécifiques à CUDA
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Utiliser 80% max de la mémoire disponible
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def get_device():
    """Obtient le device approprié (GPU spécifié ou CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Utilise le premier GPU visible (0)
        print(f"✅ GPU utilisé: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ Mémoire GPU: {gpu_mem:.2f} GB")
        return device
    else:
        print("⚠️ Aucun GPU disponible, utilisation du CPU")
        return torch.device("cpu")

def download_sam_model():
    """Télécharge le modèle SAM s'il n'existe pas"""
    if not os.path.exists(SAM_CHECKPOINT):
        print("📥 Téléchargement du modèle SAM...")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        os.makedirs(os.path.dirname(SAM_CHECKPOINT), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, SAM_CHECKPOINT)
            print("✅ Modèle SAM téléchargé avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement du modèle SAM: {e}")
            raise

def load_models():
    """Charge tous les modèles sur le device approprié"""
    device = get_device()
    
    # Télécharger le modèle SAM si nécessaire
    download_sam_model()
    
    # Chargement de SAM
    print("🔄 Chargement du modèle SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    sam_mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Chargement de CLIP
    print("🔄 Chargement du modèle CLIP...")
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
    """Extrait les embeddings d'une image"""
    image_pil = Image.fromarray(image_array)
    image = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image).squeeze().cpu().numpy()

def cosine_similarity(embedding1, embedding2):
    """Calcule la similarité cosinus entre deux embeddings"""
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

def process_image(img_path, query, models):
    """Traite une image avec les modèles chargés"""
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

    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)  # S'assurer que l'image est en RGB

    tokenized_query = tokenizer(query).to(device)
    query_embedding = clip_model.encode_text(tokenized_query).squeeze().detach().cpu().numpy()

    print(f"🔄 Segmentation de l'image {img_name}...")
    segmentations = sam_mask_generator.generate(source_img)
    
    print(f"🔄 Analyse des segments pour {img_name}...")
    best_match, best_score, best_mask = process_segments_batch(
        segmentations, source_img, query_embedding, models
    )

    if best_match is None:
        print(f"❌ Aucun segment trouvé pour {img_name}.")
        return None

    # Mise en surbrillance du segment trouvé
    X, Y, W, H = best_match
    source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)  # Reconvertir en BGR pour l'affichage/sauvegarde
    highlighted_image = source_img.copy()
    
    # Méthode 1: Remplissage du masque
    # mask_uint8 = best_mask.astype(np.uint8) * 255
    # mask_bg = cv2.GaussianBlur(mask_uint8, (15, 15), 5)
    # mask_idx = mask_bg > 50
    # highlighted_image[mask_idx] = [13, 218, 253]  # Format BGR pour OpenCV

    # Méthode alternative: Contours uniquement
    mask_uint8 = best_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(highlighted_image, contours, -1, (13, 218, 253), 10)  # Format BGR


    # Sauvegarde des résultats
    img_output_path = os.path.join(RESULTS_DIR, img_name)
    highlighted_output_path = os.path.join(RESULTS_DIR, f"{img_stem}_h.jpg")
    cv2.imwrite(img_output_path, source_img)
    cv2.imwrite(highlighted_output_path, highlighted_image)

    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✅ Image {img_name} traitée en {processing_time:.2f}s, score: {best_score:.4f}")
    return img_output_path, highlighted_output_path, best_score

def process_images_in_folder(folder_path, query):
    """Traite toutes les images dans un dossier en séquentiel sur un seul GPU"""
    models = load_models()
    
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Traite chaque image séquentiellement
    results = []
    for img_path in image_paths:
        result = process_image(img_path, query, models)
        if result:
            results.append(result)
    
    return results

if __name__ == "__main__":
    images_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")
    if not os.path.exists(images_folder):
        print("📂 Le dossier 'Images' n'existe pas. Utilisation du répertoire courant.")
        folder_path = os.path.dirname(os.path.abspath(__file__))
        query = "Un zebre"
    else:
        folder_path = images_folder
        query = "Une etiquette ronde rouge et noire"
    
    # Afficher des informations système
    total_cpu_mem = psutil.virtual_memory().total / (1024**3)
    print(f"🖥️ Mémoire système: {total_cpu_mem:.2f} GB")
    
    if SELECTED_GPU and torch.cuda.is_available():
        selected_idx = 0  # Premier et seul GPU visible après config CUDA_VISIBLE_DEVICES
        gpu_mem = torch.cuda.get_device_properties(selected_idx).total_memory / (1024**3)
        print(f"🎮 GPU utilisé: {torch.cuda.get_device_name(selected_idx)} avec {gpu_mem:.2f} GB")
    else:
        print("⚠️ Mode CPU activé")
    
    # Traiter toutes les images
    results = process_images_in_folder(folder_path, query)
    print(f"✅ Traitement terminé pour {len(results)} images")
