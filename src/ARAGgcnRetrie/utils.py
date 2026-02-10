# --- START OF FILE utils.py ---

from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any, Callable, Dict
import torch
import torch.nn.functional as F
import json
import os
import csv
from datetime import datetime
import threading

def normalize_item_data(item: dict) -> dict:
    item_id = str(item.get('item_id', item.get('sub_item_id', 'unknown_id')))
    name = item.get('title') or item.get('name') or item.get('business_name') or f"Item {item_id}"
    raw_desc = item.get('description') or item.get('text') or ""
    if isinstance(raw_desc, list):
        clean_desc = " ".join([str(x) for x in raw_desc])
    else:
        clean_desc = str(raw_desc)
    category = item.get('categories') or item.get('type') or "General"
    if isinstance(category, list):
        category = ", ".join(category)

    return {
        "item_id": item_id,
        "name": name,
        "description": clean_desc,
        "category": str(category),
        "original_data": item 
    }

def get_batch_embeddings(texts: List[str], embedding_function: Callable) -> torch.Tensor:
    vectors = embedding_function.embed_documents(texts)
    return torch.tensor(vectors)


def perform_rag_retrieval(
    augmented_query: str,
    candidate_list: List[dict],
    embedding_function: Callable,
    top_k: int = 5
) -> List[dict]:
    """
    Hàm RAG thuần túy:
    Dùng câu query (đã được bơm GCN Context) để tìm kiếm Semantic Similarity.
    """
    sims = []
    
    query_vec = embedding_function.embed_query(augmented_query)
    
    item_texts = [str(item) for item in candidate_list]
    item_vectors = embedding_function.embed_documents(item_texts) 
    
    for item, item_vec in zip(candidate_list, item_vectors):
        sim = cosine_similarity([item_vec], [query_vec])[0][0]
        sims.append((item, sim))
    
    sims.sort(key = lambda x: x[1], reverse = True)
    top_k_list = [item for item, sim in sims[:top_k]]
    
    return top_k_list


def print_agent_step(agent_name: str, message: str, data: Any = None):
    header = f"=== [AGENT: {agent_name.upper()}] ==="
    print(f"\n\033[94m{header}\033[0m") 
    print(f"💬 {message}")
    if data:
        if isinstance(data, list):
            print(f"📊 Items count: {len(data)}")
            for i, item in enumerate(data[:3]):
                print(f"   - Item {i+1}: {item}")
            if len(data) > 3: print("   ...")
        else:
            print(f"📝 Data: {data}")
    print("\033[94m" + "="*len(header) + "\033[0m\n")

def get_gcn_latent_interests(user_history_ids, gcn_embeddings, candidate_list):
    """
    Tìm ra 'Gu' của người dùng dựa trên các item lân cận trong đồ thị.
    """
    if not gcn_embeddings or not user_history_ids:
        return "No specific graph patterns detected."

    # 1. Tìm User Embedding
    user_vecs = [gcn_embeddings[uid] for uid in user_history_ids if uid in gcn_embeddings]
    if not user_vecs: return "No graph data available."
    user_emb = torch.stack(user_vecs).mean(dim=0)

    # 2. Tìm các Item 'hàng xóm' trong đồ thị
    cand_ids = [str(item['item_id']) for item in candidate_list if str(item['item_id']) in gcn_embeddings]
    if not cand_ids: return "New user profile with limited graph connections."
    
    cand_tensor = torch.stack([gcn_embeddings[iid] for iid in cand_ids])
    sims = torch.nn.functional.cosine_similarity(user_emb.unsqueeze(0), cand_tensor)
    
    top_k = torch.topk(sims, k=min(5, len(cand_ids)))
    
    suggested_features = []
    for idx in top_k.indices:
        item_id = cand_ids[idx]
        item_data = next(i for i in candidate_list if str(i['item_id']) == item_id)
        
        name = item_data.get('name') or item_data.get('title') or "Unknown Item"
        
        cats = item_data.get('categories', [])
        if isinstance(cats, list) and len(cats) > 0:
            main_cat = cats[0] 
        elif isinstance(cats, str) and cats:
            main_cat = cats.split(',')[0].strip() 
        else:
            shelves = item_data.get('popular_shelves', [])
            if isinstance(shelves, list) and len(shelves) > 0:
                ignore_tags = {'to-read', 'currently-reading', 'owned', 'favorites', 'default', 'all-time-favorites'}
                main_cat = "Book" 
                for s in shelves:
                    tag_name = s.get('name')
                    if tag_name not in ignore_tags:
                        main_cat = tag_name
                        break
            else:
                main_cat = item_data.get('category', 'General')
        suggested_features.append(f"{main_cat} ({name})")

    return f"Structural community patterns suggest a preference for: {', '.join(suggested_features)}."

class ARAGMetrics:
    def __init__(self, experiment_name="ARAG_GCN_Retrie"):
        self.experiment_name = experiment_name
        self.filename = None
        self.lock = threading.Lock()
        self.initialized = False  

    def _initialize_file(self):
        if not self.initialized:
            log_dir = "experiment_logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = os.path.join(log_dir, f"stats_{self.experiment_name}_{timestamp_str}.csv")
            
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Task_Set", "Index", "Stage", "GT_ID", "Hit", "Total_Items", "Position"])
            
            self.initialized = True
            print(f"🚀 [METRICS] File log đã được khởi tạo: {self.filename}")

    def log_hit(self, task_set, index, stage, gt_id, is_hit, total_items, position):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with self.lock:
            if not self.initialized:
                self._initialize_file()
                
            with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, task_set, index, stage, gt_id, is_hit, total_items, position])
metrics_logger = ARAGMetrics()

def debug_ground_truth_hit(index: int, stage: str, items: list, gt_folder: str, task_set: str = "unknown"):
    """
    Kiểm tra và ghi log kết quả tìm thấy Ground Truth
    """
    file_path = os.path.join(gt_folder, f"groundtruth_{index}.json")
    
    if not os.path.exists(file_path):
        return False, "N/A"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            gt_id = str(gt_data.get("ground truth", ""))
    except Exception:
        return False, "N/A"

    current_ids = []
    for item in items:
        if isinstance(item, dict):
            current_ids.append(str(item.get('item_id', '')))
        else:
            current_ids.append(str(getattr(item, 'item_id', '')))

    is_hit = gt_id in current_ids
    position = current_ids.index(gt_id) + 1 if is_hit else -1

    metrics_logger.log_hit(
        task_set=task_set,
        index=index,
        stage=stage,
        gt_id=gt_id,
        is_hit=is_hit,
        total_items=len(current_ids),
        position=position
    )
    status_icon = "✅" if is_hit else "❌"
    print(f"{status_icon} [{stage}] Index {index}: GT {gt_id} {'FOUND at pos ' + str(position) if is_hit else 'NOT FOUND'}")
    
    return is_hit