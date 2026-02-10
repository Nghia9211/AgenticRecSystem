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

def generate_graph_context_string(
    user_history_ids: List[str],
    gcn_embeddings: Dict[str, torch.Tensor],
    candidate_list: List[dict],
    top_k_neighbors: int = 5
) -> str:
    if not gcn_embeddings or not user_history_ids:
        return ""

    user_vecs = []
    for uid in user_history_ids:
        if uid in gcn_embeddings:
            user_vecs.append(gcn_embeddings[uid])
    
    if not user_vecs:
        return ""
    
    user_emb = torch.stack(user_vecs).mean(dim=0).unsqueeze(0)

    candidates_with_gcn = []
    for item in candidate_list:
        # Đảm bảo item là dict
        if not isinstance(item, dict): continue
        
        iid = str(item.get('item_id', ''))
        if iid in gcn_embeddings:
            candidates_with_gcn.append((item, gcn_embeddings[iid]))
    
    if not candidates_with_gcn:
        return ""

    cand_tensor = torch.stack([x[1] for x in candidates_with_gcn])
    scores = F.cosine_similarity(user_emb, cand_tensor)
    top_indices = torch.topk(scores, k=min(top_k_neighbors, len(candidates_with_gcn))).indices.tolist()
    
    suggested_items = [candidates_with_gcn[i][0] for i in top_indices]
    suggested_strings = []

    for item in suggested_items:
        name = item.get('name') or item.get('title') or "Unknown Item"
        cats = item.get('categories', [])
        main_cat = "Item"

        if isinstance(cats, list) and len(cats) > 0:
            main_cat = cats[0] 
        elif isinstance(cats, str) and cats.strip():
            main_cat = cats.split(',')[0] 
        else:
            shelves = item.get('popular_shelves', []) 
            if not isinstance(shelves, list):
                shelves = []

            ignore_tags = {'to-read', 'currently-reading', 'owned', 'favorites', 'default', 'all-time-favorites'}
            main_cat = "Book" 
            
            for s in shelves:
                # Kiểm tra s phải là dict mới được .get()
                if isinstance(s, dict):
                    tag_name = s.get('name')
                    if tag_name and tag_name not in ignore_tags:
                        main_cat = tag_name
                        break 
        
        suggested_strings.append(f"'{name}' ({main_cat})")
    
    context_str = (
        f"Graph Analysis: The user's interaction network strongly aligns with these items: {', '.join(suggested_strings)}. "
        "Items sharing similar structural patterns in the graph should be prioritized."
    )

    return context_str
def perform_rag_retrieval(
    augmented_query: str,
    candidate_list: List[dict],
    embedding_function: Callable,
    top_k: int = 5) -> List[dict]:
    """
    Hàm RAG thuần túy:
    Dùng câu query (đã được bơm GCN Context) để tìm kiếm Semantic Similarity.
    """
    
    query_vec = embedding_function.embed_query(augmented_query)

    candidate_texts = [
        f"Name: {item['name'] or item['title']}. Category: {item['category']}. Description: {item['description'][:300]}" 
        for item in candidate_list
    ]
    cand_vecs = embedding_function.embed_documents(candidate_texts)
    sims = cosine_similarity([query_vec], cand_vecs)[0]
    
    sorted_indices = sims.argsort()[::-1][:top_k]
    
    results = []
    for idx in sorted_indices:
        results.append(candidate_list[idx])
        
    return results



class ARAGMetrics:
    def __init__(self, filename="recommendation_stats_ARAG_GCN.csv"):
        self.filename = filename
        self.lock = threading.Lock()

        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Task_Set", "Index", "Stage", "GT_ID", "Hit", "Total_Items", "Position"])

    def log_hit(self, task_set, index, stage, gt_id, is_hit, total_items, position):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
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