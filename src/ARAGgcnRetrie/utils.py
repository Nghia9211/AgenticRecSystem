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


def find_top_k_similar_items(
    query: str,
    candidate_list: List[Any],
    embedding_function: Callable,
    k: int = 5
) -> List[Any]:

    query_vec = embedding_function.embed_query(query)
    texts = [json.dumps(normalize_item_data(c)) for c in candidate_list]

    item_vecs = embedding_function.embed_documents(texts)
    sims = cosine_similarity([query_vec], item_vecs)[0]
    
    results = sorted(zip(candidate_list, sims), key=lambda x: x[1], reverse=True)
    return [res[0] for res in results[:k]]


def normalize_item_data(item: Any) -> dict:
    """Chuẩn hóa dữ liệu item từ nhiều nguồn khác nhau."""
    if isinstance(item, str):
        try: item = json.loads(item.replace("'", '"'))
        except: return {}

    return {
        "item_id": str(item.get('item_id', item.get('sub_item_id', 'unknown'))),
        "name": item.get('title') or item.get('name') or item.get('title_without_series'),
        "description": str(item.get('description') or item.get('attributes') or "")[:200],
        "Rating": str(item.get('stars') or item.get('average_rating') or ""),
    }
def get_last_message(blackboard, role):
    return next((msg for msg in reversed(blackboard) if msg.role == role), None)

def get_user_understanding(state):
    msg = get_last_message(state['blackboard'], "UserUnderStanding")
    return msg.content if msg else ""

def get_user_summary(state):
    msg = get_last_message(state['blackboard'], "ContextSummary")
    return msg.content if msg else ""

# def perform_rag_retrieval(
#     augmented_query: str,
#     candidate_list: List[dict],
#     embedding_function: Callable,
#     top_k: int = 5
# ) -> List[dict]:
#     """
#     Hàm RAG thuần túy:
#     Dùng câu query (đã được bơm GCN Context) để tìm kiếm Semantic Similarity.
#     """
#     sims = []
    
#     query_vec = embedding_function.embed_query(augmented_query)
    
#     item_texts = [str(item) for item in candidate_list]
#     item_vectors = embedding_function.embed_documents(item_texts) 
    
#     for item, item_vec in zip(candidate_list, item_vectors):
#         sim = cosine_similarity([item_vec], [query_vec])[0][0]
#         sims.append((item, sim))
    
#     sims.sort(key = lambda x: x[1], reverse = True)
#     top_k_list = [item for item, sim in sims[:top_k]]
    
#     return top_k_list


def get_gcn_latent_interests(user_history_ids, gcn_embeddings, candidate_list):
    if not gcn_embeddings or not user_history_ids:
        return "No specific graph patterns detected."

    user_vecs = [gcn_embeddings[uid] for uid in user_history_ids if uid in gcn_embeddings]
    if not user_vecs: return "No graph data available."
    user_emb = torch.stack(user_vecs).mean(dim=0)

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
