"""
===============================================================================
 GCN MODULE — Adapter for Pre-trained LightGCN 3-Hop Embeddings
===============================================================================
 
 This module DOES NOT re-implement or re-train the GCN.
 Instead, it loads the embeddings your existing LightGCN pipeline produces
 (the gcn_embeddings_3hop_*.pt files) and exposes two strategies:
 
   Strategy 1: Neighbor-Aware Context Expansion
   Strategy 2: Corpus Pre-Filtering
 
 Your existing training pipeline (build_graph_lgcn.py → train.py) remains 
 100% unchanged. This module is a READ-ONLY consumer of those embeddings.
 
 Expected input file format (from your train.py):
   torch.save(final_dict, export_file)
   where final_dict = {original_id_str: embedding_tensor, ...}
===============================================================================
"""

import torch
import torch.nn.functional as F
import json
import os
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class GCNEngine:
    """
    Loads pre-trained LightGCN 3-hop embeddings and provides:
      - Strategy 1: get_expanded_context()  → neighbor user items
      - Strategy 2: prefilter_candidates()  → narrowed candidate corpus
    
    This class needs TWO inputs:
      1. The embedding file (gcn_embeddings_3hop_*.pt)
      2. The interaction data (to know which users bought which items)
    """
    
    def __init__(self):
        # Embedding storage: {original_id_str: tensor}
        self.embeddings: Dict[str, torch.Tensor] = {}
        
        # Interaction graph (built from review data)
        # user_id -> set of item_ids
        self.user_to_items: Dict[str, set] = defaultdict(set)
        # item_id -> set of user_ids (reverse index)
        self.item_to_users: Dict[str, set] = defaultdict(set)
        
        # Item metadata cache
        self.item_metadata: Dict[str, dict] = {}
        
        # Track which IDs are users vs items
        self.user_ids: set = set()
        self.item_ids: set = set()
        
        self.is_trained = False
        self.embedding_dim = 0
    
    # =========================================================================
    # Loading — connects to YOUR existing LightGCN output
    # =========================================================================
    
    def load_embeddings(self, embedding_path: str):
        """
        Load the pre-trained embeddings from your LightGCN pipeline.
        
        This reads the file produced by your train.py:
            final_dict[original_id] = final_node_embeddings[idx].cpu()
            torch.save(final_dict, args.export_file)
        
        Args:
            embedding_path: Path to gcn_embeddings_3hop_*.pt
        """
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"GCN embedding file not found: {embedding_path}")
        
        self.embeddings = torch.load(embedding_path, map_location='cpu')
        
        # Ensure all keys are strings
        self.embeddings = {str(k): v for k, v in self.embeddings.items()}
        
        if self.embeddings:
            sample_emb = next(iter(self.embeddings.values()))
            self.embedding_dim = sample_emb.shape[0]
        
        print(f"[GCN] Loaded {len(self.embeddings)} embeddings "
              f"(dim={self.embedding_dim}) from {embedding_path}")
    
    def load_interactions(self, review_file: str, 
                          item_file: Optional[str] = None,
                          user_file: Optional[str] = None,
                          gt_mask_file: Optional[str] = None):
        """
        Load the interaction graph so we know which users bought which items.
        
        Uses the SAME data files as your build_graph_lgcn.py.
        Optionally masks ground truth pairs (same logic as your pipeline).
        
        Args:
            review_file:  Path to review_{task_type}.json
            item_file:    Path to item_{task_type}.json (for metadata)
            user_file:    Path to user_{task_type}.json (for user IDs)
            gt_mask_file: Path to final_mask_{task_type}.json (to exclude GT)
        """
        # Load user IDs
        if user_file and os.path.exists(user_file):
            with open(user_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.user_ids.add(str(data['user_id']))
            print(f"[GCN] Loaded {len(self.user_ids)} user IDs")
        
        # Load item metadata
        if item_file and os.path.exists(item_file):
            with open(item_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    item_id = str(data['item_id'])
                    self.item_ids.add(item_id)
                    self.item_metadata[item_id] = data
            print(f"[GCN] Loaded {len(self.item_ids)} items with metadata")
        
        # Load ground truth mask (same logic as your build_graph_lgcn.py)
        masked_pairs = set()
        if gt_mask_file and os.path.exists(gt_mask_file):
            with open(gt_mask_file, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
                if isinstance(gt_data, dict):
                    gt_data = [gt_data]
                for entry in gt_data:
                    u = str(entry.get('user_id', ''))
                    i = str(entry.get('item_id', ''))
                    if u and i:
                        masked_pairs.add((u, i))
            print(f"[GCN] Loaded {len(masked_pairs)} ground truth pairs to mask")
        
        # Build interaction graph from reviews
        edge_count = 0
        skipped = 0
        with open(review_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                u = str(data.get('user_id', ''))
                i = str(data.get('item_id', ''))
                
                if not u or not i:
                    continue
                
                # Apply same masking as your build_graph_lgcn.py
                if (u, i) in masked_pairs:
                    skipped += 1
                    continue
                
                self.user_to_items[u].add(i)
                self.item_to_users[i].add(u)
                self.user_ids.add(u)
                self.item_ids.add(i)
                edge_count += 1
        
        self.is_trained = len(self.embeddings) > 0
        
        print(f"[GCN] Interaction graph: {edge_count} edges, "
              f"{skipped} masked, "
              f"{len(self.user_to_items)} active users")
    
    def load(self, embedding_path: str, review_file: str,
             item_file: Optional[str] = None,
             user_file: Optional[str] = None,
             gt_mask_file: Optional[str] = None):
        """
        Convenience method: load everything in one call.
        """
        self.load_embeddings(embedding_path)
        self.load_interactions(
            review_file=review_file,
            item_file=item_file,
            user_file=user_file,
            gt_mask_file=gt_mask_file
        )
        print(f"[GCN] Engine ready. is_trained={self.is_trained}")
    
    # =========================================================================
    # ⭐ STRATEGY 1: Neighbor-Aware Context Expansion
    # =========================================================================
    
    def get_expanded_context(
        self,
        user_id: str,
        user_history_item_ids: Optional[List[str]] = None,
        top_m_neighbors: int = 5,
        max_expanded_items: int = 10
    ) -> List[Dict]:
        """
        Find users with similar GCN embeddings, collect their popular items
        that the target user hasn't interacted with.
        
        Returns items as dicts with metadata — ready for the UUA prompt.
        
        Args:
            user_id: The target user's ID
            user_history_item_ids: Items to exclude (user already saw these).
                                   If None, extracted from interaction graph.
            top_m_neighbors: How many similar users to find
            max_expanded_items: Max items to return
        
        Returns:
            List of item dicts with keys: item_id, title, description, 
            neighbor_popularity, source
        """
        if not self.is_trained:
            return []
        
        user_id = str(user_id)
        
        # Get user embedding
        u_emb = self.embeddings.get(user_id)
        if u_emb is None:
            print(f"[GCN S1] User {user_id} has no embedding. Returning empty.")
            return []
        
        # Items the user has already seen
        if user_history_item_ids is None:
            user_seen = self.user_to_items.get(user_id, set())
        else:
            user_seen = set(str(x) for x in user_history_item_ids)
        
        # ── Find top-m similar users by cosine similarity ──
        u_emb = u_emb.unsqueeze(0)  # [1, dim]
        
        other_user_ids = []
        other_user_embs = []
        for uid in self.user_ids:
            if uid == user_id:
                continue
            emb = self.embeddings.get(uid)
            if emb is not None:
                other_user_ids.append(uid)
                other_user_embs.append(emb)
        
        if not other_user_embs:
            print(f"[GCN S1] No other users with embeddings found.")
            return []
        
        all_user_embs = torch.stack(other_user_embs)  # [N, dim]
        sims = F.cosine_similarity(u_emb, all_user_embs, dim=1)  # [N]
        
        k = min(top_m_neighbors, len(other_user_ids))
        top_indices = torch.topk(sims, k).indices.tolist()
        neighbor_ids = [other_user_ids[i] for i in top_indices]
        
        # ── Collect items from neighbors that user hasn't seen ──
        item_popularity = defaultdict(int)
        for n_uid in neighbor_ids:
            for item_id in self.user_to_items.get(n_uid, set()):
                if item_id not in user_seen:
                    item_popularity[item_id] += 1
        
        # Sort by how many neighbors liked this item
        sorted_items = sorted(
            item_popularity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_expanded_items]
        
        # Build output with metadata
        expanded = []
        for item_id, count in sorted_items:
            meta = self.item_metadata.get(item_id, {})
            expanded.append({
                "item_id": item_id,
                "title": meta.get('title', meta.get('name', 
                         meta.get('title_without_series', 'Unknown'))),
                "description": str(meta.get('description', 
                               meta.get('attributes', '')))[:200],
                "neighbor_popularity": count,
                "source": "gcn_neighbor_expansion"
            })
        
        print(f"[GCN Strategy 1] {len(expanded)} expanded items "
              f"from {len(neighbor_ids)} neighbors "
              f"(user had {len(user_seen)} seen items)")
        return expanded
    
    # =========================================================================
    # ⭐ STRATEGY 2: Corpus Pre-Filtering
    # =========================================================================
    
    def prefilter_candidates(
        self,
        user_id: str,
        candidate_list: List[Dict],
        top_K: int = 50
    ) -> List[Dict]:
        """
        Score all candidate items by GCN user-item embedding similarity,
        keep only the top-K. This narrows the corpus BEFORE RAG retrieval.
        
        Items without GCN embeddings get a neutral score (0.0) so they
        aren't automatically discarded — important for cold-start items.
        
        Args:
            user_id: Target user ID
            candidate_list: Full list of candidate item dicts
            top_K: How many to keep
        
        Returns:
            Filtered candidate list (top-K by GCN score)
        """
        if not self.is_trained:
            return candidate_list
        
        user_id = str(user_id)
        u_emb = self.embeddings.get(user_id)
        
        if u_emb is None:
            print(f"[GCN S2] User {user_id} has no embedding. No filtering.")
            return candidate_list
        
        u_emb = u_emb.unsqueeze(0)  
        
        scored = []
        n_with_emb = 0
        
        for item in candidate_list:
            item_id = str(item.get('item_id', item.get('sub_item_id', '')))
            i_emb = self.embeddings.get(item_id)
            
            if i_emb is not None:
                score = F.cosine_similarity(
                    u_emb, i_emb.unsqueeze(0), dim=1
                ).item()
                n_with_emb += 1
            else:
                score = 0.0
            
            scored.append((item, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        filtered = [item for item, _ in scored[:top_K]]
        
        print(f"[GCN Strategy 2] {len(candidate_list)} → {len(filtered)} candidates "
              f"({n_with_emb}/{len(candidate_list)} had GCN embeddings)")
        
        return filtered
    
    
    def get_embedding(self, node_id: str) -> Optional[torch.Tensor]:
        """Get the pre-trained embedding for any user or item."""
        return self.embeddings.get(str(node_id))
    
    def get_user_neighbors(self, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get most similar users by embedding similarity."""
        u_emb = self.embeddings.get(str(user_id))
        if u_emb is None:
            return []
        
        u_emb = u_emb.unsqueeze(0)
        results = []
        
        for uid in self.user_ids:
            if uid == str(user_id):
                continue
            emb = self.embeddings.get(uid)
            if emb is not None:
                sim = F.cosine_similarity(u_emb, emb.unsqueeze(0)).item()
                results.append((uid, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]