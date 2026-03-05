import json
import networkx as nx
import torch
import numpy as np
import scipy.sparse as sp
import argparse
import os
from utils import load_groundtruth_pairs


def build_graph_with_masking(user_file, item_file, review_file, gt_folder, output_file):
    print("\n--- 1. INITIALIZE GRAPH NETWORKX (WITH MASKING) ---")

    # 1. Load Users
    print("Loading Users...")
    user_ids = []
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_ids.append(str(data['user_id']))

    # 2. Load Items
    print("Loading Items...")
    item_ids = []
    with open(item_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            item_ids.append(str(data['item_id']))

    # FIX: Build deterministic integer mapping — users FIRST, then items.
    # This guarantees user indices = [0, num_users) and item indices = [num_users, num_nodes)
    # which is required for correct negative sampling in utils.py.
    num_users = len(user_ids)
    num_items = len(item_ids)
    num_nodes = num_users + num_items

    node2idx = {}
    for i, uid in enumerate(user_ids):
        node2idx[uid] = i
    for j, iid in enumerate(item_ids):
        node2idx[iid] = num_users + j

    # node_mapping[idx] = original string ID (for saving embeddings)
    node_mapping = user_ids + item_ids

    print(f"  Users : {num_users}  (indices 0 ~ {num_users - 1})")
    print(f"  Items : {num_items}  (indices {num_users} ~ {num_nodes - 1})")

    # 3. Load Mask List
    masked_pairs = load_groundtruth_pairs(gt_folder)

    # 4. Load Reviews & Build Edge Lists
    print("Loading Reviews and Building Edges...")
    rows, cols = [], []
    edge_count = 0
    skipped_count = 0

    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                u = str(data.get('user_id'))
                i = str(data.get('item_id'))

                if (u, i) in masked_pairs:
                    skipped_count += 1
                    continue

                if u in node2idx and i in node2idx:
                    u_idx = node2idx[u]
                    i_idx = node2idx[i]
                    # Bipartite edges: user->item and item->user (symmetric)
                    rows.append(u_idx);  cols.append(i_idx)
                    rows.append(i_idx);  cols.append(u_idx)
                    edge_count += 1

    except FileNotFoundError:
        print("Error: Review file not found!")
        return

    print(f"Total Edges Added   : {edge_count}")
    print(f"Total Edges Masked  : {skipped_count}")

    # 5. Build train_dict  {user_idx: [item_idx, ...]}
    print("Generating Interaction Dictionary...")
    train_dict = {}
    for u_idx, i_idx in zip(rows[0::2], cols[0::2]):   # only user->item direction
        if u_idx not in train_dict:
            train_dict[u_idx] = []
        train_dict[u_idx].append(i_idx)

    print(f"Users with interactions: {len(train_dict)}")

    print("\n--- 2. CONVERTING TO PYTORCH FORMAT ---")

    # 6. Build Adjacency Matrix (no self-connection — LightGCN paper Section 3.1.1)
    print("Creating Adjacency Matrix (WITHOUT self-connection, per LightGCN paper)...")
    data_vals = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data_vals, (rows, cols)), shape=(num_nodes, num_nodes))

    # 7. Symmetric normalisation  D^{-1/2} A D^{-1/2}  (Equation 7 in paper)
    print("Normalizing Adjacency Matrix...")
    degrees = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_norm = sp.diags(d_inv_sqrt)

    A_norm = D_norm @ A @ D_norm   # NO identity added — self-connection removed
    A_norm = A_norm.tocoo()

    indices = torch.from_numpy(
        np.vstack((A_norm.row, A_norm.col)).astype(np.int64)
    )
    values  = torch.from_numpy(A_norm.data.astype(np.float32))
    shape   = torch.Size(A_norm.shape)
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape)

    save_data = {
        'adj_norm'         : adj_tensor,
        'train_dict'       : train_dict,
        'node_item_mapping': node_mapping,
        'num_users'        : num_users,
        'num_nodes'        : num_nodes,
    }

    torch.save(save_data, output_file)
    print(f"\n✅ MASKED Graph saved to: {output_file}")
    print("You can now run 'train.py' using this file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Masked Graph for LightGCN")
    parser.add_argument('--user_file',   type=str, required=True)
    parser.add_argument('--item_file',   type=str, required=True)
    parser.add_argument('--review_file', type=str, required=True)
    parser.add_argument('--gt_folder',   type=str, required=True)
    parser.add_argument('--output_file', type=str, default='processed_graph_masked.pt')
    args = parser.parse_args()

    build_graph_with_masking(
        user_file=args.user_file,
        item_file=args.item_file,
        review_file=args.review_file,
        gt_folder=args.gt_folder,
        output_file=args.output_file,
    )