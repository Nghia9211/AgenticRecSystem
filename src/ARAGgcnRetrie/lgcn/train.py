import torch
import time
import numpy as np
import argparse

from layer import LightGCN_3Hop
from utils import sample_bpr_batch, plot_loss
from bprloss import BPRLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Training LightGCN 3-Hop Model")

    parser.add_argument('--data_file',   type=str, default='processed_graph_data.pt')
    parser.add_argument('--export_file', type=str, default='gcn_embeddings_3hop.pt')

    parser.add_argument('--epochs',     type=int,   default=1000)
    parser.add_argument('--batch_size', type=int,   default=1024)
    parser.add_argument('--lr',         type=float, default=0.001)   # paper default
    # FIX: raised default from 1e-4 → 1e-3 to control the continuously growing L2 norm
    parser.add_argument('--reg',        type=float, default=1e-3)
    parser.add_argument('--emb_dim',    type=int,   default=64)

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    print(f"Settings: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}, "
          f"Dim={args.emb_dim}, Reg={args.reg}")

    try:
        data = torch.load(args.data_file)
    except FileNotFoundError:
        print(f"Error: File not found at '{args.data_file}'")
        return

    train_dict       = data['train_dict']
    node_item_mapping = data['node_item_mapping']
    A_norm           = data['adj_norm']
    num_users        = data['num_users']
    num_nodes        = data['num_nodes']

    model     = LightGCN_3Hop(num_nodes, args.emb_dim).to(device)
    A_norm    = A_norm.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BPRLoss(reg_weight=args.reg)

    print("\n--- Start Training (LightGCN 3-Hop) ---")
    model.train()
    start_time = time.time()

    loss_history = []
    reg_history  = []

    best_loss = float('inf')
    patience  = 50         
    no_improve = 0

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        final_embs, initial_embs = model(A_norm)

        users_idx, pos_idx, neg_idx = sample_bpr_batch(
            train_dict, num_users, num_nodes, args.batch_size
        )
        users_idx = users_idx.to(device)
        pos_idx   = pos_idx.to(device)
        neg_idx   = neg_idx.to(device)

        u_final     = final_embs[users_idx]
        i_pos_final = final_embs[pos_idx]
        i_neg_final = final_embs[neg_idx]

        u_0     = initial_embs[users_idx]
        i_pos_0 = initial_embs[pos_idx]
        i_neg_0 = initial_embs[neg_idx]

        loss, bpr, reg = criterion(
            users_idx, pos_idx, neg_idx,
            u_final, i_pos_final, i_neg_final,
            u_0, i_pos_0, i_neg_0
        )

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        reg_history.append(reg.item())

        if loss.item() < best_loss - 1e-4:
            best_loss  = loss.item()
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')   
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:04d}/{args.epochs} | "
                  f"Loss: {loss.item():.4f} | BPR: {bpr.item():.4f} | Reg: {reg.item():.4f}")

    print(f"\nTraining finished in {time.time() - start_time:.2f}s")
    plot_loss(loss_history, reg_history)
    model.load_state_dict(torch.load('best_model.pt'))
    # Save embeddings
    model.eval()
    with torch.no_grad():
        final_node_embeddings, _ = model(A_norm)

        final_dict = {
            original_id: final_node_embeddings[idx].cpu()
            for idx, original_id in enumerate(node_item_mapping)
        }
        torch.save(final_dict, args.export_file)
        print(f"--> Saved 3-hop embeddings to '{args.export_file}'")
        print(f"--> Vector shape: {final_node_embeddings.shape}")


if __name__ == "__main__":
    main()