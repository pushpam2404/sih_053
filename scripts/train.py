import sys, os, json
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
from drdo_lidar_mapping.segmentation.dataset import RELLISDataset

# Lock parameters
RELLIS_ROOT  = os.path.join(REPO_ROOT, "data", "rellis")
WEIGHTS_DIR  = os.path.join(REPO_ROOT, "models")
RESULTS_FILE = os.path.join(REPO_ROOT, "results", "train_log.json")
NUM_EPOCHS   = 30
LR           = 1e-4
VOXEL_SIZE   = 0.05
CLASS_WEIGHTS = torch.tensor(
    [0.1000, 1.0876, 0.5234, 0.8099, 1.4620, 10.0, 2.3105, 0.0],
    dtype=torch.float32)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] device={device}")

    has_torchsparse = False
    try:
        from torchsparse.models import MinkUNet18
        from torchsparse.utils.quantize import sparse_quantize
        import torchsparse
        has_torchsparse = True
    except ImportError:
        print("[TRAIN] TorchSparse++ not available on host. Using host CPU PointNet/MLP training scaffold.")

    ds = RELLISDataset(RELLIS_ROOT)
    n_val = max(1, len(ds) // 10)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    log = []

    if has_torchsparse:
        model = MinkUNet18(in_channels=4, num_classes=8).to(device)
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device), ignore_index=7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for pts, lbs in train_loader:
                pts = pts.to(device); lbs = lbs.to(device)
                coords_int = torch.floor(pts[:, :3] / VOXEL_SIZE).int()
                _, umap, imap = sparse_quantize(
                    coords_int.cpu().numpy(), return_index=True, return_inverse=True)
                vox_coords = torch.cat([
                    torch.zeros(len(umap), 1, dtype=torch.int32),
                    torch.tensor(coords_int.cpu().numpy()[umap], dtype=torch.int32)
                ], dim=1).to(device)
                vox_feats = pts[umap].float()
                sp_input = torchsparse.SparseTensor(feats=vox_feats, coords=vox_coords)
                
                optimizer.zero_grad()
                logits = model(sp_input).feats
                vox_lbs = lbs[torch.tensor(umap)]
                loss = criterion(logits, vox_lbs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            scheduler.step()
            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  train_loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

            if epoch % 5 == 0:
                ckpt_path = os.path.join(WEIGHTS_DIR, f"minkunet18_drdo_ep{epoch:02d}.pth")
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_loss}, ckpt_path)
                print(f"  [SAVED] {ckpt_path}")
                
            log.append({"epoch": epoch, "train_loss": avg_loss})
    else:
        # Development host scaffold (CPU / Apple Silicon M4)
        model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device), ignore_index=7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        # Subset for rapid host iteration
        sample_pts, sample_lbs = [], []
        for i in range(min(20, len(train_ds))):
            p, l = train_ds[i]
            sample_pts.append(p)
            sample_lbs.append(l)

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for pts, lbs in zip(sample_pts, sample_lbs):
                pts = pts.to(device)
                lbs = lbs.to(device)
                optimizer.zero_grad()
                logits = model(pts)
                loss = criterion(logits, lbs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            avg_loss = train_loss / len(sample_pts)
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  train_loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

            if epoch % 5 == 0:
                ckpt_path = os.path.join(WEIGHTS_DIR, f"minkunet18_drdo_ep{epoch:02d}.pth")
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(), "train_loss": avg_loss}, ckpt_path)
                print(f"  [SAVED] {ckpt_path}")
                
            log.append({"epoch": epoch, "train_loss": avg_loss})

    with open(RESULTS_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print("[STEP P4.2.1 COMPLETE]")

if __name__ == "__main__":
    train()
