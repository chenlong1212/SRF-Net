import os
import argparse
import random
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

from models import model_dict  # LSTM / GRU / TCN / STGCN / Transformer


seed = 42

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def augment_skeleton(x):
    T, D = x.shape
    C = 2
    J = D // C
    x = x.reshape(T, J, C)

    angle = np.random.uniform(-30, 30)/180*np.pi
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    x[..., :2] = x[..., :2] @ rotation.T

    if np.random.rand() < 0.5:
        x[..., 0] = -x[..., 0]

    if np.random.rand() < 0.5:
        num_drop = max(1, J//10)
        drop_joints = np.random.choice(J, num_drop, replace=False)
        x[:, drop_joints, :] = 0

    noise_std = np.random.uniform(0.01,0.05)
    x += np.random.normal(0, noise_std, size=x.shape)

    if T > 10 and np.random.rand() < 0.5:
        factor = np.random.uniform(0.9,1.1)
        new_T = max(1,int(T*factor))
        x_idx = np.linspace(0,T-1,new_T)
        x_interp = np.zeros((new_T,J,C))
        for j in range(J):
            for c in range(C):
                x_interp[:,j,c] = np.interp(x_idx,np.arange(T),x[:,j,c])
        x = x_interp

    scale = np.random.uniform(0.85,1.15)
    x *= scale
    x = x.reshape(x.shape[0], -1)
    return x

def augment_racket(x):
    T,D = x.shape
    x += np.random.normal(0,0.01,size=x.shape)
    if T > 5 and np.random.rand() < 0.5:
        factor = np.random.uniform(0.9,1.1)
        new_T = max(1,int(T*factor))
        x_idx = np.linspace(0,T-1,new_T)
        x_interp = np.zeros((new_T,D))
        for d in range(D):
            x_interp[:,d] = np.interp(x_idx,np.arange(T),x[:,d])
        x = x_interp
    scale = np.random.uniform(0.9,1.1)
    x *= scale
    return x

# --------------------------- Dataset ---------------------------
class FusionDataset(Dataset):
    def __init__(self, skeleton_files, racket_files, labels, augment=False):
        self.skeleton_files = skeleton_files
        self.racket_files = racket_files
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # --- skeleton ---
        x_s = np.load(self.skeleton_files[idx]).astype(np.float32)
        if self.augment:
            x_s = augment_skeleton(x_s)

        # --- racket ---
        x_r = np.genfromtxt(self.racket_files[idx], delimiter=',', skip_header=1, dtype=np.float32)
        if self.augment:
            x_r = augment_racket(x_r)

        def pad_or_cut(x, target_len=60):
            if x.shape[0] > target_len:
                x = x[:target_len]
            elif x.shape[0] < target_len:
                pad = np.zeros((target_len - x.shape[0], x.shape[1]), dtype=x.dtype)
                x = np.vstack([x, pad])
            return x

        x_s = pad_or_cut(x_s, 60)
        x_r = pad_or_cut(x_r, 60)

        y = self.labels[idx]
        return torch.from_numpy(x_s), torch.from_numpy(x_r), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    xs, xr, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    feat_dim_s = xs[0].shape[1]
    feat_dim_r = xr[0].shape[1]

    padded_s = torch.zeros(len(xs), max_len, feat_dim_s)
    padded_r = torch.zeros(len(xs), max_len, feat_dim_r)
    for i,(x_s,x_r) in enumerate(zip(xs,xr)):
        padded_s[i,:x_s.shape[0],:] = x_s
        padded_r[i,:x_r.shape[0],:] = x_r
    return padded_s, padded_r, torch.tensor(ys)

class MidFusionModel(nn.Module):
    def __init__(self, base_model_name, hidden_dim, num_classes):
        super().__init__()
        self.skeleton_model = model_dict[base_model_name](input_dim=132, hidden_dim=hidden_dim, num_classes=num_classes)
        self.racket_model   = model_dict[base_model_name](input_dim=7, hidden_dim=hidden_dim, num_classes=num_classes)

        self.skeleton_model.fc = nn.Identity()
        self.racket_model.fc   = nn.Identity()

        with torch.no_grad():
            dummy_s = torch.zeros(1, 60, 132)
            dummy_r = torch.zeros(1, 60, 7)
            feat_s = self.skeleton_model(dummy_s)
            feat_r = self.racket_model(dummy_r)
            if feat_s.dim() == 3:
                feat_s = feat_s.mean(dim=1)
            if feat_r.dim() == 3:
                feat_r = feat_r.mean(dim=1)
            fused_dim = feat_s.shape[1] + feat_r.shape[1]

        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, x_s, x_r):
        feat_s = self.skeleton_model(x_s)
        feat_r = self.racket_model(x_r)
        if feat_s.dim() == 3:
            feat_s = feat_s.mean(dim=1)
        if feat_r.dim() == 3:
            feat_r = feat_r.mean(dim=1)
        feat = torch.cat([feat_s, feat_r], dim=1)
        return self.classifier(feat)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x_s,x_r,y in loader:
        x_s,x_r,y = x_s.to(device), x_r.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x_s,x_r)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x_s.size(0)
    return total_loss / len(loader.dataset)

def compute_metrics(model, loader, device, criterion):
    model.eval()
    y_true,y_pred = [],[]
    total_loss = 0
    with torch.no_grad():
        for x_s,x_r,y in loader:
            x_s,x_r,y = x_s.to(device), x_r.to(device), y.to(device)
            outputs = model(x_s,x_r)
            loss = criterion(outputs,y)
            total_loss += loss.item()*x_s.size(0)
            preds = outputs.argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred,average='macro',zero_division=0)
    rec = recall_score(y_true,y_pred,average='macro',zero_division=0)
    f1 = f1_score(y_true,y_pred,average='macro',zero_division=0)
    avg_loss = total_loss / len(loader.dataset)
    return acc, prec, rec, f1, avg_loss

# --------------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='STGCN_AG') # LSTM GRU TCN Transformer STGCN STGCN_TA STGCN_AG STGCN_AG_TA
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    skeleton_dir = Path('data/racket_csv')
    racket_dir   = Path('data/skeleton_npy')

    skeleton_files, racket_files, labels = [], [], []
    for cls_dir in sorted(skeleton_dir.iterdir()):
        if cls_dir.is_dir():
            racket_cls_dir = racket_dir / cls_dir.name
            for f in cls_dir.glob("*.npy"):
                skeleton_files.append(str(f))
                racket_file = racket_cls_dir / (f.stem + ".csv")
                racket_files.append(str(racket_file))
                labels.append(cls_dir.name)

    assert len(skeleton_files) == len(racket_files), "skeleton and racket files count mismatch"
    assert len(skeleton_files) > 0, "no data found"

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print("Classes:", le.classes_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    #  (8:1:1)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        np.arange(len(labels)), labels, test_size=0.2, stratify=labels, random_state=seed
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    train_skeleton = [skeleton_files[i] for i in train_files]
    train_racket   = [racket_files[i] for i in train_files]
    val_skeleton   = [skeleton_files[i] for i in val_idx]
    val_racket     = [racket_files[i] for i in val_idx]
    test_skeleton  = [skeleton_files[i] for i in test_idx]
    test_racket    = [racket_files[i] for i in test_idx]

    train_ds = FusionDataset(train_skeleton, train_racket, train_labels, augment=True)
    val_ds   = FusionDataset(val_skeleton, val_racket, val_labels, augment=False)
    test_ds  = FusionDataset(test_skeleton, test_racket, test_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = MidFusionModel(base_model_name=args.model, hidden_dim=16, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    os.makedirs("save", exist_ok=True)
    save_csv = f"save/{args.model}_fusion_metrics.csv"
    print(f"model: {args.model}")

    with open(save_csv,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','train_acc','train_prec','train_rec','train_f1',
                         'test_loss','test_acc','test_prec','test_rec','test_f1'])

        for epoch in range(1, args.epochs+1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            train_acc, train_prec, train_rec, train_f1, _ = compute_metrics(model, train_loader, device, criterion)
            val_acc, _, _, _, val_loss = compute_metrics(model, val_loader, device, criterion)
            test_acc, test_prec, test_rec, test_f1, test_loss = compute_metrics(model, test_loader, device, criterion)

            print(f"Epoch {epoch}: "
                  f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
                  f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")

            writer.writerow([epoch,
                             f"{train_loss:.4f}", f"{train_acc:.4f}", f"{train_prec:.4f}", f"{train_rec:.4f}", f"{train_f1:.4f}",
                             f"{test_loss:.4f}", f"{test_acc:.4f}", f"{test_prec:.4f}", f"{test_rec:.4f}", f"{test_f1:.4f}"])

            scheduler.step()

if __name__ == '__main__':
    main()
