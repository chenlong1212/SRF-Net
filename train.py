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

from models import model_dict  # LSTM/GRU/TCN/STGCN/Transformer


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
class PingpangDataset(Dataset):
    def __init__(self, files, labels, modality='skeleton', augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment
        self.modality = modality

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.modality == 'skeleton':
            x = np.load(self.files[idx]).astype(np.float32)
            if self.augment:
                x = augment_skeleton(x)
        elif self.modality == 'racket':
            x = np.genfromtxt(self.files[idx], delimiter=',', skip_header=1, dtype=np.float32)
            if self.augment:
                x = augment_racket(x)
        if x.shape[0] > 60:
            x = x[:60]
        elif x.shape[0] < 60:
            pad = np.zeros((60-x.shape[0], x.shape[1]), dtype=x.dtype)
            x = np.vstack([x, pad])
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y,dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max([x.shape[0] for x in xs])
    feat_dim = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, feat_dim)
    for i,x in enumerate(xs):
        padded[i,:x.shape[0],:] = x
    return padded, torch.tensor(ys)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*x.size(0)
    return total_loss / len(loader.dataset)

def compute_metrics(model, loader, device, criterion):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs,y)
            total_loss += loss.item()*x.size(0)
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
    parser.add_argument('--type', type=str, default='racket', choices=['skeleton','racket'])
    parser.add_argument('--model', type=str, default='STGCN') # LSTM GRU TCN Transformer STGCN
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    if args.type == 'skeleton':
        data_dir = 'data/racket_csv'
        D = 132
    else:
        data_dir = 'data/skeleton_npy'
        D = 7

    data_dir = Path(data_dir)
    files, labels = [], []
    for cls_dir in sorted(data_dir.iterdir()):
        if cls_dir.is_dir():
            ext = "*.npy" if args.type=="skeleton" else "*.csv"
            for f in cls_dir.glob(ext):
                files.append(str(f))
                labels.append(cls_dir.name)
    assert len(files) > 0, f"no data found in {data_dir}"

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print("Classes:", le.classes_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)

    # ----------------  8:1:1 ----------------
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=seed
    )

    print(f"split: train {len(train_files)}, test {len(test_files)}, val {len(val_files)}")

    train_ds = PingpangDataset(train_files, train_labels, modality=args.type, augment=True)
    val_ds = PingpangDataset(val_files, val_labels, modality=args.type, augment=False)
    test_ds = PingpangDataset(test_files, test_labels, modality=args.type, augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)


    model_cls = model_dict[args.model]
    model = model_cls(input_dim=D, hidden_dim=16, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

 
    os.makedirs("save", exist_ok=True)
    save_csv = f"save/{args.model}_{args.type}_metrics.csv"
    save_model = f"save/best.pth" 
    print(f"model: {args.model}, modality: {args.type}")


    with open(save_csv,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','train_acc','train_prec','train_rec','train_f1',
                         'test_loss','test_acc','test_prec','test_rec','test_f1'])
        best_test_acc = 0.0

        for epoch in range(1, args.epochs+1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            train_acc, train_prec, train_rec, train_f1, _ = compute_metrics(model, train_loader, device, criterion)
     
            val_acc, _, _, _, val_loss = compute_metrics(model, val_loader, device, criterion)
            test_acc, test_prec, test_rec, test_f1, test_loss = compute_metrics(model, test_loader, device, criterion)

            print(f"Epoch {epoch}: "
                  f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
                  f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")

            # writer.writerow([epoch,
            #                  f"{train_loss:.4f}", f"{train_acc:.4f}", f"{train_prec:.4f}", f"{train_rec:.4f}", f"{train_f1:.4f}",
            #                  f"{test_loss:.4f}", f"{test_acc:.4f}", f"{test_prec:.4f}", f"{test_rec:.4f}", f"{test_f1:.4f}"])

            scheduler.step()

        torch.save(model.state_dict(), save_model)

if __name__ == '__main__':
    main()
