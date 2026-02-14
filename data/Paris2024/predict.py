import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model import SRF_Net 

def get_label_from_filename(filename):
    class_names = [
        "Forehand_Drive", "Backhand_Push", "Forehand_Loop", "Backhand_Chop",
        "Forehand_Chop", "Backhand_Drive", "Forehand_Flick", "Backhand_Loop",
        "Forehand_Push", "Banana_Flick"
    ]
    for cls in class_names:
        if filename.startswith(cls):
            return class_names.index(cls)
    return -1

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SRF_Net(num_classes=10, channels=3)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    test_data_path = "./test_data_npy/" 
    files = [f for f in os.listdir(test_data_path) if f.endswith('.npy')]
    
    true_labels = []
    pred_labels = []
    
    class_names = [
        "Forehand_Drive", "Backhand_Push", "Forehand_Loop", "Backhand_Chop",
        "Forehand_Chop", "Backhand_Drive", "Forehand_Flick", "Backhand_Loop",
        "Forehand_Push", "Banana_Flick"
    ]

    print("-" * 90)
    print(f"{'File Name':<30} | {'True Label
