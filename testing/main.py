import os
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef,
)
from sklearn.metrics import balanced_accuracy_score

def find_metrics(y_test, y_predict, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    acc = accuracy_score(y_test, y_predict)
    prec = tp / (tp + fp)
    f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    auPR = average_precision_score(y_test, y_proba[:, 1])
    bal_acc = balanced_accuracy_score(y_test, y_predict)

    return sensitivity, specificity, acc, prec, f1_score_1, mcc, auc, y_proba[:, 1], bal_acc, auPR

############################################
# Seed
############################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

############################################
# Dataset
############################################
class ProteinDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, max_len=7200):
        self.samples = []
        self.max_len = max_len

        for f in os.listdir(pos_dir):
            self.samples.append((os.path.join(pos_dir, f), 1))
        for f in os.listdir(neg_dir):
            self.samples.append((os.path.join(neg_dir, f), 0))

    def __len__(self):
        return len(self.samples)

    def pad_or_truncate(self, feat):
        L = feat.shape[0]
        if L < self.max_len:
            pad = self.max_len - L
            feat = F.pad(feat, (0, 0, 0, pad))
        else:
            feat = feat[:self.max_len]
        return feat

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = np.load(path)                      # (L, 1152)
        feat = torch.tensor(feat, dtype=torch.float32)
        feat = self.pad_or_truncate(feat)

        label = torch.tensor(label, dtype=torch.float32)

        return feat, label

############################################
# Model
############################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)

        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))

        out += identity
        return self.relu(out)


class CustomResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv1d(2560, 1560, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(8, 1560)
        self.pool = nn.MaxPool1d(2)

        self.conv0 = nn.Conv1d(1560, 1024, kernel_size=3, stride=1, padding=1)
        self.gn0 = nn.GroupNorm(8, 1024)
        self.pool0 = nn.MaxPool1d(2)

        self.conv1 = nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(8, 512)
        self.pool1 = nn.MaxPool1d(2)

        self.layer1 = ResidualBlock(512)

        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(8, 256)
        self.pool2 = nn.MaxPool1d(2)

        self.layer2 = ResidualBlock(256)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.gn3 = nn.GroupNorm(8, 128)

        self.layer3 = ResidualBlock(128)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, L, 1152) → (B, 1152, L)
        x = x.permute(0, 2, 1)

        x = self.pool(F.relu(self.gn(self.conv(x))))
        x = self.pool0(F.relu(self.gn0(self.conv0(x))))
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = self.layer1(x)

        x = self.pool2(F.relu(self.gn2(self.conv2(x))))
        x = self.layer2(x)

        x = F.relu(self.gn3(self.conv3(x)))
        x = self.layer3(x)

        x = self.avgpool(x).squeeze(-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomResNet().to(device)

test_dataset = ProteinDataset(
    "../ESMC_6B/test/positive",
    "../ESMC_6B/test/negative"
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# load the trained model
model.load_state_dict(torch.load("model.pth"))
model.eval()

all_probs = []
all_labels = []
with torch.no_grad():
    for feats, labels in test_loader:
        feats = feats.to(device)
        outputs = model(feats)
        probs = torch.sigmoid(outputs).cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels.numpy())


all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
y_proba = np.vstack((1 - all_probs, all_probs)).T
y_predict = (all_probs >= 0.5).astype(int)
sensitivity, specificity, acc, prec, f1_score_1, mcc, auc, _, bal_acc, auPR = find_metrics(all_labels, y_predict, y_proba)
with open("results.csv", "w") as f:
    f.write("SN,SP,BACC,ACC,PREC,F1,MCC,AUC,AUPR\n")
    # upto 3 decimal places
    f.write(f"{sensitivity:.3f},{specificity:.3f},{bal_acc:.3f},{acc:.3f},{prec:.3f},{f1_score_1:.3f},{mcc:.3f},{auc:.3f},{auPR:.3f}\n")

np.save("y_true.npy", all_labels)
np.save("y_proba.npy", all_probs)
np.save("y_predict.npy", y_predict)

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import csv

fpr, tpr, _ = roc_curve(np.array(all_labels), np.array(all_probs))

f = open("./outputs/fpr.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[fp] for fp in fpr])
f.close()

f = open("./outputs/tpr.csv", "w", newline="")
writer = csv.writer(f)
writer.writerows([[tp] for tp in tpr])
f.close()

precision, recall, _ = precision_recall_curve(np.array(all_labels), np.array(all_probs))

f_name = f'./outputs/precision.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[p] for p in precision])

f_name = f'./outputs/recall.csv'
with open(f_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[r] for r in recall])