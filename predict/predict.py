import numpy as np
import pandas as pd
import torch
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, LogitsConfig
import time
import re
import argparse
import torch.nn.functional as F
import torch.nn as nn


def pad_or_truncate(feat):
    L = feat.shape[0]
    max_len = 7200
    if L < max_len:
        pad = max_len - L
        feat = F.pad(feat, (0, 0, 0, pad))
    else:
        feat = feat[:max_len]
    return feat

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

# load the trained model
model.load_state_dict(torch.load("model.pth"))
model.eval()

parser = argparse.ArgumentParser(description="A simple command line argument example")
parser.add_argument("-str", "--str", type=str, help="Seq")  # optional flag
args = parser.parse_args()

with open(f'example.fasta', 'r') as f:
    lines = f.readlines()

Sequence = []
for line in lines:
    if line.startswith('>'):
        continue
    line = re.sub(r'\s+', '', line)  # Remove all whitespace characters
    Sequence.append(line)

forge_client = ESM3ForgeInferenceClient(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai", token="Hn4AZibnSDbmF3gDqc6up")
print(f"Number of sequences: {len(Sequence)}")

for ind in range(len(Sequence)):

    print(f"Processing sequence {ind+1}, length: {len(Sequence[ind])}")
    seq = Sequence[ind]

    # Handle long sequences by splitting into two parts if >2046
    if len(seq) > 2046:
        chunk_size = 2046
        part_embeddings = []

        # Split sequence into chunks of at most 2046
        for i in range(0, len(seq), chunk_size):
            part = seq[i:i+chunk_size]
            protein = ESMProtein(sequence=part)
            protein_tensor = forge_client.encode(protein)
            logits_output = forge_client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            emb = logits_output.embeddings[0].to(torch.float32).cpu().numpy()[1:-1]
            part_embeddings.append(emb)

        # concat all chunk embeddings
        embeddings = np.concatenate(part_embeddings, axis=0)
        print(f"Concatenated embeddings shape before final averaging: {embeddings.shape}")

    else:

        protein = ESMProtein(sequence=seq)
        protein_tensor = forge_client.encode(protein)
        logits_output = forge_client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        emb = logits_output.embeddings[0].to(torch.float32).cpu().numpy()[1:-1]
        embeddings = emb
        print(f"Embeddings shape for sequence {ind+1}: {embeddings.shape}")

    feats = pad_or_truncate(torch.from_numpy(embeddings)).unsqueeze(0)  # (1, L, 2560)
    feats = feats.to(device)
    outputs = model(feats)
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    print(f"Predicted LLPS probability for sequence {ind+1}: {probs[0]:.4f}")

