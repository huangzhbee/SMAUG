import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import random
import tqdm
import os
import setproctitle
from dataset import get_dataloader
import pickle

setproctitle.setproctitle("hzh")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class AppSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)


def generate_pairs(batch_sequences, device):
    batch_size = len(batch_sequences)
    positive_pairs = []
    negative_pairs = []
    labels = []

    for i in range(batch_size):
        seq = batch_sequences[i]
        pos_seq = torch.roll(seq, shifts=random.randint(1, 9), dims=0)  # 数据增强，生成正样本对
        positive_pairs.append((seq, pos_seq))
        labels.append(0)

        neg_idx = (i + random.randint(1, batch_size - 1)) % batch_size
        neg_seq = batch_sequences[neg_idx]
        negative_pairs.append((seq, neg_seq))
        labels.append(1)

    pairs = positive_pairs + negative_pairs
    labels = torch.tensor(labels, dtype=torch.float32, device=device)
    return torch.stack([pair[0] for pair in pairs]), torch.stack([pair[1] for pair in pairs]), labels


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.norm(output1 - output2, p=2, dim=1)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B = x.size(0)
        outputs = []
        for b in range(B):
            mask = (x[b].sum(dim=1) != 0)
            filtered_x = x[b][mask]
            _, (hn, _) = self.lstm(filtered_x.unsqueeze(0))
            hn = hn[-1]
            out = self.fc(hn)
            outputs.append(out)
        outputs = torch.stack(outputs).squeeze(1)
        return outputs


# sequences = np.random.randn(1000, 1856, 32)  # 1000个样本，每个样本是10个时间步，每个时间步有32个特征
# dataset = AppSequenceDataset(sequences)
#
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

train_loader, val_loader, all_loader = get_dataloader(seed=1, batch_size=32)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMEncoder(input_size=16, hidden_size=64, num_layers=2, output_size=32).to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
patience = 5  # 如果验证损失在指定的epoch数量中没有改善，提前停止训练
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in tqdm.tqdm(range(num_epochs)):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        batch_emb = torch.tensor(batch["app_seq_emb"]).to(device).float()
        seq1, seq2, labels = generate_pairs(batch_emb, device)

        outputs1 = model(seq1)
        outputs2 = model(seq2)

        loss = criterion(outputs1, outputs2, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch_emb = torch.tensor(batch["app_seq_emb"]).to(device).float()
            seq1, seq2, labels = generate_pairs(batch_emb, device)

            outputs1 = model(seq1)
            outputs2 = model(seq2)

            loss = criterion(outputs1, outputs2, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 检查是否需要早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # torch.save(model.state_dict(), 'best_model.pt')  # 保存最佳模型
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping')
            break

# 获取表征向量
model.eval()
embeds = []
idxs = []
with torch.no_grad():
    for batch in all_loader:
        batch_emb = torch.tensor(batch["app_seq_emb"]).to(device).float()
        seq_idx = batch["seq_idx"]
        embeddings = model(batch_emb)
        embeds.append(embeddings.cpu().numpy())
        idxs.append(seq_idx.cpu().numpy())
embeds = np.concatenate(embeds, axis=0)
idxs = np.concatenate(idxs, axis=0)
with open("session_emb.pk", "wb") as f:
    pickle.dump(
        [idxs, embeds],
        f
    )
