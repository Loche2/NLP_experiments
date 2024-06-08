import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tarfile
import os

# 不需要再下载数据集，只解压已下载的数据集
def extract_dataset(tar_path, extract_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_path)

# 显示数据集样例
def display_sample_data(file_path, num_samples=5):
    with open(file_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if 0 < i < num_samples:
                tokens, labels = line.strip().split('\t')
                print(f"{i}: {tokens}")
                print(f"   {labels}")

# 将tokens转换为ids
def convert_tokens_to_ids(tokens, vocab, oov_token=None):
    oov_id = vocab.get(oov_token) if oov_token else None
    return [vocab.get(token, oov_id) for token in tokens]

# 加载词典
def load_dict(dict_path):
    vocab = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            key = line.strip('\n')
            vocab[key] = i
    return vocab

# 自定义数据集
class NERDataset(Dataset):
    def __init__(self, datafiles):
        self.data = []
        self.read_data(datafiles)

    def read_data(self, datafiles):
        if isinstance(datafiles, str):
            datafiles = [datafiles]
        for datafile in datafiles:
            with open(datafile, 'r', encoding='utf-8') as fp:
                next(fp)
                for line in fp.readlines():
                    words, labels = line.strip('\n').split('\t')
                    words = words.split('\002')
                    labels = labels.split('\002')
                    self.data.append((words, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 数据转换和批处理
def collate_fn(batch):
    token_ids = [convert_tokens_to_ids(item[0], word_vocab, 'OOV') for item in batch]
    label_ids = [convert_tokens_to_ids(item[1], label_vocab, 'O') for item in batch]
    lengths = [len(ids) for ids in token_ids]

    max_len = max(lengths)
    padded_token_ids = [ids + [word_vocab['OOV']] * (max_len - len(ids)) for ids in token_ids]
    padded_label_ids = [ids + [label_vocab['O']] * (max_len - len(ids)) for ids in label_ids]

    return torch.tensor(padded_token_ids), torch.tensor(lengths), torch.tensor(padded_label_ids)

# 定义BiGRU+CRF模型
class BiGRUWithCRF(nn.Module):
    def __init__(self, emb_size, hidden_size, word_num, label_num):
        super(BiGRUWithCRF, self).__init__()
        self.word_emb = nn.Embedding(word_num, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, label_num)
        self.crf = CRF(label_num)

    def forward(self, x, lens):
        embs = self.word_emb(x)
        lens = lens.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embs, lens, batch_first=True, enforce_sorted=False)
        output, _ = self.gru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.fc(output)
        return output

# CRF层
class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return log_likelihood

    def _compute_log_likelihood(self, emissions, tags, mask):
        batch_size, seq_length, num_tags = emissions.size()
        alpha = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            emission_scores = emissions[:, i].unsqueeze(2)
            transition_scores = self.transitions.unsqueeze(0)
            broadcast_alpha = alpha.unsqueeze(1)
            inner = broadcast_alpha + emission_scores + transition_scores
            alpha = torch.logsumexp(inner, dim=1)

        log_likelihood = torch.logsumexp(alpha + self.end_transitions, dim=1)
        return log_likelihood.sum()

    def decode(self, emissions, mask):
        batch_size, seq_length, num_tags = emissions.size()
        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = []

        for i in range(1, seq_length):
            viterbi_t = []
            backpointers_t = []
            for tag in range(num_tags):
                max_tr_prob = viterbi + self.transitions[tag]
                best_tag_id = torch.argmax(max_tr_prob, dim=1)
                viterbi_t.append(max_tr_prob[range(batch_size), best_tag_id])
                backpointers_t.append(best_tag_id)
            viterbi = torch.stack(viterbi_t, dim=1) + emissions[:, i]
            backpointers.append(torch.stack(backpointers_t, dim=1))

        terminal_viterbi = viterbi + self.end_transitions
        best_tag_id = torch.argmax(terminal_viterbi, dim=1)

        best_path = [best_tag_id]
        for backpointers_t in reversed(backpointers):
            best_tag_id = torch.gather(backpointers_t, 1, best_tag_id.unsqueeze(1)).squeeze(1)
            best_path.insert(0, best_tag_id)

        return torch.stack(best_path, dim=1)

# 定义损失函数
class CRFLoss(nn.Module):
    def __init__(self, crf):
        super(CRFLoss, self).__init__()
        self.crf = crf

    def forward(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask)

# 训练和评估函数
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    for batch in data_loader:
        tokens, lens, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(tokens, lens)
        mask = (tokens != word_vocab['OOV']).to(device)
        loss = criterion(outputs, labels, mask)
        loss.backward()
        optimizer.step()

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            tokens, lens, labels = [x.to(device) for x in batch]
            outputs = model(tokens, lens)
            mask = (tokens != word_vocab['OOV']).to(device)
            pred = model.crf.decode(outputs, mask)
            loss = crf_loss(outputs, labels, mask)
            total_loss += loss.item()

            # 计算准确率
            for i in range(len(labels)):
                true_labels = labels[i][:lens[i]]
                pred_labels = pred[i][:lens[i]]
                correct += (true_labels == pred_labels).sum().item()
                total += lens[i].item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    print(f"Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# 主脚本
if __name__ == "__main__":
    data_path = 'data'
    tar_path = 'waybill.tar.gz'

    # 解压已下载的数据集
    extract_dataset(tar_path, data_path)

    # 显示数据集样例
    display_sample_data('data/train.txt')

    # 加载词典
    label_vocab = load_dict(os.path.join(data_path, 'tag.dic'))
    word_vocab = load_dict(os.path.join(data_path, 'word.dic'))

    # 加载数据集
    train_ds = NERDataset(os.path.join(data_path, 'train.txt'))
    dev_ds = NERDataset(os.path.join(data_path, 'dev.txt'))
    test_ds = NERDataset(os.path.join(data_path, 'test.txt'))

    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = BiGRUWithCRF(300, 300, len(word_vocab), len(label_vocab)).to(device)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    crf_loss = CRFLoss(network.crf)

    # 训练和评估模型
    for epoch in range(1):
        train_epoch(network, train_loader, optimizer, crf_loss, device)
        evaluate(network, dev_loader, device)

    evaluate(network, test_loader, device)
