# -*- encoding:utf-8 -*-
# author: liuheng
import torch
from torch import nn, optim
import random
import numpy as np
from collections import Counter

FILE_PATH = "./ptb/ptb.train.txt"

# 读取文件
with open(file=FILE_PATH) as f:
    text = f.read()

# 预处理
def preprocess(text, freq=5):
    text = text.lower()
    text = text.replace(".", "<PERIO>")
    words = text.split()
    word_counts = Counter(words)
    trimmed_word = [word for word in words if word_counts[word] > freq]
    return trimmed_word

# 准备字典
words = preprocess(text)
vocab = set(words)
vocab2int = {w: c for c, w in enumerate(vocab)}
int2vocab = {c: w for w, c in enumerate(vocab)}
int_words = [vocab2int[word] for word in words]
# print(int_words)

t = 1e-5

int_word_counts = Counter(int_words)
total_count = len(int_word_counts)
word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
prob_drop = {w: 1-np.sqrt(t / word_freqs[w]) for w in int_word_counts}
train_words = [w for w in int_words if random.random()<(1-prob_drop[w])]

# 获取targets
def get_target(words, idx, window_size=5):
    target_window = np.random.randint(1, window_size + 1)
    start_int = idx - window_size if idx - window_size > 0 else 0
    end_int = start_int + window_size
    targets = set(words[start_int:idx] + words[idx:end_int + 1])
    return list(targets)

# 构造batch迭代器
def get_batch(words, batch_size, window_size):
    n_batches = len(words) // batch_size
    words = words[:n_batches * batch_size]
    for idx in range(0, len(words), batch_size):
        batch_x, batch_y = [], []
        batch_center = words[idx:idx+batch_size]
        for i in range(len(batch_center)):
            x = batch_center[i]
            y = get_target(words, i, window_size)
            batch_x.extend([x] * len(y))
            batch_y.extend(y)
        yield batch_x, batch_y

class SkipGramNeg(nn.Module):

    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist

        # 定义embedding层
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        # 初始化
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, input_words):
        return self.in_embed(input_words)

    def forward_outout(self, output_words):
        return self.out_embed(output_words)

    def forward_noise(self, batch_size, n_samples):
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist

        noise_words = torch.multinomial(noise_dist, batch_size * n_samples, replacement=True)

        noise_vectors = self.out_embed(noise_words).reshape(batch_size, n_samples, self.n_embed)

        return noise_vectors

# 负采样
word_freqs = np.array(word_freqs.values())
unigram_dist = word_freqs / word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist ** 0.75) / np.sum(unigram_dist ** 0.75)

# 构造损失函数
class NegativeSamplingLoss(nn.Module):

    def __init__(self):
        super(NegativeSamplingLoss, self).__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, embed_size = input_vectors.shape
        # 一个batch的列向量
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        # 一个batch的行向量
        output_vectors = output_vectors.view(batch_size, 1, embed_size)

        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss.squeeze_()

        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss.squeeze_().sum(dim=1)

        return -(out_loss + noise_loss).mean()

# 模型训练
def train():
    embedding_dim = 300
    model = SkipGramNeg(len(vocab2int), embedding_dim, noise_dist=noise_dist)

    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    print_every = 1500
    steps = 0
    epochs = 5
    batch_size = 500
    n_samples = 5

    for e in range(epochs):
        for input_words, target_words in get_batch(train_words, batch_size):
            steps += 1
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)

            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_outout(targets)
            noise_vectors = model.forward_noise(batch_size, n_samples)

            loss = criterion(input_vectors, output_vectors, noise_vectors)
            if steps // print_every == 0:
                print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



