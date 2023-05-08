import torch
from torch.utils.data.sampler import WeightedRandomSampler

weights = []
for _ in range(100):
    weights.append(1.01)
for _ in range(2):
    weights.append(10.1)
for _ in range(1):
    weights.append(10.1)
for _ in range(200):
    weights.append(1.01)
weights = torch.tensor(weights, dtype=torch.float)
samples = torch.multinomial(weights, len(weights), replacement=True)
target_0_bias_0 = 0
target_1_bias_0 = 0
target_0_bias_1 = 0
target_1_bias_1 = 0
for sample in samples:
    i = sample.item()
    if i in range(100):
        target_0_bias_0 += 1
    elif i in range(100, 102):
        target_1_bias_0 += 1
    elif i in range(102, 103):
        target_0_bias_1 += 1
    elif i in range(103, 303):
        target_1_bias_1 += 1
print(samples)
print(target_0_bias_0, target_1_bias_0, target_0_bias_1, target_1_bias_1)
