import torch
import frnn
import torch.utils.benchmark as benchmark

# Sample 2 long array of 3D points
a = torch.randn((1, 1000000, 3), device="cuda")
b = torch.randn((1, 400000, 3), device="cuda")

# idxs, counts, _, _ = frnn.frnn_grid_resampling(a, b, K=50, r=0.1)
idxs, counts, _ = frnn.frnn_bf_resampling(a, b, K=50, r=0.1)
counts = counts.view(-1)
counts = counts.cpu().numpy()
print("Counts shape:", counts.shape)
print("Counts:", counts)

# Count how many exceed 50
exceed_50 = (counts > 50).sum()
print("Number of counts exceeding 50:", exceed_50.item())
# Count how many are less than 50
less_than_50 = (counts < 50).sum()
print("Number of counts less than 50:", less_than_50.item())

pts = frnn.frnn_gather(b, idxs)
print(pts)

# Matplotlib to visualize the count distribution as a histogram
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
bins = plt.hist(counts, bins=51, color='blue', alpha=0.7, range=(-1, 50))
print("Histogram bins:", bins)
plt.title('Histogram of Counts')
plt.xlabel('Counts')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


# t0 = benchmark.Timer(
#     stmt='frnn.frnn_bf_points(a, b, K=50, r=0.1)',
#     setup='import frnn',
#     globals={'a': a, 'b': b})

# t1 = benchmark.Timer(
#     stmt='frnn.frnn_bf_resampling(a, b, K=50, r=0.1)',
#     setup='import frnn',
#     globals={'a': a, 'b': b})

# t2 = benchmark.Timer(
#     stmt='frnn.frnn_grid_points(a, b, K=50, r=0.1)',
#     setup='import frnn',
#     globals={'a': a, 'b': b})

# print(t0.timeit(10))
# print(t1.timeit(10))
# print(t2.timeit(10))
