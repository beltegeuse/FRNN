import torch
import frnn
import torch.utils.benchmark as benchmark

# Sample 2 long array of 3D points
a = torch.randn((1, 400000, 3), device="cuda")
b = torch.randn((1, 1000000, 3), device="cuda")

t0 = benchmark.Timer(
    stmt='frnn.frnn_bf_points(a, b, K=50, r=0.1)',
    setup='import frnn',
    globals={'a': a, 'b': b})

t1 = benchmark.Timer(
    stmt='frnn.frnn_grid_points(a, b, K=50, r=0.1)',
    setup='import frnn',
    globals={'a': a, 'b': b})

print(t0.timeit(10))
print(t1.timeit(10))