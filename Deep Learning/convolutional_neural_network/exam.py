import torch
import numpy as np

x = np.zeros([4, 4])
x[1, 1] = x[2, 2] = 1
x = x.reshape([1, *x.shape])
x = torch.tensor(x, requires_grad=True)

w1 = np.zeros([3, 3])
w1[1, 0], w1[1, 1] = -1, 1
w1 = w1.reshape([1, 1, *w1.shape])
w1 = torch.tensor(w1, requires_grad=True)

w2 = np.zeros([3, 3])
w2[2, 1], w2[1, 1] = -1, 1
w2 = w2.reshape([1, 1, *w2.shape])
w2 = torch.tensor(w2, requires_grad=True)
W = torch.tensor(np.eye(2), requires_grad=True)

f1 = torch.nn.functional.conv2d(x, w1)
f1r = torch.nn.functional.relu(f1)

f1m = torch.nn.functional.max_pool2d(f1r, [2, 2])
f2 = torch.nn.functional.conv2d(x, w2)

f2r = torch.nn.functional.relu(f2)
f2m = torch.nn.functional.max_pool2d(f2r, [2, 2])
h = torch.concat([f1m, f2m]).squeeze()
s = h @ W

for t in [f1, f2, h, s]:
    t.retain_grad()

L = torch.nn.functional.cross_entropy(s, torch.tensor(1))

L.backward()
for t in [s, h, W, f1, f2, w1, w2]:
    print(t.data, t.grad.data, sep='\n', end='\n\n')
