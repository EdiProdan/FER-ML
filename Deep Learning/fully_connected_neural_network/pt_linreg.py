import torch
import torch.optim as optim


a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2, 4])
Y = torch.tensor([3, 5, 1])

optimizer = optim.SGD([a, b], lr=0.1)

param_niter = 100
for i in range(1, param_niter+1):
    Y_ = a*X + b

    diff = (Y-Y_)

    loss = torch.mean(diff**2)

    loss.backward()

    optimizer.step()

    grad_a = torch.mean(2*diff*(-X))
    grad_b = torch.mean(2*diff*(-1))

    if i % 10 == 0:
        print(f'epoch: {i}')
        print(f'\tgrad_a: {grad_a}, grad_b: {grad_b}')
        print(f'\ta.grad: {a.grad}, b.grad: {b.grad}')

    optimizer.zero_grad()
