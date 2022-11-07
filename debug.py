import torch
import torch.nn as nn

model = nn.Linear(16, 16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

while True:
    x = torch.randn(16, 16, requires_grad=True)
    y = model(x)
    loss = (x - y).abs().mean()
    grad, = torch.autograd.grad(loss, [y])

    optimizer.zero_grad()
    y.backward(grad)
    optimizer.step()

    print(loss)