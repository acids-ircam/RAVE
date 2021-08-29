#%%
import torch

torch.set_grad_enabled(False)
from fd.parallel_model.buffer_conv import CachedConv1d, Conv1d, AlignBranches
from fd.parallel_model.core import get_padding
import matplotlib.pyplot as plt

c1 = Conv1d(1, 1, 3, padding=get_padding(3))
c1.weight.data.copy_(torch.arange(3).reshape(1, 1, -1).float())
c1.bias.data.zero_()

c2 = Conv1d(1, 1, 5, stride=2, padding=get_padding(5, 2))
c2.weight.data.copy_(torch.arange(5).reshape(1, 1, -1).float())
c2.bias.data.zero_()

cc1 = CachedConv1d(1, 1, 3, padding=get_padding(3))
cc1.weight.data.copy_(torch.arange(3).reshape(1, 1, -1).float())
cc1.bias.data.zero_()

cc2 = CachedConv1d(1, 1, 5, stride=2, padding=get_padding(5, 2))
cc2.weight.data.copy_(torch.arange(5).reshape(1, 1, -1).float())
cc2.bias.data.zero_()

a = AlignBranches(c1, c2)
ca = AlignBranches(cc1, cc2)

x = torch.arange(32).reshape(1, 1, -1).float()
y1, y2 = a(x)

cy1 = []
cy2 = []
for _x in torch.split(x, 16, 1):
    _cy1, _cy2 = ca(_x)
    cy1.append(_cy1)
    cy2.append(_cy2)
cy1 = torch.cat(cy1, -1)
cy2 = torch.cat(cy2, -1)

y = y1 + y2.repeat_interleave(2).reshape_as(y1)
cy = cy1 + cy2.repeat_interleave(2).reshape_as(cy1)

plt.plot(y1.reshape(-1)[:-cc2.future_compensation])
plt.plot(cy1.reshape(-1)[cc2.future_compensation:])
plt.show()

plt.plot(y2.reshape(-1)[:-cc2.future_compensation // 2])
plt.plot(cy2.reshape(-1)[cc2.future_compensation // 2:])
plt.show()

plt.plot(y.reshape(-1)[:-cc2.future_compensation])
plt.plot(cy.reshape(-1)[cc2.future_compensation:])
plt.show()
# %%
