# %%
import math
import torch
from math import exp

# %%
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[2,2],
    stride=1, padding=0, bias=True, padding_mode='zeros') # device=None, dtype=None

conv1.weight.data.fill_(0.1)
conv1.bias.data.fill_(0.5)

in1 = torch.Tensor([[[1, 3],[5, 7]], [[2, 4],[6, 8]]])
in1 = torch.reshape(in1, (2, 1, 2, 2)) # batch_size + in_filters

out_conv = conv1(in1)
assert abs(float(out_conv[0,0,0,0]) - (0.1*1 + 0.1*3 + 0.1*5 + 0.1*7 + 0.5) ) < 0.001

activation_layer = torch.nn.ReLU()
assert out_conv[0,0,0,0] == activation_layer(out_conv[0,0,0,0]) # out>0

# %%
model = torch.nn.Sequential(
    conv1, 
    torch.nn.Softmax(dim=0)
)

optimizer = torch.optim.SGD(
        model.parameters(), lr = 0.02,
        momentum=0, weight_decay=0)

out_softmax = model(in1)
out_s1, out_s2 = float(out_softmax[0,0,0,0]), float(out_softmax[1,0,0,0])
out_conv1, out_conv2 = float(out_conv[0,0,0,0]), float(out_conv[1,0,0,0])

assert abs( out_s1 - exp(out_conv1)/(exp(out_conv1) + exp(out_conv2)) ) < 0.0001

out_softmax = out_softmax.reshape([1,2])

# %%
target = torch.tensor([1])
loss = torch.nn.CrossEntropyLoss()(out_softmax, target)

# TODO: Shouldn't the output be ?:
# assert loss == - math.log(float(out_softmax[0,1]))

# %%
optimizer.zero_grad()
loss.backward() # TODO Where are
# %%
# TODO why is conv1.weight.grad: -0.2166
conv1.weight.grad
# %%
optimizer.step()
# %%
conv1.weight

# %%
