import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._logging
import logging

from torch._inductor.debug import DebugContext
torch._inductor.config.trace.enabled = True

torch._logging.set_logs(
            dynamo=logging.DEBUG,
            aot=logging.DEBUG,
            inductor=logging.DEBUG,
            schedule=True,
            output_code=True,  # 默认是关闭的
        )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.softmax(x)    
        out = F.dropout(out, p=0.2, training=True)

        return out
    
net = Net()
net = torch.compile(net)

x = torch.rand(128, 12, 128, 128, dtype=torch.float16).cuda()
with DebugContext():  # <----- 保存所有的中间代码到当前 torch_compile_debug 目录下
    out = net(x)
