
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = True




isolate_fails_code_str = None



# torch version: 2.3.0
# torch cuda version: 12.1
# torch git version: 97ff6cfd9c86c5c09d7ce775ab64ec5c99230f5d


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Wed_Nov_22_10:17:15_PST_2023 
# Cuda compilation tools, release 12.3, V12.3.107 
# Build cuda_12.3.r12.3/compiler.33567101_0 

# GPU Hardware Info: 
# Tesla V100-SXM2-32GB : 2 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        amax = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None
        inductor_seeds_default = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.2);  convert_element_type_default = None
        mul = torch.ops.aten.mul.Tensor(gt, convert_element_type_1);  gt = convert_element_type_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, 1.25);  mul = None
        return (mul_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (128, 12, 128, 128), dtype=torch.float16, is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
