class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[128, 12, 128, 128]"):
        # File: test_log.py:24 in forward, code: out = self.softmax(x)
        convert_element_type: "f32[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        amax: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "f16[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.float16);  div = None
        
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 12, 128, 128]" = torch.ops.prims.inductor_random.default([128, 12, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[128, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        
        # File: test_log.py:25 in forward, code: out = F.dropout(out, p=0.2, training=True)
        gt: "b8[128, 12, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.2);  convert_element_type_default = None
        mul: "f16[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(gt, convert_element_type_1);  gt = convert_element_type_1 = None
        mul_1: "f16[128, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul, 1.25);  mul = None
        return (mul_1,)
        