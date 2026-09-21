import unittest

import torch
from torch import nn

from cdlno.linearno.attention import LinearNOAttention
from cdlno.linearno_loop.v2.attention import ContextLinearNOAttention
from cdlno.linearno_loop.v2.latent import LatentContextFFN
from test_lf2_core import tiny_core


def attention_oracle(module, x, latent):
    B,N,H=x.shape
    if module.variant in ("conv","conv_temp"):
        projected=module.in_project_x(x.transpose(1,2).reshape(B,H,module.H,module.W))
        features=projected.reshape(B,module.heads,module.dim_head,N).transpose(-1,-2)
    else:
        projected=module.in_project_x(x)
        features=projected.reshape(B,N,module.heads,module.dim_head).transpose(1,2)
        if module.variant=="airfrans": features=features.contiguous()
    q,k,v=module.to_q(features),module.to_k(features),module.to_v(features)
    if module.variant in ("temp","conv_temp"):
        q=q/module.temperature_q.clamp(.01,1.);k=k/module.temperature_k.clamp(.01,1.)
    elif module.variant=="shapenet":
        q=q/module.tempreature_q.clamp(.1,2.);k=k/module.tempreature_k.clamp(.1,2.)
    q=q.softmax(-1);k=k.softmax(-2)
    context=torch.einsum("bhnm,bhnd->bhmd",k,v)
    tokens=context.transpose(1,2).contiguous().reshape(B,module.rank,module.dim)
    processed=tokens+latent.linear2(torch.nn.functional.gelu(latent.linear1(
        torch.nn.functional.layer_norm(tokens,(module.dim,),latent.norm.weight,latent.norm.bias,1e-5))))
    context=processed.reshape(B,module.rank,module.heads,module.dim_head).transpose(1,2).contiguous()
    readout=torch.einsum("bhnm,bhmd->bhnd",q,context)
    return module.to_out(readout.transpose(1,2).reshape(B,N,module.dim))


class LF3LatentTests(unittest.TestCase):
    def make(self,variant,dtype=torch.double):
        kw=dict(heads=2,dim_head=4,rank=3,variant=variant,dropout=0.)
        if variant in ("conv","conv_temp"):kw.update(H=2,W=3)
        torch.manual_seed(4);native=LinearNOAttention(8,**kw).to(dtype)
        torch.manual_seed(9);enhanced=ContextLinearNOAttention(8,**kw).to(dtype)
        enhanced.load_state_dict(native.state_dict(),strict=True)
        latent=LatentContextFFN(8,11).to(dtype);latent.initialize_release_identity()
        return native,enhanced,latent,6 if "conv" in variant else 5

    def test_off_path_exact_and_zero_identity_six_variants(self):
        for variant in ("plain","temp","conv","conv_temp","airfrans","shapenet"):
            native,enhanced,latent,N=self.make(variant)
            x=torch.randn(2,N,8,dtype=torch.double,requires_grad=True)
            expected=native(x); actual=enhanced(x)
            self.assertTrue(torch.equal(expected,actual),variant)
            with_latent=enhanced.forward_with_context(x,latent)
            self.assertTrue(torch.equal(actual,with_latent),variant)
            g1=torch.autograd.grad(expected.sum(),x,retain_graph=True)[0]
            g2=torch.autograd.grad(with_latent.sum(),x)[0]
            self.assertTrue(torch.equal(g1,g2),variant)

    def test_nonzero_matches_independent_oracle_and_vjp(self):
        for variant in ("plain","temp","conv","conv_temp","airfrans","shapenet"):
            _,enhanced,latent,N=self.make(variant)
            torch.manual_seed(12);latent.linear2.weight.data.normal_(0,.03);latent.linear2.bias.data.normal_(0,.01)
            x=torch.randn(2,N,8,dtype=torch.double,requires_grad=True)
            expected=attention_oracle(enhanced,x,latent);actual=enhanced.forward_with_context(x,latent)
            self.assertTrue(torch.allclose(expected,actual,atol=1e-12,rtol=1e-12),variant)
            vector=torch.randn_like(actual)
            ge=torch.autograd.grad((expected*vector).sum(),x,retain_graph=True)[0]
            ga=torch.autograd.grad((actual*vector).sum(),x)[0]
            self.assertTrue(torch.allclose(ge,ga,atol=1e-12,rtol=1e-12),variant)

    def test_zero_init_gradient_staging_and_permutation(self):
        latent=LatentContextFFN(8,13).double();latent.initialize_release_identity()
        context=torch.randn(2,2,5,4,dtype=torch.double,requires_grad=True)
        out=latent(context);out.square().mean().backward()
        self.assertEqual(latent.linear1.weight.grad.count_nonzero(),0)
        self.assertEqual(latent.norm.weight.grad.count_nonzero(),0)
        self.assertGreater(latent.linear2.weight.grad.abs().sum(),0)
        permutation=torch.tensor([3,0,4,1,2])
        self.assertTrue(torch.equal(latent(context.detach()[:,:,permutation]),out.detach()[:,:,permutation]))

    def test_shape_dtype_and_no_token_pair_matrices(self):
        latent=LatentContextFFN(8,7).float();latent.initialize_release_identity()
        for shape in ((1,1,3,8),(2,2,7,4),(3,4,2,2)):
            x=torch.randn(*shape);self.assertEqual(latent(x).shape,shape)
        for bad in (torch.randn(2,3,8),torch.randn(2,2,3,3),torch.ones(2,2,3,4,dtype=torch.int64)):
            with self.assertRaises((ValueError,TypeError)):latent(bad)

    def test_core_install_is_rng_isolated_shared_by_round_and_active_in_all_residuals(self):
        for residual in ("sr_1_over_r","rb_attnres","lb_attnres_1_over_r"):
            torch.manual_seed(31)
            core=tiny_core(residual,C=2,R=3)
            core.core_ffn_mode="round_specific_latent"
            state_before=torch.random.get_rng_state().clone()
            core.install_latent_ffns(inner_width=9,feature_seed=777)
            self.assertTrue(torch.equal(state_before,torch.random.get_rng_state()))
            self.assertEqual(len(core.latent_ffns),2)
            self.assertFalse(any("latent_ffns.0" in key and "latent_ffns.1" in key for key in core.state_dict()))
            calls=[0,0];handles=[]
            for p,module in enumerate(core.latent_ffns):
                handles.append(module.register_forward_hook(lambda m,a,o,p=p:calls.__setitem__(p,calls[p]+1)))
            x=torch.randn(2,5,4,requires_grad=True);core(x).sum().backward()
            for handle in handles:handle.remove()
            self.assertEqual(calls,[3,3])
            self.assertGreater(sum(parameter.grad.abs().sum() for parameter in core.latent_ffns[0].linear2.parameters()),0)


if __name__=="__main__":unittest.main()
