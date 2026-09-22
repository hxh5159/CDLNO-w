import unittest
import torch
from .core_oracle import loop_equations,route


class OracleTests(unittest.TestCase):
    def test_sr_rb_lb_small_arithmetic_independently(self):
        x=torch.full((1,2,2),3.,dtype=torch.float64)
        def raw(x,i,r,branch):return 2*x
        def receiver(sources,key):return route(sources,torch.zeros(2,dtype=x.dtype),torch.ones(2,dtype=x.dtype))
        for mode,expected in [('sr_1_over_r',48.),('rb_attnres',20.),('lb_attnres_1_over_r',10.)]:
            y,t=loop_equations(x,P=0,C=1,R=2,S=1,mode=mode,raw=raw,native=lambda x,g,i:x,receiver=receiver,finalize=lambda x:x)
            torch.testing.assert_close(y,torch.full_like(x,expected),atol=1e-14,rtol=1e-14)
            if mode=='rb_attnres':
                self.assertEqual([len(z['sources']) for z in t['routes']],[1,2,2,3,3])
                self.assertEqual([r['summary'][0,0,0].item() for r in t['rounds']],[15.,42.])
            if mode=='lb_attnres_1_over_r':
                self.assertEqual(t['rounds'][0]['delta'][0,0,0].item(),9.)
                self.assertEqual(t['rounds'][1]['entry'][0,0,0].item(),6.)

    def test_route_singleton_identity_and_source_axis(self):
        a=torch.arange(12,dtype=torch.float64).reshape(2,3,2)
        q=torch.zeros(2,dtype=a.dtype);scale=torch.ones_like(q)
        self.assertIs(route([a],q,scale)[0],a)
        y,w=route([a,a+2,a+4],q,scale)
        torch.testing.assert_close(y,a+2,atol=1e-14,rtol=1e-14)
        self.assertEqual(w.shape,(2,3,3))
