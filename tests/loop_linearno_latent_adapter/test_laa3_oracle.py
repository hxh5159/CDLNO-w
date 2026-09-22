import math
import unittest
import torch
from .attention_oracle import attention_reference


class AttentionOracleTests(unittest.TestCase):
    def test_two_points_two_slots_hand_computed(self):
        eye = torch.eye(2, dtype=torch.float64)
        zero = torch.zeros(2, dtype=torch.float64)
        weights = {'in_project_x.weight': eye, 'in_project_x.bias': zero,
                   'to_q.weight': eye*0, 'to_k.weight': eye*0, 'to_v.weight': eye,
                   'to_out.0.weight': eye, 'to_out.0.bias': zero,
                   'temperature_q': torch.tensor([[[[.5]]]], dtype=torch.float64),
                   'temperature_k': torch.tensor([[[[.25]]]], dtype=torch.float64),
                   'adapter.A_q': eye, 'adapter.A_k': eye,
                   'adapter.B_q': eye, 'adapter.B_k': eye*.5}
        x = eye.unsqueeze(0)
        result, trace = attention_reference(x, weights, variant='temp', heads=1,
                                           round_index=1, adapter=True, alpha=2.)
        p = math.exp(2)/(math.exp(2)+1)
        expected = torch.tensor([[[p*p+(1-p)**2, 2*p*(1-p)],
                                   [2*p*(1-p), p*p+(1-p)**2]]], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-15, rtol=1e-15)
        torch.testing.assert_close(trace['queries'].sum(-1), torch.ones(1,1,2,dtype=torch.float64))
        torch.testing.assert_close(trace['keys'].sum(-2), torch.ones(1,1,2,dtype=torch.float64))
        wrong, _ = attention_reference(x, weights, variant='temp', heads=1,
                                       round_index=1, adapter=True, alpha=2., wrong_temperature_order=True)
        self.assertGreater((result-wrong).abs().max().item(), .1)

    def test_nonzero_latent_hand_computed(self):
        eye = torch.eye(2,dtype=torch.float64); zero = torch.zeros(2,dtype=torch.float64)
        weights = {'in_project_x.weight':eye,'in_project_x.bias':zero,
                   'to_q.weight':eye*0,'to_k.weight':eye*0,'to_v.weight':eye,
                   'to_out.0.weight':eye,'to_out.0.bias':zero,
                   'latent_processor.norm.weight':torch.ones(2,dtype=torch.float64),
                   'latent_processor.norm.bias':zero,
                   'latent_processor.linear1.weight':eye*0,'latent_processor.linear1.bias':torch.ones(2,dtype=torch.float64),
                   'latent_processor.linear2.weight':eye,'latent_processor.linear2.bias':zero}
        output,_ = attention_reference(eye.unsqueeze(0),weights,variant='plain',heads=1,round_index=0,latent=True)
        expected = .5 + .5*(1+math.erf(1/math.sqrt(2)))
        torch.testing.assert_close(output,torch.full((1,2,2),expected,dtype=torch.float64),atol=1e-15,rtol=1e-15)
