"""CPU/reference agreement; optional real CUDA parity when available."""
import unittest
import numpy as np
import torch
import evaluate_gpu as ev

class PatchTests(unittest.TestCase):
 def test_reference_and_cuda(self):
  torch.set_num_threads(2)
  for arch in ev.core.ARCHS:
   torch.manual_seed(123);m=ev.core.Network(arch).eval();H=torch.rand(8,16,32,32)
   with torch.no_grad():L=m.tail(H)
   nc=2;orders=np.tile(np.arange(16),(nc,1));rg=ev.core.rng(40,2000,1);ros=[np.stack([rg.permutation(16) for _ in range(nc)]) for _ in range(8)]
   ref=ev.core.patch_eval(m,H,L,orders,'two_concepts',ros);pairs=ev.pair_tensors('two_concepts',len(H),'cpu');b,c,g=pairs;p0=L[b].softmax(1);p1=L[c].softmax(1);good=(L[b].argmax(1)==b%4)&(L[c].argmax(1)==c%4)
   for k in [0,1,4,8,16]:
    prob=ev.patch(m,H,orders,k,pairs,3)
    if k==0:self.assertTrue(torch.allclose(prob,p0,atol=1e-6))
    elif k==16:self.assertTrue(torch.allclose(prob,p1,atol=1e-6))
    else:
     got=ev.measure(prob,p0,p1,c%4,good)
     self.assertAlmostEqual(got['fidelity'],ref['selected'][str(k)]['fidelity'],places=4)
    if torch.cuda.is_available():
     gpu=ev.core.Network(arch).cuda().eval();gpu.load_state_dict(m.state_dict());gp=ev.patch(gpu,H.cuda(),orders,k,ev.pair_tensors('two_concepts',len(H),'cuda'),3)
     self.assertTrue(torch.allclose(gp.cpu(),prob,atol=2e-5,rtol=1e-4))
 def test_undefined_effect(self):
  p=torch.ones(4,4)/4;r=ev.measure(p,p,p,torch.arange(4),torch.zeros(4,dtype=torch.bool))
  self.assertIsNone(r['fidelity']);self.assertIsNone(r['cf_accuracy_both_correct'])
if __name__=='__main__':unittest.main()
