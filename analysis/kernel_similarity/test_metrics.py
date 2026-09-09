"""Known-answer tests for assignment, polarity, centering and degeneracy."""
import unittest
import numpy as np
from analyze import compare, normalized

class KernelMetrics(unittest.TestCase):
    def setUp(self):
        self.t=np.array([[1.,-1.,0.,0.],[0.,0.,1.,-1.]])
    def test_permutation_and_affine_invariance(self):
        w=self.t[::-1]*7+3
        m,s,nearest,cols=compare(w,self.t,w)
        self.assertAlmostEqual(m['assignment_cosine'],1)
        np.testing.assert_array_equal(cols,[1,0])
        self.assertAlmostEqual(m['nearest_distance'],0,places=6)
    def test_duplicate_rows_penalized_only_by_assignment(self):
        w=np.array([self.t[0],self.t[0]])
        m,*_=compare(w,self.t,w)
        self.assertAlmostEqual(m['nearest_cosine'],1)
        self.assertAlmostEqual(m['assignment_cosine'],.5)
        self.assertEqual(m['nearest_template_count'],1)
    def test_polarity_is_not_discarded(self):
        m,s,*_=compare(-self.t,self.t,self.t)
        self.assertAlmostEqual(s[0,0],-1)
        self.assertAlmostEqual(m['nearest_cosine'],0)
    def test_zero_centered_norm_rejected(self):
        with self.assertRaises(ValueError):normalized(np.ones((2,4)))
    def test_assignment_matches_bruteforce(self):
        from itertools import permutations
        rng=np.random.default_rng(7);a=rng.normal(size=(4,9));b=rng.normal(size=(4,9))
        m,s,*_=compare(a,b,a)
        expected=max(s[np.arange(4),p].mean() for p in permutations(range(4)))
        self.assertAlmostEqual(m['assignment_cosine'],expected)
        self.assertGreaterEqual(m['matching_gap'],-1e-12)

if __name__=='__main__':unittest.main()
