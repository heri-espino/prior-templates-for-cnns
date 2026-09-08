import numpy as np
from analyze_study import estimate,holm
assert np.allclose(holm([.01,.04,.03,.2]),[.04,.09,.09,.2])
x=np.array([-.2,-.1,0,.1,.2]);e=estimate(x);assert e['n']==5 and abs(e['mean'])<1e-10 and e['lo']<0<e['hi']
p0=np.array([[.8,.2]]);p1=np.array([[.2,.8]]);den=((p1-p0)**2).sum()
def f(p):return 1-((p-p1)**2).sum()/den
assert np.isclose(f(p0),0) and np.isclose(f(p1),1) and np.isclose(f((p0+p1)/2),.75)
# Unequal minibatches must not have equal statistical weight.
correct=np.r_[np.ones(128),np.zeros(8)];assert np.isclose(correct.mean(),128/136) and not np.isclose(correct.mean(),.5)
print('Holm, paired summary, fidelity identities and weighting checks passed.')
