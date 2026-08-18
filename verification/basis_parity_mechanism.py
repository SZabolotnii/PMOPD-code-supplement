"""What causes the pairwise scheme to fail: degree of nonlinearity or basis PARITY?

Refuted hypothesis: It initially appeared that the undecided event stems from the magnitude
of nonlinearity: on a standard power basis the undecided fraction reached ~10%, whereas on
PATP with alpha=0.6 it was zero. However, at alpha=1, the PATP exponents are exactly 1, 2, 3
(identical nonlinearity to the power basis), yet undecided events remain virtually absent
(0.0007 vs 0.106).

Mechanism: PATP yields sgn(x)|x|^p -- ODD functions for any p.
The standard power basis contains x^2, x^4 -- EVEN functions. Even functions respond to |shift|
rather than shift, making the geometry in basis space non-monotonic with respect to signal level,
which twists pairwise metrics differently across pairs.

Verified via two protocols: fixed bases on representative scenarios and a random sweep over
22 configurations. Mixed parity produces undecided rates > 1% in 7 of 22 configurations;
an all-odd basis yields none.

Practical takeaway: signed (odd) bases -- including fractional powers -- largely eliminate
the incompleteness, whereas the classical power basis induces it.
"""
import sys; sys.path.insert(0,'verification')
import numpy as np
from p2_intransitivity import Scenario, pairwise_rules

SEED=20260817
BASES = {
    'power {x, x², x³}':             lambda x: np.stack([x, x**2, x**3], 1),
    'odd-only {x, x³, x⁵}':          lambda x: np.stack([x, x**3, x**5], 1),
    'signed PATP α=1 {x, sgn·x², x³}': lambda x: np.stack([x, np.sign(x)*x**2, x**3], 1),
    'even-heavy {x, x², x⁴}':        lambda x: np.stack([x, x**2, x**4], 1),
    'even-only {x², x⁴, x⁶}':        lambda x: np.stack([x**2, x**4, x**6], 1),
}
SCEN = [('M=5 close, distinct shapes',
         Scenario([0.0,0.04,1.28,1.39,1.45],[0.05,1.83,1.36,0.39,1.39])),
        ('M=4 moderate, distinct shapes',
         Scenario([0.0,1.5,3.0,4.5],[0.2,1.6,0.4,1.8]))]

def gap(sc, f, n_mom=250_000, n_ev=350_000, seed=SEED):
    rng=np.random.default_rng(seed); s=3
    Xw=np.concatenate([sc.sample(m,120_000//sc.M,rng) for m in range(sc.M)])
    P=f(Xw); mean=P.mean(0); C=np.cov(P,rowvar=False,ddof=1)
    w,V=np.linalg.eigh(C); W=V@np.diag(np.maximum(w,1e-300)**-0.5)@V.T
    tr=lambda x:(f(x)-mean)@W
    mus,Cs=[],[]
    for m in range(sc.M):
        Q=tr(sc.sample(m,n_mom,rng)); mus.append(Q.mean(0))
        Cs.append(np.cov(Q,rowvar=False,ddof=1))
    mus,Cs=np.array(mus),np.array(Cs)
    try: K=pairwise_rules(mus,Cs)
    except np.linalg.LinAlgError: return float('nan')
    X=np.concatenate([sc.sample(m,n_ev//sc.M,rng) for m in range(sc.M)])
    Phi=tr(X); out=np.zeros((len(X),sc.M),int)
    for (m,n),(k,c,_) in K.items():
        win=(Phi-c)@k>0; out[:,n]+=win; out[:,m]+=~win
    return float((~(out==sc.M-1).any(1)).mean())

print(f'{"basis":>36}' + ''.join(f'{n[:22]:>24}' for n,_ in SCEN))
for name,f in BASES.items():
    print(f'{name:>36}' + ''.join(f'{gap(sc,f):>24.5f}' for _,sc in SCEN))
