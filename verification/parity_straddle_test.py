"""Tests whether basis parity effects manifest when signal levels straddle zero (both sides of zero)."""
import sys; sys.path.insert(0,'verification')
import numpy as np
from p2_intransitivity import Scenario
from basis_parity_sweep import evaluate
sgn=lambda p:(lambda x: np.sign(x)*np.abs(x)**p)
mk=lambda fs:(lambda x: np.stack([f(x) for f in fs],1))
B={'mixed {x,x²,x³}':mk([lambda x:x,lambda x:x**2,lambda x:x**3]),
   'patp1 {x,sgn·x²,x³}':mk([lambda x:x,sgn(2),lambda x:x**3]),
   'odd {x,x³,x⁵}':mk([lambda x:x,lambda x:x**3,lambda x:x**5])}

for label, straddle in (('Signal levels non-negative only (>= 0)', False),
                        ('Signal levels straddle zero (both sides of 0)', True)):
    print(f'\n=== {label} ===')
    print(f'{"basis":>22}{"med. gap":>14}{"max":>10}{">1%":>8}')
    for name,f in B.items():
        rng=np.random.default_rng(31); gaps=[]
        for _ in range(45):
            M=int(rng.integers(3,6))
            lv=np.sort(rng.uniform(-1.5,1.5,M)) if straddle else np.sort(rng.uniform(0,2.5,M))
            if not straddle: lv-=lv[0]
            sc=Scenario(lv, rng.uniform(0.0,1.9,M))
            try: g,_,_,_=evaluate(sc,f,n_mom=100_000,n_ev=120_000)
            except np.linalg.LinAlgError: continue
            gaps.append(g)
        g=np.array(gaps)
        print(f'{name:>22}{np.median(g):>14.5f}{g.max():>10.5f}{(g>.01).sum():>5}/{len(g)}')
