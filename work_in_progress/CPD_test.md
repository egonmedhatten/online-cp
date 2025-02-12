```python
# Safety check
check = False
if check:
    for p in np.linspace(0, 1, endpoint=True, num=100):
        tau = rnd_gen.uniform(0, 1)
        q0, q1 = cpd.quantile(p)
        q = cpd.quantile(p, tau)
        assert(cpd(q0, 0) >= p)
        assert(cpd(q1, 1) >= p)
        assert(cpd(q, tau) >= p)

    yrange = np.linspace(cpd.C[1], cpd.C[-2], endpoint=True, num=100)
    for y in yrange:
        tau = rnd_gen.uniform(0, 1)
        F0, F1 = cpd(y)
        assert cpd.quantile(F0)[0] <= y
        assert cpd.quantile(F1)[1] <= y
        F = cpd(y, tau)
        assert cpd.quantile(F, tau) <= y
```