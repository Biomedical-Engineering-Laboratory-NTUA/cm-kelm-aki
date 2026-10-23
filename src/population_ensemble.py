"""
Population ensembles from a SINGLE evolutionary run.
====================================================

Seed ensembles showed that the diversity created by the evolutionary search,
not landmark averaging, improves discrimination.

An evolutionary run already HOLDS a diverse set of good solutions: its
final population. This module turns that population into an ensemble at
the cost of k final fits (~1 s each at m=1000), with no extra search.

Strategies
----------
topk:K      the K fittest distinct individuals
islands     the champion of each topographic island (Cultural Algorithm cells)
diverse:K   quality-diversity selection: start from the best; repeatedly add
            the individual that maximises its minimum feature-mask Hamming
            distance to those already chosen, among individuals whose
            fitness is within `tol` of the best (default 0.01)

Duplicates (identical mask and gamma/lambda to 3 decimals in log space) are
removed first, because DE populations often contain near-copies.
"""

import numpy as np

from nystrom_kelm import NystromKELM


def _decoded(model):
    out = []
    for g, f in zip(model.final_population_, model.final_fitnesses_):
        gamma, lam, mask = model.encoding.decode(g)
        out.append(dict(gamma=gamma, lam=lam, mask=mask, fitness=float(f)))
    return out


def _dedupe(members):
    seen, uniq = set(), []
    for mbr in sorted(members, key=lambda d: -d["fitness"]):
        key = (tuple(mbr["mask"].astype(int)), round(np.log(mbr["gamma"]), 3), round(np.log(mbr["lam"]), 3))
        if key not in seen:
            seen.add(key)
            uniq.append(mbr)
    return uniq


def select_members(model, strategy):
    members = _dedupe(_decoded(model))
    name, _, arg = strategy.partition(":")
    if name == "topk":
        return members[: int(arg or 5)]
    if name == "islands":
        if model.cell_assignment_ is None:
            return members[:1]
        pop_dec = _decoded(model)
        chosen = []
        for c in np.unique(model.cell_assignment_):
            idx = np.flatnonzero(model.cell_assignment_ == c)
            chosen.append(max((pop_dec[i] for i in idx), key=lambda d: d["fitness"]))
        return _dedupe(chosen)
    if name == "diverse":
        k, tol = int(arg or 5), 0.01
        best = members[0]["fitness"]
        pool = [mbr for mbr in members if mbr["fitness"] >= best - tol]
        chosen = [pool.pop(0)]
        while pool and len(chosen) < k:
            dist = [min(np.sum(c["mask"] != p["mask"]) for c in chosen) for p in pool]
            chosen.append(pool.pop(int(np.argmax(dist))))
        return chosen
    raise ValueError(f"unknown strategy '{strategy}'")


def population_ensemble_predict(model, Xtr, ytr, Xte, strategy):
    """Fits each selected member on the full training set (same settings as
    the single best model) and returns (mean probability on Xte, members)."""
    members = select_members(model, strategy)
    ps = []
    for mbr in members:
        mask = mbr["mask"]
        kelm = NystromKELM(gamma=mbr["gamma"], lam=mbr["lam"], n_landmarks=model.n_landmarks,
                           random_state=0, rank=model.rank, eig_rtol=model.eig_rtol)
        ps.append(kelm.fit(Xtr * mask, ytr).predict_proba(Xte * mask))
    info = [dict(gamma=float(m_["gamma"]), lam=float(m_["lam"]), n_features=int(m_["mask"].sum()),
                 features=np.flatnonzero(m_["mask"]).tolist(), fitness=m_["fitness"]) for m_ in members]
    return np.mean(ps, axis=0), info
