"""Revision analyses, part 3: training the hybrid with the observed sowing calendar (PSR).
Variants of the features that represent the municipal yield of a season:
  V0 equal mean of 4 window dates (current); V1 mean of the 9 dates weighted by the observed calendar;
  V2 one date drawn per municipality-year from the calendar (3 draws, models averaged) — keeps the
     single-date variance in training; V3 same as V2 with the 4 window dates equally likely.
Evaluated by (a) leave-one-year-out accuracy and (b) whether the per-date response reproduces the
within-season sowing-date gradient of indemnities in the PSR (independent of the training target).
Run with cwd = ML_maize_pr/notebooks; argv[1] = psr_pr.parquet."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import statsmodels.formula.api as smf

RAIZ = Path.cwd().parent
DADOS, RES = RAIZ / "dados", RAIZ / "resultados"
OUT = RES / "revisao"; OUT.mkdir(exist_ok=True)
ORDEM = [f"{d:02d}/{m:02d}" for m in (1, 2, 3) for d in (5, 15, 25)]
DOY = np.array([5, 15, 25, 36, 46, 56, 64, 74, 84])
JANELA = ["15/01", "05/02", "25/02", "15/03"]
SEED = 42

# ------------------------------------------------------------------ features per date (as notebook 03)
sim = pd.read_parquet(RES / "simulacao_mza_pr.parquet")
fase_cols = [c for c in sim.columns if any(c.endswith("_" + f) for f in ["estabelecimento", "vegetativo", "floracao", "enchimento", "maturacao"])]
cnt = [c for c in fase_cols if c.startswith(("geada_", "calor_"))]
sim[cnt] = sim[cnt].fillna(0); sim[fase_cols] = sim[fase_cols].fillna(sim[fase_cols].median())
sim["EC"] = sim.PA / sim.PPf
sim["geada_reprod"] = sim.geada_floracao + sim.geada_enchimento
sim["frio_reprod"] = np.clip(2 - sim[["Tmin_floracao", "Tmin_enchimento"]].min(axis=1), 0, None)
VAR_MZA = {"EC": 1, "ISNA_vegetativo": 1, "ISNA_floracao": 1, "ISNA_enchimento": 1, "DEF_floracao": -1,
           "frio_reprod": -1, "geada_reprod": -1, "calor_floracao": -1}
STAT = ["altitude_m", "dta_mm_m", "latitude", "longitude"]
DYN = list(VAR_MZA); VARS = DYN + STAT; MONO = list(VAR_MZA.values()) + [0] * 4
VARS_SVR = [v for v in VARS if v not in ("EC", "DEF_floracao")]
mun = pd.read_csv(DADOS / "municipios_pr.csv")
base0 = pd.read_parquet(RES / "base_treino_hibrido.parquet")[["cod_ibge", "ano", "rendimento_kg_ha", "log_y"] + STAT]
F = sim.set_index(["cod_ibge", "ano", "semeadura"])[DYN]  # per-date features

# ------------------------------------------------------------------ observed sowing calendar from the PSR
raw = pd.read_parquet(sys.argv[1]); raw = raw[raw.NM_CULTURA_GLOBAL.str.startswith("Milho 2")]
ind_col = [c for c in raw.columns if c.startswith("VALOR_INDENIZ")][0]
pol = pd.DataFrame({"cod_ibge": pd.to_numeric(raw.CD_GEOCMU, errors="coerce"),
                    "inicio": pd.to_datetime(raw.DT_INICIO_VIGENCIA, format="%d/%m/%Y", errors="coerce"),
                    "ind": pd.to_numeric(raw[ind_col].str.replace(".", "", regex=False).str.replace(",", ".", regex=False).replace("-", "0"), errors="coerce").fillna(0) > 0})
pol = pol[pol.inicio.dt.month.isin([12, 1, 2, 3, 4])]
pol["ano"] = pol.inicio.dt.year + (pol.inicio.dt.month == 12)
pol["doy"] = (pol.inicio - pd.to_datetime(pol.ano.astype(str) + "-01-01")).dt.days + 1
pol["data"] = np.array(ORDEM)[np.abs(pol.doy.to_numpy()[:, None] - DOY[None, :]).argmin(1)]
pol = pol[pol.ano.between(2016, 2025)].merge(mun[["cod_ibge", "mesorregiao"]], on="cod_ibge")
cnt_m = pol.groupby(["cod_ibge", "ano", "data"]).size().unstack(fill_value=0).reindex(columns=ORDEM, fill_value=0)
cnt_r = pol.groupby(["mesorregiao", "ano", "data"]).size().unstack(fill_value=0).reindex(columns=ORDEM, fill_value=0)
sh_r = cnt_r.div(cnt_r.sum(1), axis=0)
sh_r_early = sh_r[sh_r.index.get_level_values("ano") <= 2019].groupby(level=0).mean()  # older practice, pre-2016 seasons
sh_state_early = sh_r_early.mean()
K0 = 10  # shrinkage of the municipal calendar towards the mesoregion of the same season


def calendar(cod, ano, meso):
    if ano >= 2016 and (meso, ano) in sh_r.index:
        prior = sh_r.loc[(meso, ano)].to_numpy()
        if (cod, ano) in cnt_m.index:
            c = cnt_m.loc[(cod, ano)].to_numpy(); n = c.sum()
            return (c + K0 * prior) / (n + K0)
        return prior
    return (sh_r_early.loc[meso] if meso in sh_r_early.index else sh_state_early).to_numpy()


base0 = base0.merge(mun[["cod_ibge", "mesorregiao"]], on="cod_ibge")
W = np.vstack([calendar(c, a, m) for c, a, m in zip(base0.cod_ibge, base0.ano, base0.mesorregiao)])
print("mean observed calendar (training rows):", dict(zip(ORDEM, W.mean(0).round(3))))
pd.DataFrame(W, columns=ORDEM).assign(cod_ibge=base0.cod_ibge, ano=base0.ano).to_csv(OUT / "calendario_semeadura_psr.csv", index=False)

# per-date feature tensor for the training rows: (rows, 9, n_dyn)
idx = pd.MultiIndex.from_arrays([np.repeat(base0.cod_ibge.to_numpy(), 9), np.repeat(base0.ano.to_numpy(), 9), np.tile(ORDEM, len(base0))])
T = F.reindex(idx).to_numpy().reshape(len(base0), 9, len(DYN))
rng = np.random.default_rng(SEED)


def feats(variant, draw=0):
    X = base0[STAT].copy()
    if variant == "V0":
        w = np.zeros(9); w[[ORDEM.index(d) for d in JANELA]] = 0.25
        dyn = np.einsum("rdk,d->rk", T, w)
    elif variant == "V1":
        dyn = np.einsum("rdk,rd->rk", T, W)
    else:
        r = np.random.default_rng(SEED + draw)
        P = W if variant == "V2" else np.tile(np.isin(ORDEM, JANELA) / 4, (len(base0), 1))
        k = (P.cumsum(1) > r.random((len(base0), 1))).argmax(1)
        dyn = T[np.arange(len(base0)), k]
    X[DYN] = dyn
    return X[VARS]


def gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=100,
                                         l2_regularization=1.0, monotonic_cst=MONO, random_state=SEED)


def svr():
    return make_pipeline(StandardScaler(), SVR(C=0.3, gamma=0.05, epsilon=0.05))


def level(tr):
    reg = LinearRegression().fit(tr[["ano"]], tr.log_y)
    eff = (tr.log_y - reg.predict(tr[["ano"]])).groupby(tr.cod_ibge).mean()
    return reg, eff


NDRAW = {"V0": 1, "V1": 1, "V2": 3, "V3": 3}
results, fullmodels = {}, {}
for v in ["V0", "V1", "V2", "V3"]:
    Xs = [feats(v, d) for d in range(NDRAW[v])]
    pred = np.zeros(len(base0))
    for ano in sorted(base0.ano.unique()):
        tr, te = (base0.ano != ano).to_numpy(), (base0.ano == ano).to_numpy()
        reg, eff = level(base0[tr])
        b = lambda m: reg.predict(base0.loc[m, ["ano"]]) + base0.loc[m, "cod_ibge"].map(eff).fillna(0).to_numpy()
        an = base0.log_y[tr] - b(tr)
        acc = 0
        for X in Xs:
            e = gbm().fit(X[tr], an); s = svr().fit(X[tr][VARS_SVR], an)
            if v in ("V0", "V1"):
                acc += (e.predict(X[te]) + s.predict(X[te][VARS_SVR])) / 2
            else:  # mixture over sowing dates: calendar-weighted mean of the per-date predictions (log)
                Pw = W[te] if v == "V2" else np.tile(np.isin(ORDEM, JANELA) / 4, (te.sum(), 1))
                Xd = base0.loc[te, STAT].copy()
                for j in range(9):
                    if Pw[:, j].max() == 0: continue
                    Xd[DYN] = T[te, j]
                    acc += Pw[:, j] * (e.predict(Xd[VARS]) + s.predict(Xd[VARS][VARS_SVR])) / 2
        pred[te] = b(te) + acc / len(Xs)
    obs = base0.rendimento_kg_ha.to_numpy(); ph = np.exp(pred)
    st = pd.DataFrame({"o": obs, "p": ph, "ano": base0.ano}).groupby("ano").mean()
    results[v] = {"R2_mun": r2_score(obs, ph), "R2_state_annual": r2_score(st.o, st.p),
                  "RMSE": np.sqrt(np.mean((obs - ph) ** 2)), "pred": pred}
    print(v, {k: round(x, 3) for k, x in results[v].items() if k != "pred"}, flush=True)
    # full-data model for the per-date response
    reg, eff = level(base0)
    an = base0.log_y - reg.predict(base0[["ano"]]) - base0.cod_ibge.map(eff).to_numpy()
    fullmodels[v] = ([(gbm().fit(X, an), svr().fit(X[VARS_SVR], an)) for X in Xs], reg, eff)

# ------------------------------------------------------------------ PSR within-season date gradient
P = sim.merge(mun[["cod_ibge"] + STAT + ["mesorregiao"]], on="cod_ibge")
P = P[P.cod_ibge.isin(base0.cod_ibge.unique())]
melhor = pd.read_csv(RES / "melhor_data_semeadura_pr.csv")
G15 = set(melhor.loc[melhor.melhor_data_zarc == "15/03", "cod_ibge"])
cell = pol[pol.ano <= 2024].groupby(["cod_ibge", "ano", "data"]).agg(n=("ind", "size"), ind=("ind", "mean")).reset_index()
rows = []
for v, (mods, reg, eff) in fullmodels.items():
    an = np.mean([(e.predict(P[VARS]) + s.predict(P[VARS_SVR])) / 2 for e, s in mods], axis=0)
    res_sd = np.std(base0.log_y - results[v]["pred"])
    Pv = P[["cod_ibge", "ano", "semeadura"]].assign(mu=reg.predict(P[["ano"]]) + P.cod_ibge.map(eff).to_numpy() + an)
    # insured (expected) yield does not depend on the sowing date: municipal mean over window dates and years
    ye = Pv[Pv.semeadura.isin(JANELA) & Pv.ano.between(2000, 2025)].groupby("cod_ibge").mu.apply(lambda s: np.log(np.exp(s).mean()))
    Pv["p65"] = norm.cdf((np.log(0.65) + Pv.cod_ibge.map(ye) - Pv.mu) / np.sqrt(res_sd ** 2 + 0.33 ** 2))
    c = cell.merge(Pv.rename(columns={"semeadura": "data"}), on=["cod_ibge", "ano", "data"])
    c["late"] = (c.data.isin(["05/03", "15/03", "25/03"])).astype(int)
    c["g15"] = c.cod_ibge.isin(G15).astype(int)
    for g, d in [("all", c), ("15/03 group", c[c.g15 == 1])]:
        mo = smf.wls("ind ~ late + C(ano) + C(cod_ibge)", data=d, weights=d.n).fit()
        mm = smf.wls("p65 ~ late + C(ano) + C(cod_ibge)", data=d, weights=d.n).fit()
        rows.append({"variant": v, "group": g, "PSR_late_effect": mo.params.late, "PSR_se": mo.bse.late,
                     "model_late_effect": mm.params.late,
                     "brier_cells": np.average((d.ind - d.p65) ** 2, weights=d.n)})
    # recommended-date proxy: mean predicted log yield by date in the 15/03 group vs others
    Pv["g"] = np.where(Pv.cod_ibge.isin(G15), "15/03 group", "other")
    print(v, "mean predicted yield by date (kg/ha):")
    print(np.exp(Pv.groupby(["semeadura", "g"]).mu.mean()).unstack().reindex(ORDEM).round(0).T.to_string())
grad = pd.DataFrame(rows)
print("\n== within-season late-sowing effect on indemnity probability (municipality and year fixed effects) ==")
print(grad.round(3).to_string())
grad.to_csv(OUT / "variantes_gradiente_psr.csv", index=False)
pd.DataFrame({v: {k: x for k, x in r.items() if k != "pred"} for v, r in results.items()}).T.to_csv(OUT / "variantes_validacao.csv")
