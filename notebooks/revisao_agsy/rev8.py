"""Revision analyses, part 8: insurance-calibrated sowing-date response.
The IBGE yield has no sowing date, so the within-season date response is not identified by the yield data; the PSR
policies are. A date shift delta_d (log yield, delta_05/01 = 0) is added to the physics-guided prediction (PPf offset,
beta = 1) and calibrated so that the model's within-season indemnity profile (municipality and season fixed effects)
reproduces the PSR profile. Calibrated on seasons 2016–2020, tested on 2021–2024 (and vice versa); IBGE slope check.
Run with cwd = notebooks; argv[1] = psr_pr.parquet."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import statsmodels.formula.api as smf

RAIZ = Path.cwd().parent; DADOS, RES = RAIZ / "dados", RAIZ / "resultados"; OUT = RES / "revisao"
ORDEM = [f"{d:02d}/{m:02d}" for m in (1, 2, 3) for d in (5, 15, 25)]
DOY = np.array([5, 15, 25, 36, 46, 56, 64, 74, 84]); JANELA = ["15/01", "05/02", "25/02", "15/03"]
FASES = ["estabelecimento", "vegetativo", "floracao", "enchimento", "maturacao"]
STAT = ["altitude_m", "dta_mm_m", "latitude", "longitude"]
V = {"EC": 1, "ISNA_vegetativo": 1, "ISNA_floracao": 1, "ISNA_enchimento": 1, "DEF_floracao": -1,
     "frio_reprod": -1, "geada_reprod": -1, "calor_floracao": -1}
DYN = list(V); VARS = DYN + STAT; MONO = list(V.values()) + [0] * 4
VSVR = [v for v in VARS if v not in ("EC", "DEF_floracao")]
SEED = 42
mun = pd.read_csv(DADOS / "municipios_pr.csv")
pam = pd.read_csv(DADOS / "ibge_pam_milho_pr.csv")
pam = pam[(pam.safra == 2) & (pam.rendimento_kg_ha > 0) & (pam.area_colhida_ha >= 100)].sort_values(["cod_ibge", "ano"])
pam = pam[~(pam.groupby("cod_ibge").rendimento_kg_ha.diff() == 0)]
sim = pd.read_parquet(RES / "simulacao_mza_pr.parquet")
fc = [c for c in sim.columns if any(c.endswith("_" + f) for f in FASES)]
cnt = [c for c in fc if c.startswith(("geada_", "calor_", "seca_seq_", "frio_acum_", "calor_acum_"))]
sim[cnt] = sim[cnt].fillna(0); sim[fc] = sim[fc].fillna(sim[fc].median())
sim["EC"] = sim.PA / sim.PPf; sim["geada_reprod"] = sim.geada_floracao + sim.geada_enchimento
sim["frio_reprod"] = np.clip(2 - sim[["Tmin_floracao", "Tmin_enchimento"]].min(axis=1), 0, None)
sim["lppf"] = np.log(sim.PPf)

# final model on all seasons (as notebook 03) and residual spread from the leave-one-year-out run
wm = sim[sim.semeadura.isin(JANELA)].groupby(["cod_ibge", "ano"])[DYN].mean().reset_index()
b = pam[["cod_ibge", "ano", "rendimento_kg_ha"]].merge(wm, on=["cod_ibge", "ano"]).merge(mun[["cod_ibge"] + STAT], on="cod_ibge")
b["log_y"] = np.log(b.rendimento_kg_ha)
reg = LinearRegression().fit(b[["ano"]], b.log_y); eff = (b.log_y - reg.predict(b[["ano"]])).groupby(b.cod_ibge).mean()
an = b.log_y - reg.predict(b[["ano"]]) - b.cod_ibge.map(eff).to_numpy()
e = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=100, l2_regularization=1.0,
                                  monotonic_cst=MONO, random_state=SEED).fit(b[VARS], an)
s = make_pipeline(StandardScaler(), SVR(C=0.3, gamma=0.05, epsilon=0.05)).fit(b[VSVR], an)
oof = pd.read_parquet(RES / "previsoes_fora_amostra.parquet")
res_sd = float(np.std(oof.log_y - oof.L_ensemble_E_SVR))
P = sim[sim.cod_ibge.isin(b.cod_ibge.unique())].merge(mun[["cod_ibge"] + STAT], on="cod_ibge")
P = P.join(P[P.semeadura.isin(JANELA)].groupby(["cod_ibge", "ano"]).lppf.mean().rename("lw"), on=["cod_ibge", "ano"])
P["mu"] = reg.predict(P[["ano"]]) + P.cod_ibge.map(eff).to_numpy() + (e.predict(P[VARS]) + s.predict(P[VSVR])) / 2 + (P.lppf - P.lw)
ye = P[P.semeadura.isin(JANELA) & P.ano.between(2000, 2025)].groupby("cod_ibge").mu.apply(lambda x: np.log(np.exp(x).mean()))
P["zl"] = np.log(0.65) + P.cod_ibge.map(ye) - P.mu
tau = pd.read_csv(OUT / "tau_por_grupo.csv", index_col=0)["tau"]
tmap = {k.replace("meso: ", ""): v for k, v in tau.items() if k.startswith("meso: ")}
P = P.merge(mun[["cod_ibge", "mesorregiao"]], on="cod_ibge")
P["sd"] = np.sqrt(res_sd ** 2 + P.mesorregiao.map(tmap).fillna(tau["all (unweighted, as main)"]) ** 2)

# PSR cells
raw = pd.read_parquet(sys.argv[1]); raw = raw[raw.NM_CULTURA_GLOBAL.str.startswith("Milho 2")]
ic = [c for c in raw.columns if c.startswith("VALOR_INDENIZ")][0]
pol = pd.DataFrame({"cod_ibge": pd.to_numeric(raw.CD_GEOCMU, errors="coerce"),
                    "inicio": pd.to_datetime(raw.DT_INICIO_VIGENCIA, format="%d/%m/%Y", errors="coerce"),
                    "ind": pd.to_numeric(raw[ic].str.replace(".", "", regex=False).str.replace(",", ".", regex=False).replace("-", "0"), errors="coerce").fillna(0) > 0})
pol = pol[pol.inicio.dt.month.isin([12, 1, 2, 3, 4])]
pol["ano"] = pol.inicio.dt.year + (pol.inicio.dt.month == 12)
pol["doy"] = (pol.inicio - pd.to_datetime(pol.ano.astype(str) + "-01-01")).dt.days + 1
pol["data"] = np.array(ORDEM)[np.abs(pol.doy.to_numpy()[:, None] - DOY[None, :]).argmin(1)]
cell = pol[pol.ano.between(2016, 2024)].groupby(["cod_ibge", "ano", "data"]).agg(n=("ind", "size"), ind=("ind", "mean")).reset_index()
CEL = cell.merge(P[["cod_ibge", "ano", "semeadura", "zl", "sd"]].rename(columns={"semeadura": "data"}), on=["cod_ibge", "ano", "data"])


def profile(df, col):
    f = smf.wls(f"{col} ~ C(data, Treatment('05/01')) + C(ano) + C(cod_ibge)", data=df, weights=df.n).fit()
    return pd.Series({k.split("T.")[1][:5]: v for k, v in f.params.items() if k.startswith("C(data")}).reindex(ORDEM[1:])


def p65(df, delta):
    return norm.cdf((df.zl - df.data.map(delta).fillna(0)) / df.sd)


def calibrate(df):
    """delta_d such that the FE-adjusted model profile matches the PSR profile, date by date (fixed point)."""
    target = profile(df, "ind")
    delta = {d: 0.0 for d in ORDEM}
    for it in range(6):
        cur = profile(df.assign(p=p65(df, delta)), "p")
        for d in ORDEM[1:]:
            m = df.data == d
            # local slope of the cell-weighted mean probability w.r.t. delta_d
            base_p = np.average(p65(df[m], delta), weights=df.n[m])
            def gap(x):
                return np.average(p65(df[m], {**delta, d: x}), weights=df.n[m]) - base_p - (target[d] - cur[d])
            try:
                delta[d] = brentq(gap, -1.5, 1.5)
            except ValueError:
                pass
    return delta, target


rows = []
for name, cal, test in [("fit 2016–20, test 2021–24", CEL.ano <= 2020, CEL.ano >= 2021),
                        ("fit 2021–24, test 2016–20", CEL.ano >= 2021, CEL.ano <= 2020),
                        ("fit all", CEL.ano > 0, CEL.ano > 0)]:
    delta, _ = calibrate(CEL[cal])
    obs = profile(CEL[test], "ind")
    m0 = profile(CEL[test].assign(p=p65(CEL[test], {})), "p")
    m1 = profile(CEL[test].assign(p=p65(CEL[test], delta)), "p")
    rows.append({"split": name, **{f"delta_{d}": delta[d] for d in ORDEM[1:]},
                 "rmse_profile_before": np.sqrt(((obs - m0) ** 2).mean()), "rmse_profile_after": np.sqrt(((obs - m1) ** 2).mean()),
                 "r_before": obs.corr(m0), "r_after": obs.corr(m1),
                 "brier_before": np.average((CEL[test].ind - p65(CEL[test], {})) ** 2, weights=CEL[test].n),
                 "brier_after": np.average((CEL[test].ind - p65(CEL[test], delta)) ** 2, weights=CEL[test].n)})
    print(name, "| test profile PSR / model before / after:")
    print(pd.DataFrame({"PSR": obs, "before": m0, "after": m1}).round(3).T.to_string(), flush=True)
DOYMAP = dict(zip(ORDEM, DOY - 5))
for name, cal, test in [("LINEAR fit 2016–20, test 2021–24", CEL.ano <= 2020, CEL.ano >= 2021),
                        ("LINEAR fit 2021–24, test 2016–20", CEL.ano >= 2021, CEL.ano <= 2020),
                        ("LINEAR fit all", CEL.ano > 0, CEL.ano > 0)]:
    tgt = profile(CEL[cal], "ind")
    best = None
    for g in np.arange(-0.004, 0.0041, 0.0005):
        dl = {d: g * DOYMAP[d] for d in ORDEM}
        sse = ((profile(CEL[cal].assign(p=p65(CEL[cal], dl)), "p") - tgt) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, g)
    g = best[1]; dl = {d: g * DOYMAP[d] for d in ORDEM}
    obs = profile(CEL[test], "ind"); m0 = profile(CEL[test].assign(p=p65(CEL[test], {})), "p"); m1 = profile(CEL[test].assign(p=p65(CEL[test], dl)), "p")
    rows.append({"split": name, "gamma_per_day": g, "rmse_profile_before": np.sqrt(((obs - m0) ** 2).mean()),
                 "rmse_profile_after": np.sqrt(((obs - m1) ** 2).mean()), "r_before": obs.corr(m0), "r_after": obs.corr(m1),
                 "brier_before": np.average((CEL[test].ind - p65(CEL[test], {})) ** 2, weights=CEL[test].n),
                 "brier_after": np.average((CEL[test].ind - p65(CEL[test], dl)) ** 2, weights=CEL[test].n)})
    print(name, "gamma/day", g, flush=True)
r = pd.DataFrame(rows).set_index("split")
print(r.round(3).T.to_string())
r.to_csv(OUT / "calibracao_data_psr.csv")
# implied yield change by date (state mean) with the full-sample delta
full = {d: r.loc["fit all", f"delta_{d}"] for d in ORDEM[1:]}
P["mu_cal"] = P.mu + P.semeadura.map(full).fillna(0)  # mu_eff = mu + delta (see p65)
print("mean predicted yield by date before/after (kg/ha):")
print(pd.DataFrame({"before": np.exp(P.groupby("semeadura").mu.mean()), "after": np.exp(P.groupby("semeadura").mu_cal.mean())}).reindex(ORDEM).round(0).T.to_string())
# IBGE check: model-implied response of the season yield to the observed calendar (PSR median start), per 10 days
cal = pd.read_csv(OUT / "calendario_semeadura_psr.csv").set_index(["cod_ibge", "ano"])[ORDEM]
agg = pd.read_csv(DADOS / "psr_milho2safra_pr_municipio_ano.csv")
for col in ["mu", "mu_cal"]:
    pw = P.pivot_table(index=["cod_ibge", "ano"], columns="semeadura", values=col).reindex(columns=ORDEM)
    j = pw.index.intersection(cal.index)
    d = pd.DataFrame({"cod_ibge": j.get_level_values(0), "ano": j.get_level_values(1),
                      "shift": (pw.loc[j].to_numpy() * cal.loc[j].to_numpy()).sum(1) - pw.loc[j, JANELA].mean(1).to_numpy()})
    d = d.merge(agg[["cod_ibge", "ano", "inicio_doy_p50", "apolices"]], on=["cod_ibge", "ano"]).query("apolices >= 5 and ano <= 2024")
    d["doy_dev"] = d.inicio_doy_p50 - d.groupby("cod_ibge").inicio_doy_p50.transform("mean")
    print(col, "IBGE-check slope per 10 d:", round(10 * smf.ols("shift ~ doy_dev + C(ano)", data=d).fit().params.doy_dev, 4), "(observed -0.014)")
