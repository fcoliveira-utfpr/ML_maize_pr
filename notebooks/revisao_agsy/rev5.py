"""Physics offset for the sowing-date response: mu_d = mu_d(ML) + beta * [log PPf_d - mean_window log PPf].
beta calibrated to the within-season late-sowing indemnity gradient of the PSR; checked against the IBGE
yield slope with respect to the observed (PSR) median sowing date. argv[1] = psr_pr.parquet."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
import statsmodels.formula.api as smf

RAIZ = Path.cwd().parent; DADOS, RES = RAIZ / "dados", RAIZ / "resultados"; OUT = RES / "revisao"
ORDEM = [f"{d:02d}/{m:02d}" for m in (1, 2, 3) for d in (5, 15, 25)]
DOY = np.array([5, 15, 25, 36, 46, 56, 64, 74, 84]); JANELA = ["15/01", "05/02", "25/02", "15/03"]
sim = pd.read_parquet(RES / "simulacao_mza_pr.parquet", columns=["cod_ibge", "ano", "semeadura", "PPf", "PA"])
sim["lp"] = np.log(sim.PPf)
wm = sim[sim.semeadura.isin(JANELA)].groupby(["cod_ibge", "ano"]).lp.mean().rename("lp_w")
sim = sim.join(wm, on=["cod_ibge", "ano"]); sim["off"] = sim.lp - sim.lp_w
print("PPf and offset by date (state mean):")
print(sim.groupby("semeadura")[["PPf", "PA", "off"]].mean().reindex(ORDEM).round(3).T.to_string())

prod = pd.read_parquet(RES / "produtividade_prevista_pr.parquet")
prod = prod[~prod.sem_historico_ibge].merge(sim[["cod_ibge", "ano", "semeadura", "off"]], on=["cod_ibge", "ano", "semeadura"])
sig = ((np.log(prod.rend_P90_kg_ha) - np.log(prod.rend_P10_kg_ha)) / (2 * norm.ppf(.9))).to_numpy()
mu0 = np.log(prod.rend_P50_kg_ha).to_numpy()

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
cell["late"] = cell.data.isin(["05/03", "15/03", "25/03"]).astype(int)
cell["doy"] = cell.data.map(dict(zip(ORDEM, DOY)))
m_obs = smf.wls("ind ~ late + C(ano) + C(cod_ibge)", data=cell, weights=cell.n).fit()
m_obs_d = smf.wls("ind ~ doy + C(ano) + C(cod_ibge)", data=cell, weights=cell.n).fit()
print(f"\nPSR: late effect {m_obs.params.late:.3f} ({m_obs.bse.late:.3f}); per 10 days {10 * m_obs_d.params.doy:.3f} ({10 * m_obs_d.bse.doy:.3f})")
cal = pd.read_csv(OUT / "calendario_semeadura_psr.csv")
oof = pd.read_parquet(RES / "previsoes_fora_amostra.parquet")
agg = pd.read_csv(DADOS / "psr_milho2safra_pr_municipio_ano.csv")
rows = []
for beta in [0, 0.25, 0.5, 0.75, 1.0, 1.25]:
    mu = mu0 + beta * prod.off.to_numpy()
    p = prod[["cod_ibge", "ano", "semeadura"]].assign(mu=mu)
    ye = p[p.semeadura.isin(JANELA) & p.ano.between(2000, 2025)].groupby("cod_ibge").mu.apply(lambda s: np.log(np.exp(s).mean()))
    p["p65"] = norm.cdf((np.log(0.65) + p.cod_ibge.map(ye) - p.mu) / np.sqrt(sig ** 2 + 0.33 ** 2))
    c = cell.merge(p.rename(columns={"semeadura": "data"}), on=["cod_ibge", "ano", "data"])
    mm = smf.wls("p65 ~ late + C(ano) + C(cod_ibge)", data=c, weights=c.n).fit()
    mmd = smf.wls("p65 ~ doy + C(ano) + C(cod_ibge)", data=c, weights=c.n).fit()
    # IBGE check: model-implied municipal yield of the season under the observed calendar vs the window mean
    pw = p.pivot_table(index=["cod_ibge", "ano"], columns="semeadura", values="mu").reindex(columns=ORDEM)
    cc = cal.set_index(["cod_ibge", "ano"])[ORDEM]
    j = pw.index.intersection(cc.index)
    mix = (pw.loc[j].to_numpy() * cc.loc[j].to_numpy()).sum(1)
    win = pw.loc[j, JANELA].mean(1).to_numpy()
    d = pd.DataFrame({"cod_ibge": j.get_level_values(0), "ano": j.get_level_values(1), "shift": mix - win}).merge(
        agg[["cod_ibge", "ano", "inicio_doy_p50", "apolices"]], on=["cod_ibge", "ano"]).query("apolices >= 5 and ano <= 2024")
    d["doy_dev"] = d.inicio_doy_p50 - d.groupby("cod_ibge").inicio_doy_p50.transform("mean")
    sl = smf.ols("shift ~ doy_dev + C(ano)", data=d).fit().params.doy_dev
    rows.append({"beta": beta, "model_late": mm.params.late, "model_per10d": 10 * mmd.params.doy,
                 "brier": np.average((c.ind - c.p65) ** 2, weights=c.n), "ibge_slope_model_per10d": 10 * sl,
                 "yield_15mar_vs_05jan_%": 100 * (np.exp(p[p.semeadura == "15/03"].mu.mean() - p[p.semeadura == "05/01"].mu.mean()) - 1)})
r = pd.DataFrame(rows)
print(r.round(4).to_string())
# IBGE observed slope (from rev2): obs anomaly per 10 d later sowing = -0.014 (all municipalities)
r.to_csv(OUT / "calibracao_beta_ppf.csv", index=False)
