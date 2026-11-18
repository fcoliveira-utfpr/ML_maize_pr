"""Revision analyses, part 2: PSR microdata (policy level, aggregated without personal data).
Run with cwd = ML_maize_pr/notebooks; argv[1] = path of psr_pr.parquet (scratchpad, not in the repo)."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
import statsmodels.formula.api as smf

RAIZ = Path.cwd().parent
DADOS, RES = RAIZ / "dados", RAIZ / "resultados"
OUT = RES / "revisao"; OUT.mkdir(exist_ok=True)
ORDEM = [f"{d:02d}/{m:02d}" for m in (1, 2, 3) for d in (5, 15, 25)]

raw = pd.read_parquet(sys.argv[1])
raw = raw[raw.NM_CULTURA_GLOBAL.str.startswith("Milho 2")].copy()
ind_col = [c for c in raw.columns if c.startswith("VALOR_INDENIZ")][0]
num = lambda s: pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
psr = pd.DataFrame({
    "cod_ibge": pd.to_numeric(raw.CD_GEOCMU, errors="coerce"),
    "inicio": pd.to_datetime(raw.DT_INICIO_VIGENCIA, format="%d/%m/%Y", errors="coerce"),
    "area": num(raw.NR_AREA_TOTAL), "prod_est": num(raw.NR_PRODUTIVIDADE_ESTIMADA),
    "cob": num(raw.NivelDeCobertura), "taxa": num(raw.PE_TAXA),
    "indeniz": num(raw[ind_col].replace("-", "0")).fillna(0),
    "evento": raw.EVENTO_PREPONDERANTE.replace("-", np.nan),
})
psr["ano"] = psr.inicio.dt.year  # second-season maize: coverage starts at sowing (Jan–Apr) = harvest year
psr = psr[psr.inicio.dt.month.between(1, 4) | (psr.inicio.dt.month == 12)]
psr.loc[psr.inicio.dt.month == 12, "ano"] += 1
psr["doy"] = (psr.inicio - pd.to_datetime(psr.ano.astype(str) + "-01-01")).dt.days + 1
psr["ind"] = psr.indeniz > 0
print(len(psr), "policies;", psr.ano.value_counts().sort_index().to_dict())
print("start-of-coverage date (proxy of sowing) quantiles by year (day of year):")
print(psr.groupby("ano").doy.quantile([.1, .5, .9]).unstack().round(0).to_string())

bins = [-40, 20, 41, 59, 200]; lab = ["<=20 Jan", "21 Jan-10 Feb", "11-28 Feb", ">=1 Mar"]
psr["epoca"] = pd.cut(psr.doy, bins, labels=lab)
melhor = pd.read_csv(RES / "melhor_data_semeadura_pr.csv")
G15 = set(melhor.loc[melhor.melhor_data_zarc == "15/03", "cod_ibge"])
psr["g"] = np.where(psr.cod_ibge.isin(G15), "15/03 group", "other")
psr = psr[psr.ano.between(2016, 2025)]

# ---------- aggregated table without personal data (goes to dados/)
agg = psr.groupby(["cod_ibge", "ano"]).agg(apolices=("ind", "size"), area_ha=("area", "sum"),
                                           frac_indenizadas=("ind", "mean"), indenizacao_rs=("indeniz", "sum"),
                                           cobertura_media=("cob", "mean"), taxa_premio_media=("taxa", "mean"),
                                           prod_esperada_media=("prod_est", "mean"),
                                           inicio_doy_p25=("doy", lambda s: s.quantile(.25)),
                                           inicio_doy_p50=("doy", "median"), inicio_doy_p75=("doy", lambda s: s.quantile(.75))).reset_index()
for l in lab:
    agg["frac_" + l.replace(" ", "_").replace("<=", "ate_").replace(">=", "desde_")] = \
        psr.groupby(["cod_ibge", "ano"]).epoca.apply(lambda s: (s == l).mean()).to_numpy()
agg.to_csv(DADOS / "psr_milho2safra_pr_municipio_ano.csv", index=False)
print("aggregated:", agg.shape)

# ---------- empirical check: indemnity frequency by sowing period, 15/03 group vs others
t = psr[psr.ano <= 2024]  # 2025 claims not yet processed
emp = t.groupby(["g", "epoca"], observed=True).agg(n=("ind", "size"), frac_ind=("ind", "mean")).unstack(0)
print("\n== indemnified share by sowing period (2016–2024) ==\n", emp.round(3).to_string())
# within-year contrast (year fixed effects): linear probability model
t = t.assign(indi=t.ind.astype(int), late=(t.doy >= 60).astype(int), mid=((t.doy > 41) & (t.doy < 60)).astype(int), g15=(t.g == "15/03 group").astype(int))
lpm = smf.ols("indi ~ C(ano) + mid*g15 + late*g15", data=t).fit(cov_type="cluster", cov_kwds={"groups": t.cod_ibge})
print(lpm.params.filter(regex="mid|late|g15").round(3).to_string())
print(lpm.bse.filter(regex="mid|late|g15").round(3).to_string())
by_year = t.groupby(["g", "ano", "epoca"], observed=True).ind.agg(["size", "mean"]).unstack("epoca")
by_year.to_csv(OUT / "psr_indenizacao_epoca_ano.csv")
emp.to_csv(OUT / "psr_indenizacao_epoca.csv")
ev = t[t.ind].groupby(["g", "epoca"], observed=True).evento.value_counts(normalize=True).unstack().fillna(0)
print("\nevents among indemnified policies:\n", ev.loc[:, ev.max() > 0.03].round(2).to_string())

# model counterpart: P(field yield < 65% of expected) by date, same groups and years
prod = pd.read_parquet(RES / "produtividade_prevista_pr.parquet")
prod = prod[prod.ano.between(2016, 2024) & ~prod.sem_historico_ibge]
mu = np.log(prod.rend_P50_kg_ha); sig = (np.log(prod.rend_P90_kg_ha) - np.log(prod.rend_P10_kg_ha)) / (2 * norm.ppf(.9))
ye = prod.groupby(["cod_ibge", "semeadura"]).rend_P50_kg_ha.transform("mean")
prod["p65"] = norm.cdf((np.log(0.65 * ye) - mu) / np.sqrt(sig ** 2 + 0.33 ** 2))
prod["g"] = np.where(prod.cod_ibge.isin(G15), "15/03 group", "other")
print("\nmodel P(field < 65%) 2016–2024 by date:\n", prod.groupby(["semeadura", "g"]).p65.mean().unstack().reindex(ORDEM).round(3).to_string())

# ---------- IBGE: yield anomaly in late-sowing vs early-sowing seasons (PSR calendar of the municipality)
base = pd.read_parquet(RES / "base_treino_hibrido.parquet")
oof = pd.read_parquet(RES / "previsoes_fora_amostra.parquet")
o = oof.merge(agg[["cod_ibge", "ano", "inicio_doy_p50", "apolices"]], on=["cod_ibge", "ano"])
o = o[o.apolices >= 5]
o["res_L"] = o.log_y - o.L_ensemble_E_SVR
o["anom_obs"] = o.log_y - o.A_media_historica
o["g"] = np.where(o.cod_ibge.isin(G15), "15/03 group", "other")
o["doy_dev"] = o.inicio_doy_p50 - o.groupby("cod_ibge").inicio_doy_p50.transform("mean")  # later than usual
for g, d in o.groupby("g"):
    m1 = smf.ols("anom_obs ~ doy_dev + C(ano)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.cod_ibge})
    m2 = smf.ols("res_L ~ doy_dev + C(ano)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.cod_ibge})
    print(f"{g}: n={len(d)}; obs anomaly per 10 d later sowing = {10 * m1.params.doy_dev:.3f} (se {10 * m1.bse.doy_dev:.3f}); "
          f"model residual per 10 d = {10 * m2.params.doy_dev:.3f} (se {10 * m2.bse.doy_dev:.3f})")

# ---------- tau by region, and adverse selection
mun = pd.read_csv(DADOS / "municipios_pr.csv")
prod = prod.merge(mun[["cod_ibge", "areia_pct", "mesorregiao"]], on="cod_ibge")
JANELA = ["15/01", "05/02", "25/02", "15/03"]
risco = pd.read_csv(RES / "risco_economico_municipio_data.csv")[["cod_ibge", "semeadura", "apto"]]
# expected yield as in notebook 05: mean of the 2000–2025 predictions of the municipality-date
pfull = pd.read_parquet(RES / "produtividade_prevista_pr.parquet")
yexp = pfull[pfull.ano.between(2000, 2025)].groupby(["cod_ibge", "semeadura"]).rend_P50_kg_ha.mean().rename("yexp")
pp = prod[prod.semeadura.isin(JANELA)].merge(risco, on=["cod_ibge", "semeadura"]).merge(yexp.reset_index(), on=["cod_ibge", "semeadura"])
pp = pp[pp.apto]
pp["sig"] = (np.log(pp.rend_P90_kg_ha) - np.log(pp.rend_P10_kg_ha)) / (2 * norm.ppf(.9))
pp["zl"] = np.log(0.65 * pp.yexp) - np.log(pp.rend_P50_kg_ha)
w_pol = agg.set_index(["cod_ibge", "ano"]).apolices
pp["w"] = w_pol.reindex(pd.MultiIndex.from_frame(pp[["cod_ibge", "ano"]])).fillna(0).to_numpy()
agg2 = agg.merge(mun[["cod_ibge", "areia_pct", "mesorregiao"]], on="cod_ibge")
grade = np.arange(0, 0.8, 0.01)


def calib(sel_pp, sel_agg, weighted):
    tgt = sel_agg[sel_agg.ano <= 2024].groupby("ano").apply(lambda d: np.average(d.frac_indenizadas, weights=d.apolices))
    d = sel_pp[sel_pp.ano <= 2024]
    best = None
    for tau in grade:
        p = pd.Series(norm.cdf(d.zl / np.sqrt(d.sig ** 2 + tau ** 2)), index=d.index)
        f = (p * d.w).groupby(d.ano).sum() / d.w.groupby(d.ano).sum() if weighted else p.groupby(d.ano).mean()
        e = ((f - tgt.reindex(f.index)) ** 2).sum()
        if best is None or e < best[0]:
            best = (e, tau, f.corr(tgt.reindex(f.index)))
    return best[1], best[2]


rows = []
groups = {"all (unweighted, as main)": (pp, agg2, False), "all (weighted by policies)": (pp, agg2, True),
          "sandy (sand > 50%)": (pp[pp.areia_pct > 50], agg2[agg2.areia_pct > 50], True),
          "clayey (sand <= 50%)": (pp[pp.areia_pct <= 50], agg2[agg2.areia_pct <= 50], True),
          "15/03 group": (pp[pp.cod_ibge.isin(G15)], agg2[agg2.cod_ibge.isin(G15)], True)}
for m in sorted(agg2.mesorregiao.dropna().unique()):
    groups[f"meso: {m}"] = (pp[pp.mesorregiao == m], agg2[agg2.mesorregiao == m], True)
for k, (a, b, wt) in groups.items():
    if b.apolices.sum() < 2000: continue
    tau, r = calib(a, b, wt)
    rows.append({"group": k, "policies": int(b.apolices.sum()), "tau": tau, "r_annual": r,
                 "frac_ind": np.average(b.frac_indenizadas, weights=b.apolices)})
taus = pd.DataFrame(rows).set_index("group")
print("\n== tau by group ==\n", taus.round(3).to_string())
taus.to_csv(OUT / "tau_por_grupo.csv")

# adverse selection: insured intensity vs municipal risk and insured expected yield vs IBGE
pam = pd.read_csv(DADOS / "ibge_pam_milho_pr.csv"); pam = pam[pam.safra == 2]
m = agg[agg.ano <= 2024].merge(pam[["cod_ibge", "ano", "area_colhida_ha", "rendimento_kg_ha"]], on=["cod_ibge", "ano"])
m["uptake"] = (m.area_ha / m.area_colhida_ha).clip(upper=1.5)
riskm = pd.read_csv(RES / "risco_economico_municipio_data.csv").query("semeadura in @JANELA").groupby("cod_ibge").p_prejuizo.mean()
cvm = pam[pam.ano.between(2003, 2025)].groupby("cod_ibge").rendimento_kg_ha.agg(lambda s: s.std() / s.mean())
mm = m.groupby("cod_ibge").agg(uptake=("uptake", "mean"), ratio_yield=("prod_esperada_media", "mean"), ibge=("rendimento_kg_ha", "mean"))
mm["ratio_yield"] = mm.ratio_yield / mm.ibge
mm["p_loss_model"] = riskm.reindex(mm.index); mm["cv_ibge"] = cvm.reindex(mm.index)
print("\n== adverse selection checks (municipalities) ==")
print("Spearman uptake vs model P(loss):", round(mm.uptake.corr(mm.p_loss_model, method="spearman"), 3),
      "| uptake vs CV of IBGE yield:", round(mm.uptake.corr(mm.cv_ibge, method="spearman"), 3))
print("insured expected yield / IBGE mean yield: median", round(mm.ratio_yield.median(), 3), "IQR",
      mm.ratio_yield.quantile([.25, .75]).round(3).tolist())
mm.to_csv(OUT / "selecao_adversa_municipio.csv")
