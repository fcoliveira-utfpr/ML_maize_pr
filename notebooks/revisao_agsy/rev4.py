"""Revision analyses, part 4 (ideas from recent Agricultural Systems papers):
 1 efficient frontier mean margin x CVaR10 by municipality; 2 decision gap vs current sowing calendar (PSR);
 3 stress-stratified validation + stress-conditional conformal; 5 lower tail and stability of yield by date/region.
Run with cwd = ML_maize_pr/notebooks. Optional env TAU_REGIONAL=1 uses tau by mesoregion (resultados/revisao/tau_por_grupo.csv)."""
import os
import numpy as np, pandas as pd
from pathlib import Path
from scipy.special import logsumexp
from scipy.stats import norm

RAIZ = Path.cwd().parent
DADOS, RES = RAIZ / "dados", RAIZ / "resultados"
OUT = RES / "revisao"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
ORDEM = [f"{d:02d}/{m:02d}" for m in (1, 2, 3) for d in (5, 15, 25)]
ANOS = range(2000, 2026); N = 200; FUN = 0.015; COB = 0.65; RR = 2
mun = pd.read_csv(DADOS / "municipios_pr.csv")

# ------------------------------------------------------------------ margin chain (as notebook 09), tau per municipality
prod = pd.read_parquet(os.environ.get("PROD_FILE", RES / "produtividade_prevista_pr.parquet")); prod = prod[~prod.sem_historico_ibge]
custos = pd.read_csv(RES / "custos_milho_safrinha_real.csv").set_index("ano")
precos = pd.read_csv(RES / "precos_milho_pr_mensal_real.csv").set_index(["ano", "mes"])["preco_real_kg"]
risco = pd.read_csv(RES / "risco_economico_municipio_data.csv")
zarc = risco[["cod_ibge", "semeadura", "apto", "risco_zarc"]]
df = prod[prod.ano.isin(ANOS)].merge(zarc, on=["cod_ibge", "semeadura"]).merge(mun[["cod_ibge", "mesorregiao"]], on="cod_ibge").reset_index(drop=True)
venda = pd.to_datetime(df.data_maturacao) + pd.Timedelta(days=20)
df["p"] = precos.reindex(pd.MultiIndex.from_arrays([venda.dt.year, venda.dt.month])).to_numpy()
c = custos.reindex(df.ano)
df["cha"], df["ckg"] = c["custo_operacional_ha_fixo"].to_numpy(), c["custo_var_real_kg"].to_numpy()
det = pd.read_excel(DADOS / "custos.xlsx", sheet_name="so_a_base", header=None).iloc[5:, [1, 5, 11]]
det.columns = ["cultura", "item", "valor"]
m2 = det[det.cultura == 7].set_index("item")["valor"].apply(pd.to_numeric, errors="coerce").fillna(0)
VAR = ["Operação de máquinas e implementos", "Despesas de manutenção de benfeitorias", "Mão-de-obra temporária",
       "Sementes/Manivas", "Fertilizantes", "Agrotóxicos", "Despesas gerais", "Transporte externo",
       "Assistência técnica", "PROAGRO/SEGURO", "Juros"]
df["premio"] = m2["PROAGRO/SEGURO"] / m2[VAR].sum() * c["custo_var_real_ha"].to_numpy()
if os.environ.get("TAU_REGIONAL"):
    t = pd.read_csv(OUT / "tau_por_grupo.csv", index_col=0)["tau"]
    tmap = {k.replace("meso: ", ""): v for k, v in t.items() if k.startswith("meso: ")}
    tau = df.mesorregiao.map(tmap).fillna(t["all (unweighted, as main)"]).to_numpy()
else:
    tau = np.full(len(df), 0.33)
mu = np.log(df.rend_P50_kg_ha.to_numpy())
sig = (np.log(df.rend_P90_kg_ha) - np.log(df.rend_P10_kg_ha)).to_numpy() / (2 * norm.ppf(0.9))
key = [df.cod_ibge.to_numpy(), df.semeadura.to_numpy()]
yexp = df.groupby(["cod_ibge", "semeadura"]).rend_P50_kg_ha.transform("mean").to_numpy()
if os.environ.get("YEXP_MUN"):  # PSR rule: insured yield = municipal expectation, independent of the sowing date
    _w = df[df.semeadura.isin(["15/01", "05/02", "25/02", "15/03"])].groupby("cod_ibge").rend_P50_kg_ha.mean()
    yexp = df.cod_ibge.map(_w).to_numpy()
zk = df.groupby(["cod_ibge", "ano"]).ngroup().to_numpy()
Z = rng.standard_normal((zk.max() + 1, N))[zk]  # common random numbers: same field draws for every date of a municipality-season
Y = np.exp(mu[:, None] + np.sqrt(sig ** 2 + tau ** 2)[:, None] * Z)
W = float(custos.loc[list(ANOS), "custo_total_real_ha"].mean()); ra = RR / W
apto = df.apto.to_numpy()[:, None]
pseg = pd.Series(df.p.to_numpy()).groupby(key).transform("mean").to_numpy()
cost = df.cha.to_numpy()[:, None] + df.ckg.to_numpy()[:, None] * Y
ind = np.maximum(COB * yexp[:, None] - Y, 0) * pseg[:, None]
M = df.p.to_numpy()[:, None] * Y * (1 - FUN) - cost + np.where(apto, ind, df.premio.to_numpy()[:, None])
grp = df.groupby(["cod_ibge", "semeadura"]).indices
ce = lambda v: -(logsumexp(-ra * v) - np.log(v.size)) / ra
rows = []
for (cod, d), ix in grp.items():
    v = M[ix].ravel(); s = np.sort(v)
    rows.append({"cod_ibge": cod, "semeadura": d, "mean": v.mean(), "CE": ce(v), "cvar10": s[: s.size // 10].mean(),
                 "ploss": (v < 0).mean()})
S = pd.DataFrame(rows).merge(zarc, on=["cod_ibge", "semeadura"])
ok = S[S.apto].copy()
best = ok.loc[ok.groupby("cod_ibge").CE.idxmax()].set_index("cod_ibge")
print("best date counts:", best.semeadura.value_counts().reindex(ORDEM).dropna().astype(int).to_dict())

# ------------------------------------------------------------------ 1. efficient frontier (mean margin vs CVaR10)
def efficient(g):
    m, c = g["mean"].to_numpy(), g.cvar10.to_numpy()
    dom = ((m[None, :] >= m[:, None]) & (c[None, :] >= c[:, None]) & ((m[None, :] > m[:, None]) | (c[None, :] > c[:, None]))).any(1)
    return pd.Series(~dom, index=g.index)


ok["efficient"] = ok.groupby("cod_ibge", group_keys=False).apply(efficient)
fr = pd.DataFrame({
    "n_efficient_dates_mean": ok.groupby("cod_ibge").efficient.sum().mean(),
    "share_mun_single_efficient": (ok.groupby("cod_ibge").efficient.sum() == 1).mean(),
    "CE_best_is_efficient": ok.set_index(["cod_ibge", "semeadura"]).efficient.reindex(pd.MultiIndex.from_arrays([best.index, best.semeadura])).mean(),
}, index=[0]).T
print("\n== efficient frontier ==\n", fr.round(3).to_string())
print("share of municipalities where each date is efficient:\n", ok.groupby("semeadura").efficient.mean().reindex(ORDEM).round(3).to_dict())
ok.to_csv(OUT / "fronteira_eficiente.csv", index=False)

# ------------------------------------------------------------------ 2. decision gap vs current sowing calendar (PSR)
cal = pd.read_csv(OUT / "calendario_semeadura_psr.csv")
cur = cal[cal.ano >= 2022].groupby("cod_ibge")[ORDEM].mean()  # current practice: seasons 2022–2025
cur = cur.div(cur.sum(1), axis=0)
area = pd.read_parquet(RES / "base_treino_hibrido.parquet").query("ano >= 2021").groupby("cod_ibge").area_colhida_ha.mean()
ap = S.pivot(index="cod_ibge", columns="semeadura", values="apto").reindex(columns=ORDEM).fillna(False).astype(bool)
gap = []
for cod in best.index.intersection(cur.index):
    w = cur.loc[cod].to_numpy().copy()
    a = ap.loc[cod].to_numpy()
    if not a.any(): continue
    # shares on dates outside the zoning move to the nearest suitable date (insured farmers sow inside ZARC)
    for j in np.where(~a & (w > 0))[0]:
        k = np.where(a)[0][np.abs(np.where(a)[0] - j).argmin()]; w[k] += w[j]; w[j] = 0
    # farm-level margin of the current calendar: same season, area-weighted mix of the dates
    blocks = {d: M[grp[(cod, d)]] for d in ORDEM if (cod, d) in grp}
    mix = sum(w[ORDEM.index(d)] * blocks[d] for d in blocks if w[ORDEM.index(d)] > 0)
    v = mix.ravel(); s = np.sort(v)
    gap.append({"cod_ibge": cod, "CE_current": ce(v), "CE_best": best.loc[cod, "CE"], "mean_current": v.mean(),
                "mean_best": best.loc[cod, "mean"], "cvar_current": s[: s.size // 10].mean(), "cvar_best": best.loc[cod, "cvar10"],
                "median_doy_current": np.interp(0.5, np.cumsum(w), [5, 15, 25, 36, 46, 56, 64, 74, 84]),
                "best": best.loc[cod, "semeadura"], "area_ha": area.get(cod, 0)})
gap = pd.DataFrame(gap)
gap["gap_CE"] = gap.CE_best - gap.CE_current
print("\n== decision gap: recommended date vs current (2022–2025 PSR) calendar ==")
print(gap[["gap_CE", "mean_best", "mean_current", "cvar_best", "cvar_current"]].describe().round(0).to_string())
tot = (gap.gap_CE * gap.area_ha).sum() / 1e6
print(f"area-weighted state gain: R$ {tot:.0f} million/yr over {gap.area_ha.sum() / 1e6:.2f} Mha; "
      f"mean gap {np.average(gap.gap_CE, weights=gap.area_ha + 1e-9):.0f} R$/ha")
gap.to_csv(OUT / "gap_decisao.csv", index=False)

# ------------------------------------------------------------------ 3. stress-stratified validation
base = pd.read_parquet(RES / "base_treino_hibrido.parquet")
oof = pd.read_parquet(RES / "previsoes_fora_amostra.parquet").merge(
    base[["cod_ibge", "ano", "ISNA_floracao", "ISNA_enchimento", "geada_reprod", "frio_reprod", "calor_floracao"]], on=["cod_ibge", "ano"])
isna = oof[["ISNA_floracao", "ISNA_enchimento"]].min(1)
cls = np.select([oof.geada_reprod >= 0.5, isna <= isna.quantile(0.2), oof.calor_floracao >= 1.0],
                ["frost/cold", "drought", "heat"], "no major stress")
oof["stress"] = cls
FIN = "L_ensemble_E_SVR"
res = oof.log_y - oof[FIN]
# Mondrian conformal by stress class (class known before harvest: it comes from the crop model, not from yield)
cov_m = np.zeros(len(oof), bool); cov_g = np.zeros(len(oof), bool)
for a in oof.ano.unique():
    te, tr = (oof.ano == a).to_numpy(), (oof.ano != a).to_numpy()
    lo, hi = res[tr].quantile([.1, .9]); cov_g[te] = res[te].between(lo, hi).to_numpy()
    for k in np.unique(cls[te]):
        r = res[tr & (cls == k)]
        if len(r) < 100: r = res[tr]
        lo, hi = r.quantile([.1, .9]); m = te & (cls == k)
        cov_m[m] = res[m].between(lo, hi).to_numpy()
oof["cov_global"], oof["cov_stress"] = cov_g, cov_m
for col in ["A_media_historica", "B_MZA_calibrado", "E_hibrido_monotonico", FIN]:
    oof["se_" + col] = (oof.rendimento_kg_ha - np.exp(oof[col])) ** 2
tab = oof.groupby("stress").agg(n=("ano", "size"), obs_mean=("rendimento_kg_ha", "mean"),
                                bias_L=("rendimento_kg_ha", lambda s: (np.exp(oof.loc[s.index, FIN]) - s).mean()),
                                cov_global=("cov_global", "mean"), cov_stress=("cov_stress", "mean"),
                                **{f"RMSE_{c[:1]}": ("se_" + c, lambda s: np.sqrt(s.mean())) for c in ["A_media_historica", "B_MZA_calibrado", "E_hibrido_monotonico", FIN]})
print("\n== stress-stratified validation (out of sample) ==\n", tab.round(3).to_string())
print("coverage 2020–2022: global", round(oof[oof.ano.between(2020, 2022)].cov_global.mean(), 3),
      "stress-conditional", round(oof[oof.ano.between(2020, 2022)].cov_stress.mean(), 3))
tab.to_csv(OUT / "validacao_por_estresse.csv")
oof[["cod_ibge", "ano", "stress", "cov_global", "cov_stress"]].to_csv(OUT / "classes_estresse.csv", index=False)

# ------------------------------------------------------------------ 5. lower tail and stability by date and region
p = prod[prod.ano.between(2000, 2025)].merge(mun[["cod_ibge", "mesorregiao"]], on="cod_ibge")
st = p.groupby(["cod_ibge", "semeadura"]).rend_P50_kg_ha.agg(mean="mean", sd="std", p10=lambda s: s.quantile(.1)).reset_index()
st["cv"] = st.sd / st["mean"]; st["lower_limit_rel"] = (st["mean"] - st.sd) / st["mean"]
st = st.merge(mun[["cod_ibge", "mesorregiao"]], on="cod_ibge")
by_date = st.groupby("semeadura")[["mean", "p10", "cv"]].mean().reindex(ORDEM)
by_reg = st.pivot_table(index="mesorregiao", columns="semeadura", values="cv").reindex(columns=ORDEM)
print("\n== yield lower tail and stability (2000–2025 weather, 2025 technology) ==\n", by_date.round(3).to_string())
print(by_reg.round(3).to_string())
st.to_csv(OUT / "cauda_estabilidade.csv", index=False)
