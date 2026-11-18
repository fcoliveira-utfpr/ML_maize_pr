"""Revision analyses (Agricultural Systems), part 1: covariate shift, forward pricing, soybean break-even.
Run with cwd = ML_maize_pr/notebooks."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.special import logsumexp
from scipy.stats import norm, chi2

RAIZ = Path.cwd().parent
DADOS, RES = RAIZ / "dados", RAIZ / "resultados"
OUT = RES / "revisao"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
ORDEM = [f"{d:02d}/{m:02d}" for m in (1, 2, 3) for d in (5, 15, 25)]
ANOS = range(2000, 2026); N = 200; FUN = 0.015; COB = 0.65; TAU = 0.33; RR = 2

melhor = pd.read_csv(RES / "melhor_data_semeadura_pr.csv")
G15 = set(melhor.loc[melhor.melhor_data_zarc == "15/03", "cod_ibge"])
print("municipalities preferring 15/03:", len(G15))

# ============================================================ 1. covariate shift
sim = pd.read_parquet(RES / "simulacao_mza_pr.parquet")
cnt = [c for c in sim.columns if c.startswith(("geada_", "calor_"))]
sim[cnt] = sim[cnt].fillna(0)
sim["EC"] = sim.PA / sim.PPf
sim["geada_reprod"] = sim.geada_floracao + sim.geada_enchimento
sim["frio_reprod"] = np.clip(2 - sim[["Tmin_floracao", "Tmin_enchimento"]].min(axis=1), 0, None)
DYN = ["EC", "ISNA_vegetativo", "ISNA_floracao", "ISNA_enchimento", "DEF_floracao", "frio_reprod", "geada_reprod", "calor_floracao"]
base = pd.read_parquet(RES / "base_treino_hibrido.parquet")
Xtr = base[DYN].to_numpy()
mu_tr = Xtr.mean(0); Sinv = np.linalg.pinv(np.cov(Xtr, rowvar=False))


def d2(X):
    Z = X - mu_tr
    return np.einsum("ij,jk,ik->i", Z, Sinv, Z)


d2_tr = d2(Xtr)
thr99 = np.quantile(d2_tr, 0.99)
pred = sim[sim.cod_ibge.isin(base.cod_ibge.unique())].dropna(subset=DYN).copy()
pred["d2"] = d2(pred[DYN].to_numpy())
pred["out99"] = pred.d2 > thr99
lo, hi = base[DYN].quantile(0.005), base[DYN].quantile(0.995)
pred["n_out_range"] = ((pred[DYN] < lo) | (pred[DYN] > hi)).sum(1)
pred["g"] = np.where(pred.cod_ibge.isin(G15), "15/03 group", "other")
cs = pred.groupby(["semeadura", "g"]).agg(D2_median=("d2", "median"), share_beyond_P99_train=("out99", "mean"),
                                          share_any_var_outside_P0_5_P99_5=("n_out_range", lambda s: (s > 0).mean())).unstack()
cs = cs.reindex(ORDEM)
print("\n== covariate shift (training P99 of D2 = %.1f) ==" % thr99)
print(cs.round(3).to_string())
# which variables leave the training range, by date
vo = pd.DataFrame({v: ((pred[v] < lo[v]) | (pred[v] > hi[v])).groupby(pred.semeadura).mean() for v in DYN}).reindex(ORDEM)
print(vo.round(3).to_string())
cs.to_csv(OUT / "covariate_shift.csv"); vo.to_csv(OUT / "covariate_shift_vars.csv")
# same check against the training rows of the same region (local support): 15/03 group vs its own training rows
loc_tr = base[base.cod_ibge.isin(G15)][DYN]
mu_l, Sl = loc_tr.mean().to_numpy(), np.linalg.pinv(np.cov(loc_tr.to_numpy(), rowvar=False))
Zl = loc_tr.to_numpy() - mu_l; thr_l = np.quantile(np.einsum("ij,jk,ik->i", Zl, Sl, Zl), 0.99)
pg = pred[pred.cod_ibge.isin(G15)]
Zp = pg[DYN].to_numpy() - mu_l
print("local support, 15/03 group: share beyond local P99 by date:",
      pd.Series(np.einsum("ij,jk,ik->i", Zp, Sl, Zp) > thr_l).groupby(pg.semeadura.to_numpy()).mean().reindex(ORDEM).round(3).to_dict())
# heat at flowering: range in training vs the dates
print("calor_floracao training P50/P99/max:", base.calor_floracao.quantile([.5, .99, 1]).round(2).tolist())
print(pred.groupby(["semeadura", "g"]).calor_floracao.mean().unstack().reindex(ORDEM).round(2).to_string())

# ============================================================ 2. margin chain (as notebook 09)
prod = pd.read_parquet(RES / "produtividade_prevista_pr.parquet"); prod = prod[~prod.sem_historico_ibge]
custos = pd.read_csv(RES / "custos_milho_safrinha_real.csv").set_index("ano")
precos = pd.read_csv(RES / "precos_milho_pr_mensal_real.csv").set_index(["ano", "mes"])["preco_real_kg"]
risco = pd.read_csv(RES / "risco_economico_municipio_data.csv")
zarc = risco[["cod_ibge", "semeadura", "apto", "risco_zarc"]]
df = prod[prod.ano.isin(ANOS)].merge(zarc, on=["cod_ibge", "semeadura"]).reset_index(drop=True)
venda = pd.to_datetime(df.data_maturacao) + pd.Timedelta(days=20)
df["p"] = precos.reindex(pd.MultiIndex.from_arrays([venda.dt.year, venda.dt.month])).to_numpy()
# forward contract closed in January of the harvest year (same contract month for every sowing date)
df["p_sow"] = precos.reindex(pd.MultiIndex.from_arrays([df.ano, np.ones(len(df), int)])).to_numpy()
# unbiased variant: rescaled so that its mean equals the mean of the harvest price (removes the level effect)
df["p_sow_unb"] = df.p_sow * df.p.mean() / df.p_sow.mean()
c = custos.reindex(df.ano)
df["cha"], df["ckg"] = c["custo_operacional_ha_fixo"].to_numpy(), c["custo_var_real_kg"].to_numpy()
det = pd.read_excel(DADOS / "custos.xlsx", sheet_name="so_a_base", header=None).iloc[5:, [1, 5, 11]]
det.columns = ["cultura", "item", "valor"]
m2 = det[det.cultura == 7].set_index("item")["valor"].apply(pd.to_numeric, errors="coerce").fillna(0)
VAR = ["Operação de máquinas e implementos", "Despesas de manutenção de benfeitorias", "Mão-de-obra temporária",
       "Sementes/Manivas", "Fertilizantes", "Agrotóxicos", "Despesas gerais", "Transporte externo",
       "Assistência técnica", "PROAGRO/SEGURO", "Juros"]
df["premio"] = m2["PROAGRO/SEGURO"] / m2[VAR].sum() * c["custo_var_real_ha"].to_numpy()
mu = np.log(df.rend_P50_kg_ha.to_numpy())
sig = (np.log(df.rend_P90_kg_ha) - np.log(df.rend_P10_kg_ha)).to_numpy() / (2 * norm.ppf(0.9))
key = [df.cod_ibge.to_numpy(), df.semeadura.to_numpy()]
yexp = df.groupby(["cod_ibge", "semeadura"]).rend_P50_kg_ha.transform("mean").to_numpy()
Z = rng.standard_normal((len(df), N))
Y = np.exp(mu[:, None] + np.sqrt(sig ** 2 + TAU ** 2)[:, None] * Z)
W = float(custos.loc[list(ANOS), "custo_total_real_ha"].mean()); ra = RR / W
apto = df.apto.to_numpy()[:, None]
pseg = pd.Series(df.p.to_numpy()).groupby(key).transform("mean").to_numpy()


def margin(rev):
    cost = df.cha.to_numpy()[:, None] + df.ckg.to_numpy()[:, None] * Y
    ind = np.maximum(COB * yexp[:, None] - Y, 0) * pseg[:, None]
    return rev - cost + np.where(apto, ind, df.premio.to_numpy()[:, None])


grp = df.groupby(["cod_ibge", "semeadura"]).indices


def summarise(M):
    rows = []
    for (cod, d), ix in grp.items():
        v = M[ix].ravel()
        rows.append({"cod_ibge": cod, "semeadura": d, "mean": v.mean(), "sd": v.std(),
                     "CE": -(logsumexp(-ra * v) - np.log(v.size)) / ra, "ploss": (v < 0).mean(),
                     "cvar10": np.sort(v)[: max(1, v.size // 10)].mean()})
    return pd.DataFrame(rows).merge(zarc, on=["cod_ibge", "semeadura"])


def best(s):
    ok = s[s.apto]
    return ok.loc[ok.groupby("cod_ibge").CE.idxmax()].set_index("cod_ibge")


p = df.p.to_numpy()[:, None]
# expected yield used for the forward quantity: municipal expected yield of that date (what a farmer would contract)
scen = {"spot (base)": margin(p * Y * (1 - FUN))}
for tag, col in [("Jan price", "p_sow"), ("unbiased", "p_sow_unb")]:
    pf = df[col].to_numpy()[:, None]
    for h in (0.3, 0.5, 1.0):
        q = h * yexp[:, None]
        scen[f"forward {tag} h={h:.0%}"] = margin((q * pf + (Y - q) * p) * (1 - FUN))
    scen[f"price fixed on output, {tag}"] = margin(pf * Y * (1 - FUN))
# constant price (no price risk, no hedge): the mean harvest price of the municipality-date
scen["constant price"] = margin(pseg[:, None] * Y * (1 - FUN))
S = {k: summarise(M) for k, M in scen.items()}
b0 = best(S["spot (base)"])
rows = []
for k, s in S.items():
    b = best(s)
    rows.append({"scenario": k, "mean_margin": s["mean"].mean(), "sd_margin": s.sd.mean(), "CE": s.CE.mean(),
                 "ploss": s.ploss.mean(), "cvar10": s.cvar10.mean(),
                 "same_best_date_%": 100 * (b.semeadura == b0.semeadura.reindex(b.index)).mean(),
                 "best_05/01": (b.semeadura == "05/01").sum(), "best_15/03": (b.semeadura == "15/03").sum()})
fw = pd.DataFrame(rows).set_index("scenario")
print("\n== forward pricing scenarios ==\n", fw.round(3).to_string())
print("corr(state predicted yield, sowing-month price) =",
      round(df.groupby("ano")[["rend_P50_kg_ha", "p_sow"]].mean().corr().iloc[0, 1], 3),
      "| corr(yield, harvest price) =", round(df.groupby("ano")[["rend_P50_kg_ha", "p"]].mean().corr().iloc[0, 1], 3))
print("mean real price sowing month vs sale month (R$/60kg):", round(60 * df.p_sow.mean(), 1), round(60 * df.p.mean(), 1))
fw.to_csv(OUT / "forward_pricing.csv")
pd.concat({k: v.set_index(["cod_ibge", "semeadura"]) for k, v in S.items()}, names=["scenario"]).to_csv(OUT / "forward_pricing_mun_date.csv")

# ============================================================ 3. soybean break-even (system trade-off)
ipca = pd.read_csv(DADOS / "ipca_mensal.csv").set_index(["ano", "mes"])["indice"]
soy = pd.read_csv(DADOS / "ipea_precos_soja_mensal.csv")
soy = soy[soy.ano.between(1999, 2025)]
soy["real_kg"] = soy.valor * ipca.loc[(2025, 12)] / ipca.reindex(pd.MultiIndex.from_arrays([soy.ano, soy.mes])).to_numpy() / 60
p_soy = soy[soy.ano.between(2000, 2025) & soy.mes.isin([1, 2])].real_kg.mean()   # early soybean sold Jan–Feb
p_soy_all = soy[soy.ano.between(2000, 2025)].real_kg.mean()
soyy = pd.read_csv(DADOS / "ibge_pam_soja_pr.csv")
ys = soyy[soyy.ano.between(2016, 2025)].groupby("cod_ibge").rendimento_kg_ha.mean()
print(f"\nreal soybean price Jan–Feb: R$ {60 * p_soy:.1f}/60 kg (all months {60 * p_soy_all:.1f}); state soybean yield 2016–25 {ys.mean():.0f}")
s0 = S["spot (base)"]
ce = s0.pivot(index="cod_ibge", columns="semeadura", values="CE")
ap = s0.pivot(index="cod_ibge", columns="semeadura", values="apto")
early = b0[b0.semeadura == "05/01"].index
rows = []
for d in ["15/01", "25/01", "05/02", "15/02", "25/02"]:
    ok = early[ap.loc[early, d].astype(bool)]
    dce = (ce.loc[ok, "05/01"] - ce.loc[ok, d])
    # best attainable date on or after d (a farmer who cannot sow before d chooses the best of the remaining dates)
    later = [x for x in ORDEM if ORDEM.index(x) >= ORDEM.index(d)]
    alt = ce.loc[early, later].where(ap.loc[early, later].astype(bool)).max(1)
    dce_alt = (ce.loc[early, "05/01"] - alt).dropna()
    kg = dce_alt / p_soy
    rows.append({"latest_feasible_maize_sowing": d, "n_mun": len(dce_alt), "dCE_mean_R$ha": dce_alt.mean(),
                 "dCE_P10": dce_alt.quantile(.1), "dCE_P90": dce_alt.quantile(.9),
                 "soy_kg_ha_mean": kg.mean(), "soy_kg_P10": kg.quantile(.1), "soy_kg_P90": kg.quantile(.9),
                 "soy_pct_of_yield_mean": 100 * (kg / ys.reindex(kg.index)).mean(),
                 "dCE_same_date_only_mean": dce.mean()})
be = pd.DataFrame(rows).set_index("latest_feasible_maize_sowing")
print("\n== tolerable soybean loss (CE gain of 5 Jan maize vs best feasible later date) ==\n", be.round(1).to_string())
be.to_csv(OUT / "soja_break_even.csv")
pd.Series({"p_soy_janfev_R$kg": p_soy, "p_soy_all_R$kg": p_soy_all, "soy_yield_mean": ys.mean()}).to_csv(OUT / "soja_referencias.csv")
