# %% [markdown]
# # 10 — Revision analyses (Agricultural Systems)
#
# Analyses added in the revision for *Agricultural Systems* (Nov 2026), on the final run of notebooks 03 → 05 → 09
# (physics offset of the potential yield by sowing date, τ by mesoregion, insured yield by the PSR rule, common random
# numbers). Inputs are only aggregated files (no personal data): `dados/psr_milho2safra_pr_celulas.csv` (PSR policies
# by municipality × season × sowing date, from the start-of-coverage date) and `dados/psr_milho2safra_pr_municipio_ano.csv`.
# Exploratory scripts that led to the modelling decisions are in `notebooks/revisao_agsy/` (README there).
#
# 1. Insurance evidence on the sowing-date response (PSR) and the physics offset
# 2. Covariate shift of the per-date predictions
# 3. Forward pricing (hedge ratios)
# 4. Soybean break-even: the system trade-off of early maize
# 5. Efficient frontier (mean margin × CVaR) and decision gap vs the current sowing calendar
# 6. Stress-stratified validation and stress-conditional conformal intervals
# 7. Lower tail and stability of yield by date and region
# 8. Drivers by mesoregion (SHAP)
# 9. Robustness tables from the exploratory runs
# 10. Figures

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import logsumexp
from scipy.stats import norm

import mapas

RAIZ = Path.cwd().parent
DADOS, RES, FIG = RAIZ / "dados", RAIZ / "resultados", RAIZ / "figuras"
OUT = RES / "revisao"; OUT.mkdir(exist_ok=True)
ART = RAIZ.parent / "artigo_elsevier" / "figuras"
rng = np.random.default_rng(42)
ORDEM = [f"{d:02d}/{m:02d}" for m in (1, 2, 3) for d in (5, 15, 25)]
DOY = dict(zip(ORDEM, [5, 15, 25, 36, 46, 56, 64, 74, 84]))
JANELA4 = ["15/01", "05/02", "25/02", "15/03"]
JAN_ZARC = ["15/01", "25/01", "05/02", "15/02", "25/02", "05/03", "15/03"]
ANOS = range(2000, 2026); N = 200; FUN = 0.015; COB = 0.65; RR = 2
COR = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "gray": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e"}
plt.rcParams.update({"font.size": 13, "axes.edgecolor": COR["ink2"], "axes.labelcolor": COR["ink"],
                     "xtick.color": COR["ink2"], "ytick.color": COR["ink2"], "axes.spines.top": False, "axes.spines.right": False})

mun = pd.read_csv(DADOS / "municipios_pr.csv")
melhor = pd.read_csv(RES / "melhor_data_semeadura_pr.csv")
risco = pd.read_csv(RES / "risco_economico_municipio_data.csv")
zarc = risco[["cod_ibge", "semeadura", "apto", "risco_zarc"]]
prod_all = pd.read_parquet(RES / "produtividade_prevista_pr.parquet")
prod = prod_all[~prod_all.sem_historico_ibge]
prod_beta0 = pd.read_parquet(RES / "_antes_revisao" / "produtividade_prevista_pr.parquet")  # before the physics offset
prod_beta0 = prod_beta0[~prod_beta0.sem_historico_ibge]
tau_meso = pd.read_csv(RES / "tau_mesorregiao.csv", index_col=0)["tau"]
cel = pd.read_csv(DADOS / "psr_milho2safra_pr_celulas.csv").astype({"cod_ibge": int})
psr_ma = pd.read_csv(DADOS / "psr_milho2safra_pr_municipio_ano.csv")
base = pd.read_parquet(RES / "base_treino_hibrido.parquet")
oof = pd.read_parquet(RES / "previsoes_fora_amostra.parquet")
meso = mun.set_index("cod_ibge")["mesorregiao"]
print("best date:", melhor.melhor_data_zarc.value_counts().to_dict())

# %% [markdown]
# ## Shared margin chain (identical to notebooks 05/09)

# %%
custos = pd.read_csv(RES / "custos_milho_safrinha_real.csv").set_index("ano")
precos = pd.read_csv(RES / "precos_milho_pr_mensal_real.csv").set_index(["ano", "mes"])["preco_real_kg"]
det = pd.read_excel(DADOS / "custos.xlsx", sheet_name="so_a_base", header=None).iloc[5:, [1, 5, 11]]
det.columns = ["cultura", "item", "valor"]
m2 = det[det.cultura == 7].set_index("item")["valor"].apply(pd.to_numeric, errors="coerce").fillna(0)
VAR = ["Operação de máquinas e implementos", "Despesas de manutenção de benfeitorias", "Mão-de-obra temporária",
       "Sementes/Manivas", "Fertilizantes", "Agrotóxicos", "Despesas gerais", "Transporte externo",
       "Assistência técnica", "PROAGRO/SEGURO", "Juros"]


def cadeia(prod):
    df = prod[prod.ano.isin(ANOS)].merge(zarc, on=["cod_ibge", "semeadura"]).reset_index(drop=True)
    venda = pd.to_datetime(df.data_maturacao) + pd.Timedelta(days=20)
    df["p"] = precos.reindex(pd.MultiIndex.from_arrays([venda.dt.year, venda.dt.month])).to_numpy()
    df["p_jan"] = precos.reindex(pd.MultiIndex.from_arrays([df.ano, np.ones(len(df), int)])).to_numpy()
    c = custos.reindex(df.ano)
    df["cha"], df["ckg"] = c["custo_operacional_ha_fixo"].to_numpy(), c["custo_var_real_kg"].to_numpy()
    df["premio"] = m2["PROAGRO/SEGURO"] / m2[VAR].sum() * c["custo_var_real_ha"].to_numpy()
    mu = np.log(df.rend_P50_kg_ha.to_numpy())
    sig = (np.log(df.rend_P90_kg_ha) - np.log(df.rend_P10_kg_ha)).to_numpy() / (2 * norm.ppf(0.9))
    tau = df.cod_ibge.map(meso).map(tau_meso).fillna(tau_meso["_state"]).to_numpy()
    yexp = df.cod_ibge.map(df[df.semeadura.isin(JAN_ZARC)].groupby("cod_ibge").rend_P50_kg_ha.mean()).to_numpy()
    zk = df.groupby(["cod_ibge", "ano"]).ngroup().to_numpy()
    Z = np.random.default_rng(42).standard_normal((zk.max() + 1, N))[zk]
    Y = np.exp(mu[:, None] + np.sqrt(sig ** 2 + tau ** 2)[:, None] * Z)
    key = [df.cod_ibge.to_numpy(), df.semeadura.to_numpy()]
    pseg = pd.Series(df.p.to_numpy()).groupby(key).transform("mean").to_numpy()
    return df, Y, yexp, pseg


df, Y, yexp, pseg = cadeia(prod)
W = float(custos.loc[list(ANOS), "custo_total_real_ha"].mean()); ra = RR / W
apto = df.apto.to_numpy()[:, None]
grp = df.groupby(["cod_ibge", "semeadura"]).indices


def margem(rev, Y=Y):
    cost = df.cha.to_numpy()[:, None] + df.ckg.to_numpy()[:, None] * Y
    ind = np.maximum(COB * yexp[:, None] - Y, 0) * pseg[:, None]
    return rev - cost + np.where(apto, ind, df.premio.to_numpy()[:, None])


def ce(v):
    return -(logsumexp(-ra * v) - np.log(v.size)) / ra


def resumo(M):
    rows = []
    for (cod, d), ix in grp.items():
        v = M[ix].ravel(); s = np.sort(v)
        rows.append({"cod_ibge": cod, "semeadura": d, "mean": v.mean(), "sd": v.std(), "CE": ce(v),
                     "ploss": (v < 0).mean(), "cvar10": s[: s.size // 10].mean()})
    return pd.DataFrame(rows).merge(zarc, on=["cod_ibge", "semeadura"])


def melhor_data(s):
    ok = s[s.apto]
    return ok.loc[ok.groupby("cod_ibge").CE.idxmax()].set_index("cod_ibge")


M0 = margem(df.p.to_numpy()[:, None] * Y * (1 - FUN))
S0 = resumo(M0)
B0 = melhor_data(S0)
chk = S0.merge(risco[["cod_ibge", "semeadura", "CE_rr2"]], on=["cod_ibge", "semeadura"])
print("CE check vs notebook 05 (corr, mean diff):", round(chk.CE.corr(chk.CE_rr2), 4), round((chk.CE - chk.CE_rr2).mean(), 1))

# %% [markdown]
# ## 1. Insurance evidence on the sowing-date response
#
# PSR policies (2016–2024; 2025 claims not yet processed) by sowing date (start of coverage, nearest of the nine
# simulated dates). Within-season profile = date coefficients of a weighted linear probability model with municipality
# and season fixed effects. The model counterpart is the probability that field yield falls below 65% of the insured
# (municipal) yield, with τ of the mesoregion — for the final model (physics offset, β = 1) and the model without it.

# %%
c = cel[cel.ano.between(2016, 2024)].rename(columns={"data_semeadura": "data"})
G15 = set(melhor.loc[melhor.melhor_data_zarc == "15/03", "cod_ibge"])
G15_old = set(pd.read_csv(RES / "_antes_revisao" / "melhor_data_semeadura_pr.csv").query("melhor_data_zarc == '15/03'").cod_ibge)


def p65_model(pr):
    p = pr[pr.ano.between(2016, 2024)].copy()
    mu = np.log(p.rend_P50_kg_ha); sig = (np.log(p.rend_P90_kg_ha) - np.log(p.rend_P10_kg_ha)) / (2 * norm.ppf(.9))
    ye = pr[pr.semeadura.isin(JAN_ZARC) & pr.ano.between(2000, 2025)].groupby("cod_ibge").rend_P50_kg_ha.mean()
    tau = p.cod_ibge.map(meso).map(tau_meso).fillna(tau_meso["_state"])
    p["p65"] = norm.cdf((np.log(COB * p.cod_ibge.map(ye)) - mu) / np.sqrt(sig ** 2 + tau ** 2))
    return p[["cod_ibge", "ano", "semeadura", "p65"]].rename(columns={"semeadura": "data"})


cc = c.merge(p65_model(prod), on=["cod_ibge", "ano", "data"]).merge(
    p65_model(prod_beta0).rename(columns={"p65": "p65_b0"}), on=["cod_ibge", "ano", "data"])


def perfil(d, col):
    f = smf.wls(f"{col} ~ C(data, Treatment('05/01')) + C(ano) + C(cod_ibge)", data=d, weights=d.apolices).fit()
    est = {k.split("T.")[1][:5]: (v, f.bse[k]) for k, v in f.params.items() if k.startswith("C(data")}
    return pd.DataFrame(est, index=["coef", "se"]).T.reindex(ORDEM[1:])


prof = pd.concat({"PSR": perfil(cc, "frac_indenizadas"), "model beta=1": perfil(cc, "p65"),
                  "model beta=0": perfil(cc, "p65_b0")}, axis=1)
print("== within-season indemnity profile relative to 05/01 ==\n", prof.xs("coef", axis=1, level=1).round(3).T.to_string())
prof.to_csv(OUT / "perfil_data_psr.csv")
# simple shares by sowing period (descriptive) for the municipalities that the earlier model sent to 15/03
c["per"] = pd.cut(c.data.map(DOY), [0, 20, 41, 59, 100], labels=["≤20 Jan", "21 Jan–10 Feb", "11–28 Feb", "≥1 Mar"])
c["grupo"] = np.where(c.cod_ibge.isin(G15_old), "27 mun. (earlier 15/03)", "rest of PR")
desc = c.groupby(["grupo", "per"], observed=True).apply(
    lambda d: pd.Series({"policies": d.apolices.sum(), "indemnified": np.average(d.frac_indenizadas, weights=d.apolices),
                         "drought": np.average(d.frac_ind_seca, weights=d.apolices), "frost": np.average(d.frac_ind_geada, weights=d.apolices)}),
    include_groups=False).unstack(0)
print(desc.round(3).to_string()); desc.to_csv(OUT / "psr_indenizacao_epoca.csv")
# within-season late-sowing effect (≥ 1 Mar) with season FE, clustered by municipality
cl = c.assign(late=(c.data.map(DOY) >= 60).astype(int), mid=c.data.map(DOY).between(42, 59).astype(int), g=c.cod_ibge.isin(G15_old).astype(int))
lpm = smf.wls("frac_indenizadas ~ C(ano) + mid*g + late*g", data=cl, weights=cl.apolices).fit(cov_type="cluster", cov_kwds={"groups": cl.cod_ibge})
print(pd.DataFrame({"coef": lpm.params, "se": lpm.bse}).filter(regex="mid|late|^g$", axis=0).round(3).to_string())

# IBGE check: season yield response to the observed median start of coverage, model vs observation (per 10 days later)
cal = (c.assign(w=c.apolices).pivot_table(index=["cod_ibge", "ano"], columns="data", values="w", aggfunc="sum", fill_value=0)
       .reindex(columns=ORDEM, fill_value=0))
calw = cal.div(cal.sum(1), axis=0)
ag = psr_ma[(psr_ma.apolices >= 5) & psr_ma.ano.between(2016, 2024)][["cod_ibge", "ano", "inicio_doy_p50"]]
o = oof.merge(ag, on=["cod_ibge", "ano"]); o["anom"] = o.log_y - o.A_media_historica
o["doy_dev"] = o.inicio_doy_p50 - o.groupby("cod_ibge").inicio_doy_p50.transform("mean")
f_obs = smf.ols("anom ~ doy_dev + C(ano)", data=o).fit(cov_type="cluster", cov_kwds={"groups": o.cod_ibge})
ibge = {"observed": (10 * f_obs.params.doy_dev, 10 * f_obs.bse.doy_dev)}
for nome, pr in [("model beta=1", prod), ("model beta=0", prod_beta0)]:
    pw = np.log(pr.pivot_table(index=["cod_ibge", "ano"], columns="semeadura", values="rend_P50_kg_ha").reindex(columns=ORDEM))
    j = pw.index.intersection(calw.index)
    d = pd.DataFrame({"cod_ibge": j.get_level_values(0), "ano": j.get_level_values(1),
                      "shift": (pw.loc[j].to_numpy() * calw.loc[j].to_numpy()).sum(1) - pw.loc[j, JANELA4].mean(1).to_numpy()}).merge(ag, on=["cod_ibge", "ano"])
    d["doy_dev"] = d.inicio_doy_p50 - d.groupby("cod_ibge").inicio_doy_p50.transform("mean")
    f = smf.ols("shift ~ doy_dev + C(ano)", data=d).fit()
    ibge[nome] = (10 * f.params.doy_dev, 10 * f.bse.doy_dev)
ibge = pd.DataFrame(ibge, index=["log yield per 10 d later", "se"]).T
print(ibge.round(4).to_string()); ibge.to_csv(OUT / "ibge_resposta_calendario.csv")

# %% [markdown]
# ## 2. Covariate shift of the per-date predictions
# Mahalanobis distance of the dynamic features of each date to the training distribution (window means).

# %%
sim = pd.read_parquet(RES / "simulacao_mza_pr.parquet")
sim[[x for x in sim.columns if x.startswith(("geada_", "calor_"))]] = sim[[x for x in sim.columns if x.startswith(("geada_", "calor_"))]].fillna(0)
sim["EC"] = sim.PA / sim.PPf; sim["geada_reprod"] = sim.geada_floracao + sim.geada_enchimento
sim["frio_reprod"] = np.clip(2 - sim[["Tmin_floracao", "Tmin_enchimento"]].min(axis=1), 0, None)
DYN = ["EC", "ISNA_vegetativo", "ISNA_floracao", "ISNA_enchimento", "DEF_floracao", "frio_reprod", "geada_reprod", "calor_floracao"]
Xtr = base[DYN].to_numpy(); mu_tr = Xtr.mean(0); Sinv = np.linalg.pinv(np.cov(Xtr, rowvar=False))
d2 = lambda X: np.einsum("ij,jk,ik->i", X - mu_tr, Sinv, X - mu_tr)
thr = np.quantile(d2(Xtr), 0.99)
pp = sim[sim.cod_ibge.isin(base.cod_ibge.unique())].dropna(subset=DYN)
pp = pp.assign(out=d2(pp[DYN].to_numpy()) > thr)
lo, hi = base[DYN].quantile(0.005), base[DYN].quantile(0.995)
pp["fora"] = ((pp[DYN] < lo) | (pp[DYN] > hi)).any(axis=1)
# stacked single-date training rows (4 window dates) as reference: how much of the shift is the averaging itself
ref = sim[sim.semeadura.isin(JANELA4)].merge(base[["cod_ibge", "ano"]], on=["cod_ibge", "ano"])
cs = pp.groupby("semeadura")[["out", "fora"]].mean().reindex(ORDEM)
cs.loc["4 window dates, single-date"] = [(d2(ref[DYN].to_numpy()) > thr).mean(), ((ref[DYN] < lo) | (ref[DYN] > hi)).any(axis=1).mean()]
cs.columns = ["share beyond training P99 (D²)", "share with a feature outside training P0.5–P99.5"]
print(cs.round(3).to_string()); cs.to_csv(OUT / "covariate_shift.csv")

# %% [markdown]
# ## 3. Forward pricing
# Part of the expected production (hedge ratio h) is sold forward in January at that month's price (unbiased version:
# rescaled to the mean harvest price, to remove the level effect); the rest is sold at harvest. A short position larger
# than the realized crop is bought back at the harvest price.

# %%
p = df.p.to_numpy()[:, None]
pf_unb = (df.p_jan * df.p.mean() / df.p_jan.mean()).to_numpy()[:, None]
cen = {"spot (base)": M0}
for h in (0.25, 0.5, 0.75, 1.0):
    q = h * yexp[:, None]
    cen[f"forward h={h:.2f}"] = margem((q * pf_unb + (Y - q) * p) * (1 - FUN))
cen["constant price (no price risk)"] = margem(pseg[:, None] * Y * (1 - FUN))
Sf = {k: resumo(M) for k, M in cen.items()}
fw = pd.DataFrame({k: {"mean margin": s["mean"].mean(), "SD": s.sd.mean(), "CE": s.CE.mean(), "P(loss)": s.ploss.mean(),
                       "CVaR10": s.cvar10.mean(),
                       "same best date (%)": 100 * (melhor_data(s).semeadura == B0.semeadura.reindex(melhor_data(s).index)).mean(),
                       "best = 05/01 (n)": (melhor_data(s).semeadura == "05/01").sum()} for k, s in Sf.items()}).T
print(fw.round(1).to_string()); fw.to_csv(OUT / "forward_pricing.csv")
st = df.groupby("ano")[["rend_P50_kg_ha", "p", "p_jan"]].mean()
print("corr(state predicted yield, harvest price) =", round(st.rend_P50_kg_ha.corr(st.p), 2),
      "| corr(yield, January price) =", round(st.rend_P50_kg_ha.corr(st.p_jan), 2))

# %% [markdown]
# ## 4. Soybean break-even
# CE gain of sowing maize on 5 January instead of the best ZARC-suitable date on or after a later date d, converted
# into soybean (real Jan–Feb price, R$ of Dec/2025): the soybean yield the farmer can give up (shorter cycle, earlier
# soybean sowing) before the double-crop system loses. Municipalities whose best date is 05/01.

# %%
ipca = pd.read_csv(DADOS / "ipca_mensal.csv").set_index(["ano", "mes"])["indice"]
soy = pd.read_csv(DADOS / "ipea_precos_soja_mensal.csv").query("2000 <= ano <= 2025")
soy["real_kg"] = soy.valor * ipca.loc[(2025, 12)] / ipca.reindex(pd.MultiIndex.from_arrays([soy.ano, soy.mes])).to_numpy() / 60
p_soy = soy[soy.mes.isin([1, 2])].real_kg.mean()
ys = pd.read_csv(DADOS / "ibge_pam_soja_pr.csv").query("2016 <= ano <= 2025").groupby("cod_ibge").rendimento_kg_ha.mean()
cev = S0.pivot(index="cod_ibge", columns="semeadura", values="CE"); apv = S0.pivot(index="cod_ibge", columns="semeadura", values="apto").astype(bool)
early = B0[B0.semeadura == "05/01"].index
rows = []
for d in ["15/01", "25/01", "05/02", "15/02", "25/02"]:
    later = ORDEM[ORDEM.index(d):]
    alt = cev.loc[early, later].where(apv.loc[early, later]).max(axis=1)
    g = (cev.loc[early, "05/01"] - alt).dropna(); kg = g / p_soy
    rows.append({"maize cannot be sown before": d, "n": len(g), "CE gain R$/ha (mean)": g.mean(), "P10": g.quantile(.1), "P90": g.quantile(.9),
                 "soybean kg/ha (mean)": kg.mean(), "soybean kg/ha P10": kg.quantile(.1), "soybean kg/ha P90": kg.quantile(.9),
                 "% of soybean yield": 100 * (kg / ys.reindex(kg.index)).mean()})
soja = pd.DataFrame(rows).set_index("maize cannot be sown before")
print(f"real soybean price Jan–Feb R$ {60 * p_soy:.1f}/60 kg; mean soybean yield {ys.mean():.0f} kg/ha")
print(soja.round(1).to_string()); soja.to_csv(OUT / "soja_break_even.csv")
be_mun = (cev.loc[early, "05/01"] - cev.loc[early, ORDEM[3:]].where(apv.loc[early, ORDEM[3:]]).max(axis=1)) / p_soy

# %% [markdown]
# ## 5. Efficient frontier and decision gap vs the current sowing calendar

# %%
ok = S0[S0.apto].copy()


def eficiente(g):
    m, v = g["mean"].to_numpy(), g.cvar10.to_numpy()
    dom = ((m[None] >= m[:, None]) & (v[None] >= v[:, None]) & ((m[None] > m[:, None]) | (v[None] > v[:, None]))).any(1)
    return pd.Series(~dom, index=g.index)


ok["efficient"] = ok.groupby("cod_ibge", group_keys=False).apply(eficiente)
fr = pd.Series({"efficient dates per municipality (mean)": ok.groupby("cod_ibge").efficient.sum().mean(),
                "municipalities with a single efficient date (%)": 100 * (ok.groupby("cod_ibge").efficient.sum() == 1).mean(),
                "CE-optimal date is efficient (%)": 100 * ok.set_index(["cod_ibge", "semeadura"]).efficient.reindex(
                    pd.MultiIndex.from_arrays([B0.index, B0.semeadura])).mean()})
print(fr.round(2).to_string()); print("share efficient by date:", ok.groupby("semeadura").efficient.mean().reindex(ORDEM).round(3).to_dict())
ok.to_csv(OUT / "fronteira_eficiente.csv", index=False)

# current practice: sowing calendar of the insured fields in seasons 2022–2025 (PSR)
cel_all = cel.rename(columns={"data_semeadura": "data"})
cal_all = cel_all.pivot_table(index=["cod_ibge", "ano"], columns="data", values="apolices", aggfunc="sum", fill_value=0).reindex(columns=ORDEM, fill_value=0)
cur = cal_all[cal_all.index.get_level_values("ano") >= 2022].groupby(level=0).sum()
cur = cur.div(cur.sum(1), axis=0)
area = base[base.ano >= 2021].groupby("cod_ibge").area_colhida_ha.mean()
gap = []
for cod in B0.index.intersection(cur.index):
    w = cur.loc[cod].to_numpy().copy(); a = apv.loc[cod].reindex(ORDEM).fillna(False).to_numpy()
    for j in np.where(~a & (w > 0))[0]:  # shares outside the zoning move to the nearest suitable date
        k = np.where(a)[0][np.abs(np.where(a)[0] - j).argmin()]; w[k] += w[j]; w[j] = 0
    mix = sum(w[i] * M0[grp[(cod, d)]] for i, d in enumerate(ORDEM) if w[i] > 0 and (cod, d) in grp)
    v = mix.ravel(); s = np.sort(v)
    gap.append({"cod_ibge": cod, "CE_current": ce(v), "CE_best": B0.loc[cod, "CE"], "mean_current": v.mean(), "mean_best": B0.loc[cod, "mean"],
                "cvar_current": s[: s.size // 10].mean(), "cvar_best": B0.loc[cod, "cvar10"], "ploss_current": (v < 0).mean(),
                "ploss_best": B0.loc[cod, "ploss"], "area_ha": area.get(cod, 0.0)})
gap = pd.DataFrame(gap); gap["gap_CE"] = gap.CE_best - gap.CE_current
wa = lambda x: np.average(x, weights=gap.area_ha + 1e-9)
print(f"decision gap (area-weighted): CE {wa(gap.gap_CE):.0f} R$/ha; mean margin {wa(gap.mean_best - gap.mean_current):.0f}; "
      f"CVaR10 {wa(gap.cvar_best - gap.cvar_current):.0f}; P(loss) {wa(gap.ploss_best - gap.ploss_current):.3f}; "
      f"state total R$ {(gap.gap_CE * gap.area_ha).sum() / 1e6:.0f} million/yr over {gap.area_ha.sum() / 1e6:.2f} Mha; "
      f"municipalities with gap > 0: {(gap.gap_CE > 0.5).mean():.1%}")
gap.to_csv(OUT / "gap_decisao.csv", index=False)

# %% [markdown]
# ## 6. Stress-stratified validation
# Classes from the crop model (known before harvest): frost/cold (reproductive frost days ≥ 0.5), drought (min ISNA of
# flowering/grain filling in the lowest quintile), heat (≥ 1 day ≥ 35 °C at flowering), no major stress. Conformal
# intervals conditional on the class (Mondrian) vs global.

# %%
ov = oof.merge(base[["cod_ibge", "ano", "ISNA_floracao", "ISNA_enchimento", "geada_reprod", "calor_floracao"]], on=["cod_ibge", "ano"])
isna = ov[["ISNA_floracao", "ISNA_enchimento"]].min(axis=1)
ov["stress"] = np.select([ov.geada_reprod >= 0.5, isna <= isna.quantile(0.2), ov.calor_floracao >= 1.0], ["frost/cold", "drought", "heat"], "no major stress")
FIN = "L_ensemble_E_SVR"; res = ov.log_y - ov[FIN]; cls = ov.stress.to_numpy()
cg, cm = np.zeros(len(ov), bool), np.zeros(len(ov), bool)
for a in ov.ano.unique():
    te, tr = (ov.ano == a).to_numpy(), (ov.ano != a).to_numpy()
    q = res[tr].quantile([.1, .9]); cg[te] = res[te].between(*q).to_numpy()
    for k in np.unique(cls[te]):
        r = res[tr & (cls == k)]; r = r if len(r) >= 100 else res[tr]
        m = te & (cls == k); cm[m] = res[m].between(*r.quantile([.1, .9])).to_numpy()
ov["cov_global"], ov["cov_stress"] = cg, cm
rm = lambda col: (lambda s: np.sqrt(((ov.loc[s.index, "rendimento_kg_ha"] - np.exp(ov.loc[s.index, col])) ** 2).mean()))
tab = ov.groupby("stress").agg(n=("ano", "size"), observed=("rendimento_kg_ha", "mean"),
                               RMSE_A=("ano", rm("A_media_historica")), RMSE_B=("ano", rm("B_MZA_calibrado")),
                               RMSE_L=("ano", rm(FIN)), cov_global=("cov_global", "mean"), cov_stress=("cov_stress", "mean"))
print(tab.round(3).to_string())
print("coverage 2020–2022: global", round(ov[ov.ano.between(2020, 2022)].cov_global.mean(), 3), "| conditional", round(ov[ov.ano.between(2020, 2022)].cov_stress.mean(), 3))
tab.to_csv(OUT / "validacao_por_estresse.csv")

# %% [markdown]
# ## 7. Lower tail and stability of yield by date and region (26 weather years at the 2025 technology level)

# %%
pq = prod[prod.ano.between(2000, 2025)]
stt = pq.groupby(["cod_ibge", "semeadura"]).rend_P50_kg_ha.agg(mean="mean", sd="std", p10=lambda s: s.quantile(.1)).reset_index()
stt["cv"] = stt.sd / stt["mean"]; stt["meso"] = stt.cod_ibge.map(meso)
print(stt.groupby("semeadura")[["mean", "p10", "cv"]].mean().reindex(ORDEM).round(3).T.to_string())
cvr = stt.pivot_table(index="meso", columns="semeadura", values="cv").reindex(columns=ORDEM)
print(cvr.round(3).to_string()); stt.to_csv(OUT / "cauda_estabilidade.csv", index=False)

# %% [markdown]
# ## 8. Drivers by mesoregion (SHAP of the monotonic gradient-boosting component, TreeSHAP)
# The SVR component is too slow for permutation SHAP on all rows; the monotonic HGB is half of the ensemble and shares
# the features.

# %%
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
VARS = DYN + ["altitude_m", "dta_mm_m", "latitude", "longitude"]
MONO = [1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0]
reg = LinearRegression().fit(base[["ano"]], base.log_y)
an = base.log_y - reg.predict(base[["ano"]]); an = an - an.groupby(base.cod_ibge).transform("mean")
hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=100,
                                    l2_regularization=1.0, monotonic_cst=MONO, random_state=42).fit(base[VARS], an)
sv = pd.DataFrame(np.abs(shap.TreeExplainer(hgb).shap_values(base[VARS])), columns=VARS)
GRP = {"EC": "water (PA/PPf)", "ISNA_vegetativo": "water, vegetative", "ISNA_floracao": "water, flowering",
       "DEF_floracao": "water, flowering", "ISNA_enchimento": "water, grain filling", "frio_reprod": "cold/frost, reproductive",
       "geada_reprod": "cold/frost, reproductive", "calor_floracao": "heat, flowering",
       "altitude_m": "static", "dta_mm_m": "static", "latitude": "static", "longitude": "static"}
sg = sv.T.groupby(sv.columns.map(GRP)).sum().T
sg["meso"] = base.cod_ibge.map(meso).to_numpy()
shr = sg.groupby("meso").mean(); shr = 100 * shr.div(shr.sum(1), axis=0)
print(shr.round(1).to_string()); shr.to_csv(OUT / "shap_mesorregiao.csv")

# %% [markdown]
# ## 9. Robustness tables from the exploratory runs (`notebooks/revisao_agsy/`)

# %%
for nome, arq in [("training with the observed sowing calendar (V0 = main)", "variantes_validacao.csv"),
                  ("root depth × extreme indices", "raiz_extremos.csv"),
                  ("date response calibrated on the PSR (not adopted)", "calibracao_data_psr.csv"),
                  ("tau by group and adverse selection", "tau_por_grupo.csv")]:
    if (OUT / arq).exists():
        print(f"\n== {nome} ==\n", pd.read_csv(OUT / arq).round(3).to_string())

# %% [markdown]
# ## 10. Figures

# %%
# Fig. A — sowing-date response: PSR indemnities vs model (within-season, municipality and season fixed effects)
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(ORDEM)); ax = axs[0]
for col, color, lab, mk in [("PSR", COR["ink"], "PSR: indemnified policies", "o"), ("model beta=1", COR["blue"], "Model with potential-yield term", "s"),
                            ("model beta=0", COR["gray"], "Model without it", "^")]:
    y = np.r_[0, prof[(col, "coef")].to_numpy()]; e = np.r_[0, prof[(col, "se")].to_numpy()]
    ax.plot(x, y, marker=mk, color=color, lw=2, ms=8, label=lab)
    if col == "PSR":
        ax.fill_between(x, y - 1.96 * e, y + 1.96 * e, color=color, alpha=0.12, lw=0)
ax.axhline(0, color=COR["ink2"], lw=0.8)
ax.set_xticks(x, ORDEM, rotation=45); ax.set_ylabel("Change in probability vs 5 Jan")
ax.set_title("(a) Within-season loss probability by sowing date", loc="left", fontsize=13); ax.legend(frameon=False, fontsize=11)
ax = axs[1]
lab = ["Observed (IBGE)", "Model with\npotential-yield term", "Model without it"]
vals = ibge["log yield per 10 d later"].to_numpy() * 100; err = 1.96 * ibge["se"].to_numpy() * 100
ax.bar(range(3), vals, yerr=err, color=[COR["ink"], COR["blue"], COR["gray"]], width=0.55, capsize=4, error_kw={"lw": 1})
ax.set_xticks(range(3), lab); ax.axhline(0, color=COR["ink2"], lw=0.8)
ax.set_ylabel("Yield change per 10 days later sowing (%)"); ax.set_title("(b) Season yield vs observed sowing calendar", loc="left", fontsize=13)
plt.tight_layout(); fig.savefig(FIG / "fig_resposta_data_psr.png", dpi=300); plt.close(fig)

# Fig. B — system view: efficient frontier example, decision gap map, soybean break-even, hedge ratio
fig = plt.figure(figsize=(16, 10.5))
a1 = fig.add_subplot(2, 2, 1); a2 = fig.add_subplot(2, 2, 2); a3 = fig.add_subplot(2, 2, 3); a4 = fig.add_subplot(2, 2, 4)
ex = {"Cascavel": COR["blue"], "Londrina": COR["orange"], "Umuarama": COR["aqua"]}
nomes = mun.set_index("municipio").cod_ibge
for nm, color in ex.items():
    g = ok[ok.cod_ibge == nomes[nm]].set_index("semeadura").reindex([d for d in ORDEM if d in set(ok[ok.cod_ibge == nomes[nm]].semeadura)])
    a1.plot(g.cvar10, g["mean"], "-", color=color, lw=1.5, alpha=0.7)
    a1.scatter(g.cvar10, g["mean"], s=np.where(g.efficient, 90, 40), color=color, edgecolor="white", lw=1.5, zorder=3, label=nm)
    for d in [g.index[0], g.index[-1]]:
        a1.annotate(d, (g.loc[d, "cvar10"], g.loc[d, "mean"]), textcoords="offset points", xytext=(6, 4), fontsize=10, color=COR["ink2"])
a1.set_xlabel(r"CVaR$_{10}$ of the margin (R\$ ha$^{-1}$)"); a1.set_ylabel(r"Mean margin (R\$ ha$^{-1}$)")
a1.set_title("(a) Return–risk of the sowing dates (large = efficient)", loc="left", fontsize=13); a1.legend(frameon=False, fontsize=11)
sm = mapas.mapa(a2, gap.set_index("cod_ibge").gap_CE.clip(lower=0), cmap="Blues", vmin=0, vmax=float(gap.gap_CE.quantile(.95)),
                titulo="(b) CE gain of the recommended date over the current\nsowing calendar (PSR 2022–2025), R\\$ ha$^{-1}$")
fig.colorbar(sm, ax=a2, shrink=0.7)
a3.bar(range(len(soja)), soja["soybean kg/ha (mean)"], color=COR["blue"], width=0.55)
a3.vlines(range(len(soja)), soja["soybean kg/ha P10"], soja["soybean kg/ha P90"], color=COR["ink"], lw=1.5)
a3.set_xticks(range(len(soja)), soja.index); a3.set_xlabel("Maize cannot be sown before (soybean not yet harvested)")
a3.set_ylabel(r"Tolerable soybean loss (kg ha$^{-1}$)"); a3.set_title("(c) Soybean yield that early maize can pay for (mean, P10–P90)", loc="left", fontsize=13)
for i, v in enumerate(soja["% of soybean yield"]):
    a3.text(i, soja["soybean kg/ha P90"].iloc[i] + 8, f"{v:.0f}%", ha="center", fontsize=10, color=COR["ink2"])
hs = [0, 0.25, 0.5, 0.75, 1.0]; keys = ["spot (base)"] + [f"forward h={h:.2f}" for h in hs[1:]]
a4.plot(hs, fw.loc[keys, "CE"] - fw.loc["spot (base)", "CE"], "o-", color=COR["blue"], lw=2, ms=8, label="Certainty equivalent")
a4.plot(hs, fw.loc[keys, "CVaR10"] - fw.loc["spot (base)", "CVaR10"], "s-", color=COR["orange"], lw=2, ms=8, label=r"CVaR$_{10}$ (worst 10%)")
a4.axhline(0, color=COR["ink2"], lw=0.8)
a4.set_xlabel("Share of expected production sold forward in January"); a4.set_ylabel(r"Change vs selling at harvest (R\$ ha$^{-1}$)")
a4.set_title("(d) Forward pricing: partial hedge pays, full hedge does not", loc="left", fontsize=13); a4.legend(frameon=False, fontsize=10)
plt.tight_layout(w_pad=3); fig.savefig(FIG / "fig_sistema_decisao.png", dpi=300); plt.close(fig)

for f in ["fig_resposta_data_psr.png", "fig_sistema_decisao.png"]:
    if ART.exists():
        import shutil; shutil.copy(FIG / f, ART / f)
print("figures ok")
