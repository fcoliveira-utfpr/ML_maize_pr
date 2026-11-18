"""Revision analyses, part 6: root depth (0.40 / 0.60 / 0.80 m) x stage-specific extreme indices.
Each configuration: leave-one-year-out accuracy of the final model (E monotonic HGB + SVR, window mean of 4 dates)
and the within-season sowing-date gradient against the PSR (with the PPf physics offset, beta = 1).
Run with cwd = notebooks; argv[1] = psr_pr.parquet."""
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

RAIZ = Path.cwd().parent; DADOS, RES = RAIZ / "dados", RAIZ / "resultados"; OUT = RES / "revisao"
ORDEM = [f"{d:02d}/{m:02d}" for m in (1, 2, 3) for d in (5, 15, 25)]
DOY = np.array([5, 15, 25, 36, 46, 56, 64, 74, 84]); JANELA = ["15/01", "05/02", "25/02", "15/03"]
FASES = ["estabelecimento", "vegetativo", "floracao", "enchimento", "maturacao"]
STAT = ["altitude_m", "dta_mm_m", "latitude", "longitude"]
BASE_VARS = {"EC": 1, "ISNA_vegetativo": 1, "ISNA_floracao": 1, "ISNA_enchimento": 1, "DEF_floracao": -1,
             "frio_reprod": -1, "geada_reprod": -1, "calor_floracao": -1}
EXTREMOS = {"seca_seq_floracao": -1, "seca_seq_enchimento": -1, "frio_acum_enchimento": -1,
            "frio_acum_maturacao": -1, "calor_acum_floracao": -1}
SEED = 42
mun = pd.read_csv(DADOS / "municipios_pr.csv")
pam = pd.read_csv(DADOS / "ibge_pam_milho_pr.csv")
pam = pam[(pam.safra == 2) & (pam.rendimento_kg_ha > 0) & (pam.area_colhida_ha >= 100)].sort_values(["cod_ibge", "ano"])
pam = pam[~(pam.groupby("cod_ibge").rendimento_kg_ha.diff() == 0)]


def prep(path):
    sim = pd.read_parquet(path)
    fc = [c for c in sim.columns if any(c.endswith("_" + f) for f in FASES)]
    cnt = [c for c in fc if c.startswith(("geada_", "calor_", "seca_seq_", "frio_acum_", "calor_acum_"))]
    sim[cnt] = sim[cnt].fillna(0); sim[fc] = sim[fc].fillna(sim[fc].median())
    sim["EC"] = sim.PA / sim.PPf
    sim["geada_reprod"] = sim.geada_floracao + sim.geada_enchimento
    sim["frio_reprod"] = np.clip(2 - sim[["Tmin_floracao", "Tmin_enchimento"]].min(axis=1), 0, None)
    sim["lppf"] = np.log(sim.PPf)
    return sim


def gbm(mono):
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=100,
                                         l2_regularization=1.0, monotonic_cst=mono, random_state=SEED)


def svr():
    return make_pipeline(StandardScaler(), SVR(C=0.3, gamma=0.05, epsilon=0.05))


def level(tr):
    reg = LinearRegression().fit(tr[["ano"]], tr.log_y)
    return reg, (tr.log_y - reg.predict(tr[["ano"]])).groupby(tr.cod_ibge).mean()


# PSR cells (municipality x season x sowing date)
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
cell["doy"] = cell.data.map(dict(zip(ORDEM, DOY))); cell["late"] = cell.data.isin(["05/03", "15/03", "25/03"]).astype(int)
cell["feb"] = cell.data.isin(["05/02", "15/02", "25/02"]).astype(int)
psr_grad = smf.wls("ind ~ feb + late + C(ano) + C(cod_ibge)", data=cell, weights=cell.n).fit()
print(f"PSR: Feb vs Jan {psr_grad.params.feb:.3f}, Mar vs Jan {psr_grad.params.late:.3f}")

CONFIGS = {"z0.40": RES / "simulacao_mza_pr.parquet", "z0.60": RES / "sensibilidade/Z060/simulacao_mza_pr.parquet",
           "z0.80": RES / "sensibilidade/Z080/simulacao_mza_pr.parquet"}
rows = []
for zname, path in CONFIGS.items():
    if not path.exists():
        print("missing", path); continue
    sim = prep(path)
    for xname, extra in [("base vars", {}), ("+ extremes", EXTREMOS)]:
        V = {**BASE_VARS, **extra}; DYN = list(V); VARS = DYN + STAT; MONO = list(V.values()) + [0] * 4
        VSVR = [v for v in VARS if v not in ("EC", "DEF_floracao")]
        wm = sim[sim.semeadura.isin(JANELA)].groupby(["cod_ibge", "ano"])[DYN].mean().reset_index()
        b = pam[["cod_ibge", "ano", "rendimento_kg_ha"]].merge(wm, on=["cod_ibge", "ano"]).merge(mun[["cod_ibge"] + STAT], on="cod_ibge")
        b["log_y"] = np.log(b.rendimento_kg_ha)
        pred = np.zeros(len(b))
        for ano in sorted(b.ano.unique()):
            tr, te = (b.ano != ano).to_numpy(), (b.ano == ano).to_numpy()
            reg, eff = level(b[tr])
            bp = lambda m: reg.predict(b.loc[m, ["ano"]]) + b.loc[m, "cod_ibge"].map(eff).fillna(0).to_numpy()
            an = b.log_y[tr] - bp(tr)
            e = gbm(MONO).fit(b.loc[tr, VARS], an); s = svr().fit(b.loc[tr, VSVR], an)
            pred[te] = bp(te) + (e.predict(b.loc[te, VARS]) + s.predict(b.loc[te, VSVR])) / 2
        obs = b.rendimento_kg_ha.to_numpy(); ph = np.exp(pred)
        st = pd.DataFrame({"o": obs, "p": ph, "a": b.ano}).groupby("a").mean()
        res_sd = np.std(b.log_y - pred)
        # full-data model -> per-date prediction with the PPf offset
        reg, eff = level(b); an = b.log_y - reg.predict(b[["ano"]]) - b.cod_ibge.map(eff).to_numpy()
        e = gbm(MONO).fit(b[VARS], an); s = svr().fit(b[VSVR], an)
        P = sim[sim.cod_ibge.isin(b.cod_ibge.unique())].merge(mun[["cod_ibge"] + STAT], on="cod_ibge")
        lw = P[P.semeadura.isin(JANELA)].groupby(["cod_ibge", "ano"]).lppf.mean().rename("lw")
        P = P.join(lw, on=["cod_ibge", "ano"])
        P["mu"] = reg.predict(P[["ano"]]) + P.cod_ibge.map(eff).to_numpy() + (e.predict(P[VARS]) + s.predict(P[VSVR])) / 2 + (P.lppf - P.lw)
        ye = P[P.semeadura.isin(JANELA) & P.ano.between(2000, 2025)].groupby("cod_ibge").mu.apply(lambda x: np.log(np.exp(x).mean()))
        P["p65"] = norm.cdf((np.log(0.65) + P.cod_ibge.map(ye) - P.mu) / np.sqrt(res_sd ** 2 + 0.33 ** 2))
        c = cell.merge(P[["cod_ibge", "ano", "semeadura", "p65", "mu"]].rename(columns={"semeadura": "data"}), on=["cod_ibge", "ano", "data"])
        mm = smf.wls("p65 ~ feb + late + C(ano) + C(cod_ibge)", data=c, weights=c.n).fit()
        bydate = np.exp(P.groupby("semeadura").mu.mean()).reindex(ORDEM)
        rows.append({"roots": zname, "features": xname, "R2_mun": r2_score(obs, ph), "R2_annual": r2_score(st.o, st.p),
                     "RMSE": np.sqrt(np.mean((obs - ph) ** 2)), "model_Feb_vs_Jan": mm.params.feb, "model_Mar_vs_Jan": mm.params.late,
                     "brier": np.average((c.ind - c.p65) ** 2, weights=c.n),
                     "y_05jan": bydate["05/01"], "y_25feb": bydate["25/02"], "y_15mar": bydate["15/03"]})
        print(rows[-1], flush=True)
out = pd.DataFrame(rows)
print("\nPSR reference: Feb vs Jan %.3f, Mar vs Jan %.3f" % (psr_grad.params.feb, psr_grad.params.late))
print(out.round(3).to_string())
out.to_csv(OUT / "raiz_extremos.csv", index=False)
