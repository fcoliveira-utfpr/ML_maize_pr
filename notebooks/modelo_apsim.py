"""APSIM Next Generation (Maize) sem calibração, para gerar variáveis do modelo híbrido.

Uma simulação por (município, data de semeadura), contínua de 1998-12 a 2025-12, semeando todo ano na
mesma data. Parte do exemplo oficial `Examples/Maize.apsimx` e troca: clima (BR-DWGD), perfil físico
do solo (granulometria do MapBiomas Solo + AWC do município), regra de semeadura e cultivar.
"""
import copy
import json
import os
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

APSIM = Path.home() / "Python" / "apsimx"
MODELS_DLL = APSIM / "bin" / "Models.dll"
EXEMPLO = APSIM / "Examples" / "Maize.apsimx"

CULTIVAR = "A_100"           # genérica, maturidade relativa 100 d, sem sensibilidade ao fotoperíodo
POPULACAO = 6.0              # plantas m-2
ADUBO_N = 200.0              # kg N ha-1 na semeadura: nitrogênio não limitante (foco no clima)
INICIO, FIM = "1998-12-01", "2025-12-31"

# Camadas do exemplo (mm) e camadas de 10 cm do MapBiomas Solo que caem em cada uma
ESPESSURA = [150, 150, 300, 300, 300, 300, 300]
MB_CAMADAS = [[1], [2], [3, 4, 5], [6, 7, 8], [9, 10], [10], [10]]
KL = [0.08, 0.08, 0.08, 0.06, 0.06, 0.04, 0.03]
MATERIA_ORGANICA = [2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5]  # % (suposição: perfil típico de Latossolo cultivado)

VARIAVEIS = [
    "[Clock].Today",
    "[Maize].Phenology.Stage",
    "[Maize].Grain.Total.Wt*10 as rend_kg_ha",
    "[Maize].AboveGround.Wt*10 as biomassa_kg_ha",
    "[Maize].Grain.NumberFunction as n_graos",
    "[Maize].Leaf.LAI as iaf",
    "[Maize].Leaf.Fw as fw",
    "[Maize].Leaf.Fn as fn",
    "[Weather].MinT as tmin",
]


# ------------------------------------------------------------------ solo
def saxton_rawls(areia, argila, mo):
    """θ1500, θ33 e θs (m3 m-3) — Saxton & Rawls (2006), frações (0–1) e MO em %."""
    S, C, OM = areia, argila, mo
    t1500t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    t1500 = t1500t + (0.14 * t1500t - 0.02)
    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    t33 = t33t + (1.283 * t33t ** 2 - 0.374 * t33t - 0.015)
    ts33t = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM - 0.584 * S * C + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    ts = t33 + ts33 - 0.097 * S + 0.043
    lam = (np.log(t33) - np.log(t1500)) / (np.log(1500) - np.log(33))
    ks = 1930 * (ts - t33) ** (3 - lam) * 24  # mm/dia
    return t1500, t33, ts, ks


def perfil(camadas_mun, dta_mm_m):
    """Perfil físico em 7 camadas. A água disponível (DUL − LL15) é escalada para a AWC do município."""
    g = camadas_mun.set_index("camada")
    areia = np.array([g.loc[c, "areia_pct"].mean() for c in MB_CAMADAS]) / 100
    argila = np.array([g.loc[c, "argila_pct"].mean() for c in MB_CAMADAS]) / 100
    ll15, dul, sat, ks = saxton_rawls(areia, argila, np.array(MATERIA_ORGANICA))
    fator = (dta_mm_m / 1000) / np.average(dul - ll15, weights=ESPESSURA)
    dul = ll15 + (dul - ll15) * fator
    sat = np.maximum(sat, dul + 0.03)
    bd = (1 - sat) * 2.65
    return {"LL15": ll15, "DUL": dul, "SAT": sat, "BD": bd, "KS": ks, "AirDry": np.r_[0.5 * ll15[0], ll15[1:]]}


# ------------------------------------------------------------------ clima
def escrever_met(clima_mun, lat, caminho):
    c = clima_mun.reset_index()
    tmed = (c["Tmax"] + c["Tmin"]) / 2
    es = 6.108 * np.exp(17.27 * tmed / (tmed + 237.3))  # hPa
    mensal = tmed.groupby([c["data"].dt.year, c["data"].dt.month]).mean().groupby(level=1).mean()
    linhas = [
        "[weather.met.weather]", f"latitude = {lat:.3f} (DECIMAL DEGREES)",
        f"tav = {tmed.mean():.2f} (oC)", f"amp = {mensal.max() - mensal.min():.2f} (oC)", "",
        "year day radn maxt mint rain vp wind", "() () (MJ/m^2) (oC) (oC) (mm) (hPa) (m/s)",
    ]
    tab = pd.DataFrame({"year": c["data"].dt.year, "day": c["data"].dt.dayofyear, "radn": c["Rs"].round(2),
                        "maxt": c["Tmax"].round(2), "mint": c["Tmin"].round(2), "rain": c["pr"].clip(lower=0).round(2),
                        "vp": (c["RH"] / 100 * es).round(2), "wind": c["u2"].round(2)})
    Path(caminho).write_text("\n".join(linhas) + "\n" + tab.to_string(header=False, index=False) + "\n")


# ------------------------------------------------------------------ arquivo .apsimx
def _filhos(no, tipo=None, nome=None):
    for f in no.get("Children", []):
        if (tipo is None or f.get("$type", "").startswith(tipo)) and (nome is None or f.get("Name") == nome):
            yield f
        yield from _filhos(f, tipo, nome)


def _um(no, tipo=None, nome=None):
    return next(_filhos(no, tipo, nome))


def montar(cod, met, solo, datas_semeadura, caminho):
    """Um .apsimx com uma simulação por data de semeadura (mesmo município, clima e solo)."""
    base = json.loads(EXEMPLO.read_text())
    modelo = _um(base, "Models.Core.Simulation")
    for g in list(_filhos(modelo, "Models.Graph")):
        pass
    modelo["Children"] = [f for f in modelo["Children"] if not f.get("$type", "").startswith("Models.Graph")]
    base["Children"] = [f for f in base["Children"] if not f.get("$type", "").startswith("Models.Core.Simulation")]

    for mes, dia in datas_semeadura:
        sim = copy.deepcopy(modelo)
        sim["Name"] = f"m{cod}_d{dia:02d}{mes:02d}"
        _um(sim, "Models.Clock").update({"Start": f"{INICIO}T00:00:00", "End": f"{FIM}T00:00:00"})
        _um(sim, "Models.Climate.Weather")["FileName"] = str(met)
        fis = _um(sim, "Models.Soils.Physical")
        for k, v in solo.items():
            fis[k] = [float(x) for x in v]
        cultura = _um(fis, nome="MaizeSoil")
        cultura.update({"LL": [float(x) for x in solo["LL15"]], "KL": KL, "XF": [1.0] * len(ESPESSURA)})
        agua = _um(sim, "Models.Soils.Water")
        agua["InitialValues"] = [float(x) for x in solo["DUL"]]
        agua["RelativeTo"] = "LL15"
        data_txt = pd.Timestamp(2000, mes, dia).strftime("%d-%b").lower()
        for m in _filhos(sim, "Models.Manager"):
            p = {d["Key"]: d for d in m.get("Parameters", [])}
            if "CultivarName" in p:
                for k, v in {"StartDate": data_txt, "EndDate": data_txt, "MinESW": "-1", "MinRain": "-1",
                             "RainDays": "1", "CultivarName": CULTIVAR, "Population": str(POPULACAO)}.items():
                    p[k]["Value"] = v
            if "Amount" in p:
                p["Amount"]["Value"] = str(ADUBO_N)
        rep = _um(sim, "Models.Report")
        rep["VariableNames"] = VARIAVEIS
        rep["EventNames"] = ["[Clock].DoReport"]
        base["Children"].insert(0, sim)
    Path(caminho).write_text(json.dumps(base))


def rodar(apsimx, cpus=1):
    env = dict(os.environ, DOTNET_ROOT=str(Path.home() / ".dotnet"))
    dotnet = str(Path.home() / ".dotnet" / "dotnet")
    r = subprocess.run([dotnet, str(MODELS_DLL), str(apsimx), f"--cpu-count={cpus}"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    db = Path(apsimx).with_suffix(".db")
    with sqlite3.connect(db) as con:
        sims = pd.read_sql("select ID as SimulationID, Name from _Simulations", con)
        dados = pd.read_sql("select * from Report", con).merge(sims, on="SimulationID")
    return dados


# Estádios do Maize no APSIM NG: 1 semeadura, 2 germinação, 3 emergência, 4 fim do juvenil,
# 5 iniciação floral, 6 folha bandeira, 7 florescimento, 8 início do enchimento, 9 fim do enchimento,
# 10 maturidade, 11 ponto de colheita.
PERIODOS = {"veg": (3, 6), "flor": (6, 8), "ench": (8, 10)}


def resumir(dados, cod):
    """Uma linha por (ano, data de semeadura) com as variáveis do ciclo."""
    d = dados.copy()
    d["t"] = pd.to_datetime(d["Clock.Today"])
    d["estadio"] = d["Maize.Phenology.Stage"]
    linhas = []
    for nome, g in d.groupby("Name"):
        ddmm = nome.split("_d")[1]
        dia, mes = int(ddmm[:2]), int(ddmm[2:])
        g = g.set_index("t").sort_index()
        for ano in range(1999, 2026):
            sem = pd.Timestamp(ano, mes, dia)
            c = g.loc[sem: sem + pd.Timedelta(days=300)]
            c = c[c["estadio"].cummax().le(11)]
            fim = c.index[c["rend_kg_ha"].eq(0) & c["estadio"].shift().ge(10)]
            c = c.loc[: fim[0]] if len(fim) else c
            if c.empty or c["estadio"].max() < 2:
                continue
            lin = {"cod_ibge": cod, "ano": ano, "semeadura": f"{dia:02d}/{mes:02d}",
                   "ap_rend": c["rend_kg_ha"].max(), "ap_biomassa": c["biomassa_kg_ha"].max(),
                   "ap_n_graos": c["n_graos"].max(), "ap_iaf_max": c["iaf"].max(),
                   "ap_maturou": bool(c["estadio"].max() >= 10)}
            flor = c.index[c["estadio"] >= 7]
            mat = c.index[c["estadio"] >= 10]
            lin["ap_dias_floracao"] = (flor[0] - sem).days if len(flor) else np.nan
            lin["ap_ciclo_dias"] = (mat[0] - sem).days if len(mat) else np.nan
            for p, (e0, e1) in PERIODOS.items():
                s = c[(c["estadio"] >= e0) & (c["estadio"] < e1)]
                lin[f"ap_estresse_{p}"] = (1 - s["fw"]).sum() if len(s) else np.nan  # estresse hídrico acumulado
                lin[f"ap_tmin_{p}"] = s["tmin"].min() if len(s) else np.nan
            linhas.append(lin)
    return pd.DataFrame(linhas)
