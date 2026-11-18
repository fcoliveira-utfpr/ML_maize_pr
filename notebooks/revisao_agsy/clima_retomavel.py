"""Resumable version of notebook 01: one parquet per month in dados/xavier_cache/, then merged into
dados/xavier_diario_pr.csv.gz (same output and quality control as notebook 01). Re-run to resume.
Run with cwd = ML_maize_pr/notebooks."""
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd

socket.setdefaulttimeout(180)
RAIZ = Path.cwd().parent
DADOS = RAIZ / "dados"
CACHE = DADOS / "xavier_cache"; CACHE.mkdir(exist_ok=True)
SAIDA = DADOS / "xavier_diario_pr.csv.gz"
ASSET = "projects/ee-alexandrexavier/assets/BR-DWGD"
BANDAS = ["pr", "Tmax", "Tmin", "RH", "Rs", "u2", "ET"]

ee.Initialize(project="mapbiomas-brazil")
colecao = ee.ImageCollection(ASSET)
props = colecao.first().toDictionary().getInfo()
ESCALA = {b: (props[f"BAND_{b}_MULT"], props[f"BAND_{b}_ADD"]) for b in BANDAS}
municipios = pd.read_csv(DADOS / "municipios_pr.csv")
pontos = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.longitude, r.latitude]), {"cod_ibge": int(r.cod_ibge)})
                               for r in municipios.itertuples()])
projecao = colecao.first().select("pr").projection()


def extrair(inicio, fim):
    arq = CACHE / f"{inicio[:7]}.parquet"
    if arq.exists():
        return arq
    imgs = colecao.filterDate(inicio, fim).select(BANDAS)
    datas = imgs.aggregate_array("system:index").getInfo()
    feats = imgs.toBands().sampleRegions(collection=pontos, properties=["cod_ibge"], projection=projecao,
                                         geometries=False).getInfo()["features"]
    linhas = []
    for f in feats:
        p = f["properties"]
        for dia in datas:
            reg = {"cod_ibge": p["cod_ibge"], "data": dia[-8:]}
            for b in BANDAS:
                v = p.get(f"{dia}_{b}"); mult, add = ESCALA[b]
                reg[b] = None if v is None else v * mult + add
            linhas.append(reg)
    pd.DataFrame(linhas).to_parquet(arq, index=False)
    return arq


meses = pd.date_range("1998-12-01", "2026-01-01", freq="MS")
periodos = [(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) for a, b in zip(meses[:-1], meses[1:])]
for tentativa in range(5):
    faltam = [p for p in periodos if not (CACHE / f"{p[0][:7]}.parquet").exists()]
    print(f"attempt {tentativa + 1}: {len(faltam)} months missing", flush=True)
    if not faltam:
        break
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(extrair, a, b): a for a, b in faltam}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                f.result()
            except Exception as e:
                print("  failed", fut[f], str(e)[:120], flush=True)
            if i % 25 == 0:
                print(f"  {i}/{len(faltam)}", flush=True)
faltam = [p for p in periodos if not (CACHE / f"{p[0][:7]}.parquet").exists()]
if faltam:
    sys.exit(f"still missing {len(faltam)} months; re-run to resume")

clima = pd.concat([pd.read_parquet(CACHE / f"{a[:7]}.parquet") for a, _ in periodos], ignore_index=True)
clima["data"] = pd.to_datetime(clima["data"], format="%Y%m%d")
clima = clima.sort_values(["cod_ibge", "data"]).reset_index(drop=True)
# quality control as notebook 01: seats without data -> mean of the 3 nearest municipalities
falt = clima[BANDAS].isna().groupby(clima["cod_ibge"]).mean()
sem = falt.index[falt.max(axis=1) > 0.5].tolist()
print("municipalities without data:", sem)
if sem:
    coords = municipios.set_index("cod_ibge")[["latitude", "longitude"]]
    comp = coords.drop(index=sem)
    piv = {b: clima.pivot(index="data", columns="cod_ibge", values=b) for b in BANDAS}
    for cod in sem:
        viz = comp.index[np.argsort(np.hypot(*(comp - coords.loc[cod]).T.values))[:3]]
        for b in BANDAS:
            piv[b][cod] = piv[b][viz].mean(axis=1)
    clima = pd.concat({b: p.stack() for b, p in piv.items()}, axis=1).reset_index()
    clima.columns = ["data", "cod_ibge"] + BANDAS
print(clima.shape, clima.data.min(), clima.data.max())
clima.round(3).to_csv(SAIDA, index=False, compression="gzip")
print("saved", SAIDA, flush=True)
