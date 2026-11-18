"""Script version of notebook 02 (Windows needs the __main__ guard for process pools).
Usage (cwd = notebooks): SENS=<scenario or empty> python revisao_agsy/rodar_mza.py"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))
import modelo_mza as M
from sensibilidade import PARAM, SAIDA

RAIZ = Path.cwd().parent
DADOS = RAIZ / "dados"
ANOS = range(1999, 2026)
DATAS = [(m, d) for m in (1, 2, 3) for d in (5, 15, 25)]

if __name__ == "__main__":
    t0 = time.time()
    CT = PARAM.get("ct", M.CT)
    print("scenario:", os.environ.get("SENS", "") or "base", "| z_max =", M.Z_MAX, "| CT =", CT, flush=True)
    if not (DADOS / "xavier_diario_pr.parquet").exists():
        pd.read_csv(DADOS / "xavier_diario_pr.csv.gz", parse_dates=["data"]).to_parquet(DADOS / "xavier_diario_pr.parquet", index=False)
    mun = pd.read_csv(DADOS / "municipios_pr.csv")
    arq = DADOS / "xavier_diario_pr.parquet"
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        fut = [ex.submit(M.simular_municipio, r.cod_ibge, r.latitude, r.dta_mm_m, arq, ANOS, DATAS, CT) for r in mun.itertuples()]
        partes = []
        for i, f in enumerate(fut, 1):
            partes.append(f.result())
            if i % 50 == 0:
                print(f"  {i}/{len(fut)} municipalities, {time.time() - t0:.0f} s", flush=True)
    sim = pd.concat(partes, ignore_index=True)
    sim.to_parquet(SAIDA / "simulacao_mza_pr.parquet", index=False)
    print(sim.shape, "saved to", SAIDA, f"{time.time() - t0:.0f} s", flush=True)
