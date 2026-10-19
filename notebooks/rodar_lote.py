"""Executa um modelo de cultura para vários municípios em paralelo, com checkpoint por município."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"


def _um_municipio(args):
    modelo, cod, datas, pasta = args
    import modelo_apsim as A
    mun = pd.read_csv(DADOS / "municipios_pr.csv").set_index("cod_ibge").loc[cod]
    camadas = pd.read_csv(DADOS / "solo_camadas_pr.csv")
    clima = pd.read_parquet(DADOS / "xavier_diario_pr.parquet", filters=[("cod_ibge", "==", cod)])
    clima = clima.set_index("data").sort_index()
    perfil = A.perfil(camadas[camadas["cod_ibge"] == cod], mun["dta_mm_m"])
    if modelo == "aquacrop":
        import modelo_aquacrop as Q
        r = Q.simular_municipio(cod, clima, perfil, datas, clima["Tmin"])
    elif modelo == "dssat":
        import modelo_dssat as D
        r = D.simular_municipio(cod, clima, perfil, mun, datas)
    else:
        raise ValueError(modelo)
    r.to_parquet(Path(pasta) / f"{cod}.parquet", index=False)
    return cod


def rodar(modelo, codigos, datas, pasta, processos=4):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    feitos = {int(p.stem) for p in pasta.glob("*.parquet")}
    tarefas = [(modelo, int(c), datas, str(pasta)) for c in codigos if int(c) not in feitos]
    falhas = {}
    with ProcessPoolExecutor(max_workers=processos) as ex:
        futuros = {ex.submit(_um_municipio, t): t[1] for t in tarefas}
        for f in as_completed(futuros):
            try:
                f.result()
            except Exception as erro:
                falhas[futuros[f]] = str(erro)[:300]
    return falhas
