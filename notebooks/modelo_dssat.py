"""DSSAT CSM-CERES-Maize sem calibração (via `DSSATTools`), para o modelo híbrido.

Uma rodada sazonal por (município, data de semeadura): 27 safras (1999–2025) semeadas na mesma data,
cultivar genérica de ciclo curto do DSSAT, clima BR-DWGD e o mesmo perfil de solo do APSIM/AquaCrop.
O executável do `DSSATTools` é de Linux; no macOS usa-se o DSSAT-CSM compilado do código aberto
(`DSSAT/dssat-csm-os`), apontado em `BINARIO`.
"""
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# O DSSAT (Fortran) aceita caminhos de até 80 caracteres; a pasta temporária do macOS é mais longa.
# Antes de importar o DSSATTools (que monta o DSSAT_HOME na pasta temporária), usa-se uma pasta curta.
TMP_CURTO = Path.home() / ".dssat_tmp"
TMP_CURTO.mkdir(exist_ok=True)
tempfile.tempdir = str(TMP_CURTO)

import DSSATTools.run as dssat_run  # noqa: E402
from DSSATTools.crop import Maize
from DSSATTools.filex import (Field, Fertilizer, FertilizerEvent, InitialConditions, InitialConditionsLayer,
                              Planting, SCGeneral, SCManagement, SCMethods, SCOptions, SCOutputs,
                              SimulationControls)
from DSSATTools.run import DSSAT
from DSSATTools.soil import SoilLayer, SoilProfile
from DSSATTools.weather import WeatherStation

import modelo_apsim as A

BINARIO = Path.home() / "Python" / "dssat-csm-os" / "build" / "bin" / "dscsm048"
if BINARIO.exists():
    dssat_run.BIN_PATH = str(BINARIO)

CULTIVAR = "990003"           # SHORT SEASON (genérica do MZCER048.CUL)
POPULACAO = 6.0               # plantas m-2
ADUBO_N = 200                 # kg N ha-1 na semeadura (N não limitante)
FRACAO_AD_INICIAL = 0.7
ANO_INI, NYERS = 1999, 27
BASES_CM = np.cumsum(A.ESPESSURA) / 10  # 15, 30, 60, ..., 180


def estacao(clima_mun, mun, insi="PRMZ"):
    c = clima_mun.reset_index().sort_values("data")
    tmed = (c["Tmax"] + c["Tmin"]) / 2
    es = 0.6108 * np.exp(17.27 * tmed / (tmed + 237.3))
    ea = (c["RH"] / 100 * es).clip(lower=0.05)
    dewp = (237.3 * np.log(ea / 0.6108)) / (17.27 - np.log(ea / 0.6108))
    tab = pd.DataFrame({"date": pd.to_datetime(c["data"]), "srad": c["Rs"].round(1), "tmax": c["Tmax"].round(1),
                        "tmin": c["Tmin"].round(1), "rain": c["pr"].clip(lower=0).round(1),
                        "dewp": np.minimum(dewp, c["Tmin"] + 2).round(1), "wind": (c["u2"] * 86.4).round(0)})
    mens = tmed.groupby(pd.to_datetime(c["data"]).dt.month).mean()
    return WeatherStation(table=tab, lat=round(mun["latitude"], 3), long=round(mun["longitude"], 3), insi=insi,
                          elev=round(mun["altitude_m"]), tav=round(tmed.mean(), 1), amp=round(mens.max() - mens.min(), 1),
                          refht=2.0, wndht=2.0)


def solo(perfil, camadas_mun, mun):
    g = camadas_mun.set_index("camada")
    argila = np.array([g.loc[c, "argila_pct"].mean() for c in A.MB_CAMADAS])
    areia = np.array([g.loc[c, "areia_pct"].mean() for c in A.MB_CAMADAS])
    topo = np.r_[0, BASES_CM[:-1]]
    linhas = [SoilLayer(slb=float(b), slll=round(float(ll), 3), sdul=round(float(du), 3), ssat=round(float(sa), 3),
                        sbdm=round(float(bd), 2), sloc=round(mo / 1.724, 2), slcl=round(float(ar), 1),
                        slsi=round(float(max(100 - ar - ae, 0)), 1), slhw=5.5,
                        srgf=1.0 if b <= 15 else round(float(np.exp(-0.02 * (t + b) / 2)), 3),
                        slni=round(mo / 1.724 / 12, 3), ssks=round(float(ks) / 10 / 24, 2))
              for b, t, ll, du, sa, bd, ks, mo, ar, ae in zip(BASES_CM, topo, perfil["LL15"], perfil["DUL"],
                                                                perfil["SAT"], perfil["BD"], perfil["KS"],
                                                                A.MATERIA_ORGANICA, argila, areia)]
    return SoilProfile(table=linhas, name="PRMZ000001", soil_data_source="MapBiomas", soil_clasification="C",
                       soil_series_name="Perfil municipal", site="PR", country="Brazil",
                       lat=round(mun["latitude"], 3), long=round(mun["longitude"], 3), scom="R", salb=0.13,
                       slu1=6.0, sldr=0.60, slro=73, slnf=1.0, slpf=1.0)


def _jd(s):
    return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%j", errors="coerce")


def _ler_summary(txt):
    linhas = txt.splitlines()
    i = next(k for k, l in enumerate(linhas) if l.startswith("@"))
    cab = linhas[i].replace("@", " ").split()
    dados = [l.split() for l in linhas[i + 1:] if l.strip()]
    df = pd.DataFrame([d for d in dados if len(d) == len(cab)], columns=cab)
    num = df.apply(pd.to_numeric, errors="coerce")
    return num.where(num.notna(), df).replace(-99, np.nan)


def _ler_plantgro(txt):
    """PlantGro.OUT com um bloco por safra (cabeçalho '@YEAR DOY ...' repetido)."""
    blocos, cab = [], None
    for l in txt.splitlines():
        if l.startswith("@"):
            cab = l.replace("@", " ").split()
        elif cab and l.strip() and l.strip()[0].isdigit():
            v = l.split()
            if len(v) == len(cab):
                blocos.append(v)
    df = pd.DataFrame(blocos, columns=cab).apply(pd.to_numeric, errors="coerce")
    df["data"] = pd.to_datetime(df["YEAR"].astype(int).astype(str) + df["DOY"].astype(int).astype(str).str.zfill(3),
                                format="%Y%j")
    return df.set_index("data").sort_index()


def simular_municipio(cod, clima_mun, perfil, mun, datas):
    camadas = pd.read_csv(A.Path(__file__).resolve().parent.parent / "dados" / "solo_camadas_pr.csv")
    ws, sp = estacao(clima_mun, mun), solo(perfil, camadas[camadas["cod_ibge"] == cod], mun)
    campo = Field(id_field="PRMZ0001", wsta=ws, id_soil=sp, flob=0, fldt="DR000", fldd=0, flds=0, sltx="C", sldp=180)
    tmin = clima_mun["Tmin"]
    linhas = []
    with tempfile.TemporaryDirectory(prefix="r", dir=TMP_CURTO) as pasta:
        env = DSSAT(pasta)
        for mes, dia in datas:
            pdate = date(ANO_INI, mes, dia)
            sdate = pdate - timedelta(days=30)
            sh2o = perfil["LL15"] + FRACAO_AD_INICIAL * (perfil["DUL"] - perfil["LL15"])
            ic = InitialConditions(pcr="SB", icdat=sdate, icres=3000, icren=0.8, icrip=0, icrid=0, icrn=1, icre=1,
                                   table=[InitialConditionsLayer(icbl=float(b), sh2o=round(float(w), 3), snh4=0.5, sno3=2.0)
                                          for b, w in zip(BASES_CM, sh2o)])
            ctrl = SimulationControls(
                general=SCGeneral(sdate=sdate, nyers=NYERS),
                options=SCOptions(water="Y", nitro="Y", symbi="N"),
                methods=SCMethods(evapo="F"),
                management=SCManagement(irrig="N", ferti="R", harvs="M"),
                outputs=SCOutputs(grout="Y", waout="N", caout="N"))
            adubo = Fertilizer(table=[FertilizerEvent(fdate=pdate, fmcd="FE005", facd="AP002", fdep=5, famn=ADUBO_N)])
            try:
                env.run_treatment(field=campo, cultivar=Maize(CULTIVAR),
                                  planting=Planting(pdate=pdate, ppop=POPULACAO, ppoe=POPULACAO, plrs=50, pldp=5),
                                  initial_conditions=ic, fertilizer=adubo, simulation_controls=ctrl, verbose=False)
            except ValueError:
                pass  # o DSSATTools não lê tabelas com várias safras no pandas 3; os .OUT são lidos abaixo
            s = _ler_summary((Path(pasta) / "Summary.OUT").read_text(errors="ignore"))
            pg = _ler_plantgro((Path(pasta) / "PlantGro.OUT").read_text(errors="ignore"))
            s["pdat"], s["adat"], s["mdat"] = _jd(s["PDAT"]), _jd(s["ADAT"]), _jd(s["MDAT"])
            for _, r in s.iterrows():
                if pd.isna(r["pdat"]):
                    continue
                lin = {"cod_ibge": cod, "ano": r["pdat"].year, "semeadura": f"{dia:02d}/{mes:02d}",
                       "ds_rend": r.get("HWAM", np.nan), "ds_biomassa": r.get("CWAM", np.nan),
                       "ds_dias_antese": (r["adat"] - r["pdat"]).days if pd.notna(r["adat"]) else np.nan,
                       "ds_ciclo_dias": (r["mdat"] - r["pdat"]).days if pd.notna(r["mdat"]) else np.nan}
                fim = r["mdat"] if pd.notna(r["mdat"]) else r["pdat"] + pd.Timedelta(days=200)
                antese = r["adat"] if pd.notna(r["adat"]) else r["pdat"] + pd.Timedelta(days=65)
                periodos = {"veg": (r["pdat"], antese - pd.Timedelta(days=10)),
                            "flor": (antese - pd.Timedelta(days=10), antese + pd.Timedelta(days=10)),
                            "ench": (antese + pd.Timedelta(days=10), fim)}
                for nome, (a, b) in periodos.items():
                    seg = pg.loc[a:b]
                    lin[f"ds_estresse_{nome}"] = float(seg["WSPD"].sum()) if "WSPD" in seg and len(seg) else np.nan
                    lin[f"ds_tmin_{nome}"] = float(tmin.loc[a:b].min()) if len(tmin.loc[a:b]) else np.nan
                linhas.append(lin)
    return pd.DataFrame(linhas)
