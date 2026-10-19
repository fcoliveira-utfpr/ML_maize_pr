"""AquaCrop (FAO; implementação AquaCrop-OSPy, pacote `aquacrop`) sem calibração, para o modelo híbrido.

Uma simulação por (município, data de semeadura), 1999–2025, semeando todo ano na mesma data, com a
cultura padrão `MaizeGDD` (fenologia por graus-dia), clima BR-DWGD (ETo Penman-Monteith do próprio
BR-DWGD) e perfil de solo com LL15/DUL/SAT do mesmo procedimento usado no APSIM.
"""
import numpy as np
import pandas as pd
from aquacrop import AquaCropModel, Crop, InitialWaterContent, Soil

import modelo_apsim as A

CULTURA = "MaizeGDD"
UMIDADE_INICIAL = 70  # % da água disponível no início da simulação


def clima(clima_mun):
    c = clima_mun.reset_index().sort_values("data")
    return pd.DataFrame({"MinTemp": c["Tmin"].to_numpy(), "MaxTemp": c["Tmax"].to_numpy(),
                         "Precipitation": c["pr"].clip(lower=0).to_numpy(),
                         "ReferenceET": c["ET"].clip(lower=0.1).to_numpy(), "Date": pd.to_datetime(c["data"]).to_numpy()})


def solo(perfil):
    s = Soil("custom", dz=[0.1] * 18, cn=72, rew=9)
    for esp, wp, fc, sat, ks in zip(A.ESPESSURA, perfil["LL15"], perfil["DUL"], perfil["SAT"], perfil["KS"]):
        s.add_layer(thickness=esp / 1000, thWP=float(wp), thFC=float(fc), thS=float(sat),
                    Ksat=float(ks), penetrability=100)
    return s


def simular_municipio(cod, clima_mun, perfil, datas, tmin_diaria):
    """Linhas por (ano, data) com rendimento simulado e estresse de transpiração por período."""
    w, s = clima(clima_mun), solo(perfil)
    linhas = []
    for mes, dia in datas:
        cultura = Crop(CULTURA, planting_date=f"{mes:02d}/{dia:02d}")
        m = AquaCropModel("1999/01/01", "2025/12/31", w, s, cultura,
                          InitialWaterContent(wc_type="Pct", value=[UMIDADE_INICIAL]))
        m.run_model(till_termination=True)
        fim = m.get_simulation_results()
        g, f = m.get_crop_growth(), m.get_water_flux()
        d = g[["season_counter", "dap", "gdd_cum", "biomass", "canopy_cover"]].join(f[["Tr", "TrPot"]])
        d = d[d["dap"] > 0]
        flor_ini, flor_fim = cultura.HIstart, cultura.HIstart + cultura.Flowering
        for k, linha in fim.iterrows():
            e = d[d["season_counter"] == linha["Season"]]
            ano = pd.Timestamp(linha["Harvest Date (YYYY/MM/DD)"]).year
            sem = pd.Timestamp(ano, mes, dia)
            if e.empty or sem.year < 1999:
                continue
            datas_e = sem + pd.to_timedelta(e["dap"] - 1, unit="D")
            razao = (e["Tr"] / e["TrPot"].where(e["TrPot"] > 0)).fillna(1).clip(0, 1)
            lin = {"cod_ibge": cod, "ano": ano, "semeadura": f"{dia:02d}/{mes:02d}",
                   "aq_rend": linha["Dry yield (tonne/ha)"] * 1000,
                   "aq_rend_pot": linha["Yield potential (tonne/ha)"] * 1000,
                   "aq_biomassa": e["biomass"].max(), "aq_cc_max": e["canopy_cover"].max(),
                   "aq_ciclo_dias": int(e["dap"].max())}
            lin["aq_razao_pot"] = lin["aq_rend"] / lin["aq_rend_pot"] if lin["aq_rend_pot"] > 0 else np.nan
            for nome, sel in {"veg": e["gdd_cum"] < flor_ini,
                              "flor": (e["gdd_cum"] >= flor_ini) & (e["gdd_cum"] < flor_fim),
                              "ench": e["gdd_cum"] >= flor_fim}.items():
                lin[f"aq_estresse_{nome}"] = float((1 - razao[sel]).sum())
                dias = datas_e[sel]
                lin[f"aq_tmin_{nome}"] = float(tmin_diaria.reindex(dias).min()) if len(dias) else np.nan
            linhas.append(lin)
    return pd.DataFrame(linhas)
