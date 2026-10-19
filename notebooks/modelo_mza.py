"""Simulation of second-season maize with the FAO Agro-Ecological Zone model (MZA-FAO).

One simulation = (municipality, year, sowing date), at daily scale, with BR-DWGD weather.
The crop water balance, potential yield (PPf) and attainable yield (PA) come from the
agrometeorologiapy library (v0.3.0); here are only the growing-degree-day phenology, the Kc and
rooting-depth curves and the agroclimatic indicators per phase.
"""
import numpy as np
import pandas as pd
from agrometeorologiapy import (
    angulo_horario_nascer,
    balanco_hidrico_cultura,
    declinacao_solar,
    fator_correcao_distancia,
    irradiancia_extraterrestre,
)
from agrometeorologiapy.produtividade import produtividade_atingivel, produtividade_potencial

# ---------------------------------------------------------------- crop parameters
TB = 10.0                 # base temperature (°C)
CT = 1450.0               # thermal time from sowing to physiological maturity (°C day)

# FAO-56 stages (Kc), as fractions of thermal time: initial, development, mid-season, late
FASES_KC = (0.17, 0.28, 0.33, 0.22)
KC_INI, KC_MED, KC_FIN = 0.3, 1.2, 0.6

# Doorenbos & Kassam (1979) periods for maize ky, as fractions of thermal time
FASES_KY = {"estabelecimento": 0.15, "vegetativo": 0.30, "floracao": 0.12,
            "enchimento": 0.30, "maturacao": 0.13}
KY = {"estabelecimento": 0.4, "vegetativo": 0.4, "floracao": 1.5, "enchimento": 0.5, "maturacao": 0.2}

Z_INI, Z_MAX = 0.10, 0.40  # effective rooting depth (m): grows until the end of the vegetative phase
IAF, CC, UMIDADE = 5.0, 0.40, 13.0
SPINUP = 30                # days of water balance before sowing (soil under straw, initial Kc)
JANELA_MAX = 300           # days after sowing; without reaching CT, the cycle is flagged incomplete

# ZARC phases (Portaria SPA/MAPA nº 329/2026, second-season maize PR, Group I), in days after sowing
FASES_ZARC = {"I": (1, 20), "II": (21, 50), "III": (51, 80), "IV": (81, 100)}


def graus_dia(tmax, tmin, tb=TB):
    """Same rule as `agrometeorologiapy.data_maturacao_fisiologica`."""
    tmed = (tmax + tmin) / 2
    amp = np.maximum(tmax - tmin, 0.1)
    return np.where(tb < tmin, tmed - tb, np.where(tmax > tb, (tmax - tb) ** 2 / (2 * amp), 0.0))


def qo_diaria(lat, datas):
    nda = datas.dayofyear.to_numpy()
    dec = declinacao_solar(nda)
    return irradiancia_extraterrestre(lat, dec, angulo_horario_nascer(lat, dec), fator_correcao_distancia(nda))


def _fase(frac_acum, limites):
    """Phase index of each day from the accumulated fraction of thermal time."""
    return np.minimum(np.searchsorted(np.cumsum(limites), frac_acum, side="right"), len(limites) - 1)


def _kc(frac):
    a, b, c, _ = np.cumsum(FASES_KC)
    return np.select(
        [frac < a, frac < b, frac < c],
        [KC_INI, KC_INI + (KC_MED - KC_INI) * (frac - a) / (b - a), KC_MED],
        KC_MED + (KC_FIN - KC_MED) * np.clip((frac - c) / (1 - c), 0, 1),
    )


def simular(clima_mun, lat, dta, semeadura, ct=CT):
    """Simulates one cycle. `clima_mun`: daily weather of one municipality indexed by date."""
    semeadura = pd.Timestamp(semeadura)
    janela = clima_mun.loc[semeadura - pd.Timedelta(days=SPINUP): semeadura + pd.Timedelta(days=JANELA_MAX)]
    pos = janela.index >= semeadura
    campo = janela.loc[pos]

    gd = graus_dia(campo["Tmax"].to_numpy(), campo["Tmin"].to_numpy())
    acum = np.cumsum(gd)
    # In cold regions thermal time is not reached before winter: the crop does not mature (PA = 0).
    completo = bool(acum[-1] >= ct)
    ciclo = int(np.searchsorted(acum, ct) + 1) if completo else len(campo)
    campo = campo.iloc[:ciclo].copy()
    frac = np.minimum(acum[:ciclo] / ct, 1.0)

    nomes_ky = list(FASES_KY)
    campo["fase_ky"] = np.array(nomes_ky)[_fase(frac, list(FASES_KY.values()))]
    frac_fim_veg = FASES_KY["estabelecimento"] + FASES_KY["vegetativo"]
    z = Z_INI + (Z_MAX - Z_INI) * np.clip(frac / frac_fim_veg, 0, 1)

    # ---- crop water balance (spin-up with initial Kc and initial roots)
    pre = janela.loc[~pos]
    bh_in = pd.DataFrame({
        "Chuva": np.r_[pre["pr"].to_numpy(), campo["pr"].to_numpy()],
        "ETc": np.r_[KC_INI * pre["ET"].to_numpy(), _kc(frac) * campo["ET"].to_numpy()],
        "CAD": np.r_[np.full(len(pre), Z_INI * dta), z * dta],
    })
    bh = balanco_hidrico_cultura(bh_in).iloc[len(pre):].reset_index(drop=True)
    for col in ["ETc", "ETR", "DEF", "EXC", "ARM"]:
        campo[col] = bh[col].to_numpy()

    # ---- potential and attainable yield
    pp_in = pd.DataFrame({"data": campo.index, "Tmed": ((campo["Tmax"] + campo["Tmin"]) / 2).to_numpy(),
                          "Qg": campo["Rs"].to_numpy(), "Qo": qo_diaria(lat, campo.index)})
    ppf = produtividade_potencial(pp_in, lat=lat, rota="C4", IAF=IAF, Cc=CC, U=UMIDADE, intervalo="d").attrs["PPf"]
    pa = produtividade_atingivel(ppf, campo["ETR"].to_numpy(), campo["ETc"].to_numpy(),
                                 ky=KY, fase=campo["fase_ky"].to_numpy())

    # ---- indicators per D&K period
    floracao = campo.index[campo["fase_ky"] == "floracao"]
    saida = {"ciclo_dias": ciclo, "ciclo_incompleto": not completo,
             "data_floracao": floracao[0] if len(floracao) else pd.NaT,
             "data_maturacao": campo.index[-1], "PPf": ppf, "PA": pa.attrs["PA"] if completo else 0.0}
    tmed = (campo["Tmax"] + campo["Tmin"]) / 2
    for nome, g in campo.groupby("fase_ky", sort=False):
        etc = g["ETc"].sum()
        saida.update({
            f"ISNA_{nome}": g["ETR"].sum() / etc if etc > 0 else 1.0,
            f"chuva_{nome}": g["pr"].sum(),
            f"DEF_{nome}": g["DEF"].sum(),
            f"EXC_{nome}": g["EXC"].sum(),
            f"Tmed_{nome}": tmed.loc[g.index].mean(),
            f"Tmax_{nome}": g["Tmax"].mean(),
            f"Tmin_{nome}": g["Tmin"].min(),
            f"Rs_{nome}": g["Rs"].mean(),
            f"UR_{nome}": g["RH"].mean(),
            f"geada_{nome}": int((g["Tmin"] <= 2.0).sum()),
            f"calor_{nome}": int((g["Tmax"] >= 35.0).sum()),
        })

    saida.update(criterios_zarc(janela, pos, dta))
    return saida


def criterios_zarc(janela, pos, dta):
    """ZARC criteria on a 100-day reference cycle (phases on fixed days after sowing)."""
    pre, ref = janela.loc[~pos], janela.loc[pos].iloc[:100]
    frac = np.arange(1, 101) / 100
    bh = balanco_hidrico_cultura(pd.DataFrame({
        "Chuva": np.r_[pre["pr"].to_numpy(), ref["pr"].to_numpy()],
        "ETc": np.r_[KC_INI * pre["ET"].to_numpy(), _kc(frac) * ref["ET"].to_numpy()],
        "CAD": np.r_[np.full(len(pre), Z_INI * dta), (Z_INI + (Z_MAX - Z_INI) * np.clip(frac / 0.5, 0, 1)) * dta],
    })).iloc[len(pre):].reset_index(drop=True)
    out = {}
    for fase, (d0, d1) in FASES_ZARC.items():
        g = bh.iloc[d0 - 1: d1]
        out[f"zarc_ISNA_{fase}"] = g["ETR"].sum() / g["ETc"].sum()
    tmed = (ref["Tmax"] + ref["Tmin"]) / 2
    out["zarc_tmed_floracao"] = tmed.iloc[50:80].mean()
    out["zarc_geada_I_III"] = int((ref["Tmin"].iloc[:80] < 2.0).sum())
    return out


def simular_municipio(cod, lat, dta, arquivo_clima, anos, datas_semeadura, ct=CT):
    """All year × sowing date combinations of one municipality (to run in parallel)."""
    clima = pd.read_parquet(arquivo_clima, filters=[("cod_ibge", "==", cod)]).set_index("data").sort_index()
    linhas = []
    for ano in anos:
        for mes, dia in datas_semeadura:
            r = simular(clima, lat, dta, pd.Timestamp(ano, mes, dia), ct=ct)
            r.update({"cod_ibge": cod, "ano": ano, "semeadura": f"{dia:02d}/{mes:02d}"})
            linhas.append(r)
    return pd.DataFrame(linhas)
