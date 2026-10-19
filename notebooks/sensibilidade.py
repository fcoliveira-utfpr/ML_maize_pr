"""Sensitivity scenarios (notebook 06).

The environment variable SENS selects the scenario when running 02 → 03 → 05; empty = main run.
Each scenario writes to resultados/sensibilidade/<SENS>/ (figures included) and reads from the main run
whatever it does not redo.
"""
import os
from pathlib import Path

RESULTADOS = Path(__file__).resolve().parent.parent / "resultados"
JANELA_ZARC = ["15/01", "25/01", "05/02", "15/02", "25/02", "05/03", "15/03"]

CENARIOS = {
    # thermal time from sowing to maturity (main: 1450 °C day)
    "CT1350": {"ct": 1350.0},
    "CT1550": {"ct": 1550.0},
    # weights of the dates in the window mean that represents the municipal yield (main: 4 equal dates)
    "janela_cedo": {"pesos_janela": {"15/01": 0.4, "05/02": 0.3, "25/02": 0.2, "15/03": 0.1}},
    "janela_7datas": {"pesos_janela": {d: 1.0 for d in JANELA_ZARC}},
}

SENS = os.environ.get("SENS", "")
PARAM = CENARIOS[SENS] if SENS else {}
SAIDA = RESULTADOS / "sensibilidade" / SENS if SENS else RESULTADOS
SAIDA.mkdir(parents=True, exist_ok=True)


def entrada(nome):
    """File redone by the scenario, if it exists; otherwise the one from the main run."""
    return SAIDA / nome if (SAIDA / nome).exists() else RESULTADOS / nome
