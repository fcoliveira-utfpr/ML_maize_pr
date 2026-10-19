"""Cenários de sensibilidade (notebook 06).

A variável de ambiente SENS escolhe o cenário ao executar 02 → 03 → 05; vazia = rodada principal.
Cada cenário grava em resultados/sensibilidade/<SENS>/ (inclusive figuras) e lê da rodada principal
o que não refaz.
"""
import os
from pathlib import Path

RESULTADOS = Path(__file__).resolve().parent.parent / "resultados"
JANELA_ZARC = ["15/01", "25/01", "05/02", "15/02", "25/02", "05/03", "15/03"]

CENARIOS = {
    # soma térmica da semeadura à maturação (principal: 1450 °C dia)
    "CT1350": {"ct": 1350.0},
    "CT1550": {"ct": 1550.0},
    # pesos das datas na média da janela que representa o rendimento municipal (principal: 4 datas iguais)
    "janela_cedo": {"pesos_janela": {"15/01": 0.4, "05/02": 0.3, "25/02": 0.2, "15/03": 0.1}},
    "janela_7datas": {"pesos_janela": {d: 1.0 for d in JANELA_ZARC}},
}

SENS = os.environ.get("SENS", "")
PARAM = CENARIOS[SENS] if SENS else {}
SAIDA = RESULTADOS / "sensibilidade" / SENS if SENS else RESULTADOS
SAIDA.mkdir(parents=True, exist_ok=True)


def entrada(nome):
    """Arquivo refeito pelo cenário, se existir; senão o da rodada principal."""
    return SAIDA / nome if (SAIDA / nome).exists() else RESULTADOS / nome
