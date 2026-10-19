"""Choropleth maps of the municipalities of Paraná without geopandas (IBGE GeoJSON mesh + matplotlib)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

MALHA = Path(__file__).resolve().parent.parent / "dados" / "malha_municipios_pr.geojson"


def _aneis(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [p[0] for p in geom["coordinates"]]


def carregar_malha():
    feats = json.load(open(MALHA))["features"]
    return {int(f["properties"]["codarea"]): _aneis(f["geometry"]) for f in feats}


def mapa(ax, valores, cmap="viridis", vmin=None, vmax=None, titulo=None, categorias=None, cores=None):
    """Colours the municipalities. `valores`: Series indexed by cod_ibge.

    For categorical variables, pass `categorias` (order) and `cores` (same order).
    """
    malha = carregar_malha()
    patches, cor = [], []
    if categorias is not None:
        mapa_cor = dict(zip(categorias, cores))
    else:
        vmin = np.nanmin(valores) if vmin is None else vmin
        vmax = np.nanmax(valores) if vmax is None else vmax
        norma = plt.Normalize(vmin, vmax)
        cm = plt.get_cmap(cmap)
    for cod, aneis in malha.items():
        v = valores.get(cod, np.nan)
        if categorias is not None:
            c = mapa_cor.get(v, "#dddddd")
        else:
            c = "#dddddd" if np.isnan(v) else cm(norma(v))
        for anel in aneis:
            patches.append(Polygon(np.asarray(anel), closed=True))
            cor.append(c)
    ax.add_collection(PatchCollection(patches, facecolor=cor, edgecolor="white", linewidth=0.15))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=15)
    if categorias is None:
        return plt.cm.ScalarMappable(norm=norma, cmap=cm)
    return None
