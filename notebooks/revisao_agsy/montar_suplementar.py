"""Builds ../artigo_elsevier/supplementary.tex from the previous supplementary (Methods S1-S2, Tables S1-S7, Figs S1-S5)
plus the revision tables (S8-S13) and Fig. S6. Run with cwd = notebooks."""
import re
from pathlib import Path

import pandas as pd

RAIZ = Path.cwd().parent
REV = RAIZ / "resultados" / "revisao"
SRC = RAIZ.parent / "artigo_springer" / "supplementary.tex"
DST = RAIZ.parent / "artigo_elsevier" / "supplementary.tex"
ORDEM = [f"{d:02d}/{m:02d}" for m in (1, 2, 3) for d in (5, 15, 25)]
s = SRC.read_text(encoding="utf-8")

s = s.replace("Sowing date as a risk-management decision: an agro-economic framework for second-season maize in southern Brazil\\\\[2pt]\n\\textit{Agronomy for Sustainable Development}",
              "Sowing date as a risk-management decision in soybean--maize double cropping: a physics-guided, insurance-validated agro-economic framework\\\\[2pt]\n\\textit{Agricultural Systems}")


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:,.{d}f}"


# ---------------- S4: regret table refreshed from the current run of notebook 05
rg = pd.read_csv(RAIZ / "resultados" / "arrependimento_decisao.csv").set_index("model")
NOMES = {"always sow": "Always sow", "A_media_historica": "A Historical mean", "B_MZA_calibrado": "B Calibrated MZA",
         "C_ML_puro": "C Pure ML", "D_hibrido": "D MZA hybrid", "E_hibrido_monotonico": "E MZA hybrid, monotonic",
         "F_hibrido_apsim": "F APSIM hybrid", "H_hibrido_aquacrop": "H AquaCrop hybrid", "I_hibrido_dssat": "I DSSAT hybrid",
         "J_hibrido_todos": "J All models", "K_hibrido_svr": "K MZA hybrid, SVR", "L_ensemble_E_SVR": "L Ensemble E + K",
         "perfect information": "Perfect information"}


def ic(txt):
    if pd.isna(txt) or txt.startswith("+0 (+0"):
        return "--"
    return txt.replace("-", "$-$").replace("+", "$+$")


lin = [f"{NOMES[k]} & {r['regret_R$_ha']:.0f} & {r['sows_%']:.0f} & {r['correct_decision_%']:.0f} & "
       f"{r['realized_margin_R$_ha']:.0f} & {ic(r['vs_E_IC95'])} \\\\".replace("& -", "& $-$").replace("$-$-", "--")
       for k, r in rg.iterrows()]
a = s.index("Always sow & ")
b = s.index("\\bottomrule", a)
s = s[:a] + "\n".join(lin) + "\n" + s[b:]

# ---------------- S8: PSR indemnities by sowing period (replaces the old list of 15 March municipalities)
t = pd.read_csv(REV / "psr_indenizacao_epoca.csv", header=[0, 1], index_col=0)
grupos = ["27 mun. (earlier 15/03)", "rest of PR"]
lin = []
for per in t.index:
    row = [per.replace("–", "--").replace("≤", "$\\le$").replace("≥", "$\\ge$")]
    for g in grupos:
        row += [f(t.loc[per, ("policies", g)], 0), f(t.loc[per, ("indemnified", g)]), f(t.loc[per, ("drought", g)]), f(t.loc[per, ("frost", g)])]
    lin.append(" & ".join(row) + r" \\")
s8 = r"""\clearpage
{\footnotesize\setlength{\tabcolsep}{4pt}
\begin{longtable}{lrrrrrrrr}
\caption{PSR second-season maize policies in Paraná (2016--2024) by sowing period (start of coverage): number of
policies and share indemnified (all events, drought and frost), for the 27 municipalities that a hybrid model without the
potential-yield term sent to 15 March sowing and for the rest of the state.}\label{tab:psr_epoca}\\
\toprule
 & \multicolumn{4}{c}{27 municipalities} & \multicolumn{4}{c}{Rest of Paraná} \\
Sowing period & Policies & Indemnified & Drought & Frost & Policies & Indemnified & Drought & Frost \\
\midrule
""" + "\n".join(lin) + r"""
\bottomrule
\end{longtable}}
"""
a = s.index("\\clearpage\n{\\footnotesize\\setlength{\\tabcolsep}{4pt}\n\\begin{longtable}{llrrrrrr}")
b = s.index("\\end{longtable}}", a) + len("\\end{longtable}}")
s = s[:a] + s8 + s[b:]

# ---------------- S9: alternative configurations of the hybrid model
v = pd.read_csv(REV / "variantes_validacao.csv", index_col=0)
rz = pd.read_csv(REV / "raiz_extremos.csv")
nomes_v = {"V0": "Mean of 4 window dates (main)", "V1": "Mean of 9 dates weighted by the observed calendar",
           "V2": "One date drawn from the observed calendar (3 draws)", "V3": "One window date drawn at random (3 draws)"}
lin = [f"{nomes_v[k]} & 0.40 m & -- & {f(r.R2_mun)} & {f(r.R2_state_annual)} & {f(r.RMSE, 0)} \\\\" for k, r in v.iterrows()]
lin += [f"Mean of 4 window dates & {r.roots[1:]} m & {'yes' if 'extremes' in r.features else 'no'} & {f(r.R2_mun)} & {f(r.R2_annual)} & {f(r.RMSE, 0)} \\\\"
        for _, r in rz.iterrows() if not (r.roots == "z0.40" and r.features == "base vars")]
s9 = r"""
\begin{table}[!ht]
\centering\footnotesize
\caption{Leave-one-year-out accuracy of alternative configurations of the final hybrid model: training features
(fixed window mean or the sowing calendar observed in the PSR records), maximum effective rooting depth of the water
balance and stage-specific extreme indices (longest dry spell, cold and heat degree-days by period).}\label{tab:variants}
\begin{tabular}{lcccrr}
\toprule
Training features & Rooting depth & Extreme indices & $R^2$ & $R^2$ state & $RMSE$ (kg ha$^{-1}$) \\
\midrule
""" + "\n".join(lin) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# ---------------- S10: SHAP by mesoregion
sh = pd.read_csv(REV / "shap_mesorregiao.csv", index_col=0)
cols = ["water, vegetative", "water, flowering", "water, grain filling", "water (PA/PPf)", "cold/frost, reproductive", "heat, flowering", "static"]
lin = [f"{m.replace(' Paranaense', '')} & " + " & ".join(f(sh.loc[m, c], 1) for c in cols) + r" \\" for m in sh.index]
s10 = r"""
\begin{table}[!ht]
\centering\footnotesize
\caption{Share (\%) of the mean absolute SHAP value of the monotonic gradient-boosting component of the final model, by
group of indicators and mesoregion.}\label{tab:shapmeso}
\begin{tabular}{lrrrrrrr}
\toprule
Mesoregion & \multicolumn{4}{c}{Water} & Cold/frost, & Heat, & Static \\
 & veg. & flow. & filling & $PA/PP_f$ & reprod. & flow. & \\
\midrule
""" + "\n".join(lin) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# ---------------- S11: date response calibrated on the PSR (not adopted)
c = pd.read_csv(REV / "calibracao_data_psr.csv", index_col=0)
lin = []
for k, r in c.iterrows():
    nome = k.replace("–", "--").replace("LINEAR ", "linear, ")
    lin.append(f"{nome} & {f(r.rmse_profile_before)} & {f(r.rmse_profile_after)} & {f(r.r_before, 2)} & {f(r.r_after, 2)} & {f(r.brier_before)} & {f(r.brier_after)} \\\\")
s11 = r"""
\begin{table}[!ht]
\centering\footnotesize
\caption{Calibration of date-specific (eight parameters) or linear (one parameter) shifts of the predicted log yield on the
within-season PSR loss profile, fitted on some seasons and evaluated on the others. Profile RMSE and correlation between
the PSR and model date coefficients (municipality and season fixed effects); Brier score of the cell-level loss
probability. Neither calibration was adopted.}\label{tab:calib}
\begin{tabular}{lrrrrrr}
\toprule
Fit / test seasons & \multicolumn{2}{c}{Profile RMSE} & \multicolumn{2}{c}{Profile $r$} & \multicolumn{2}{c}{Brier} \\
 & before & after & before & after & before & after \\
\midrule
""" + "\n".join(lin) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# ---------------- S12: covariate shift
cs = pd.read_csv(REV / "covariate_shift.csv", index_col=0)
lin = [f"{i} & {f(r.iloc[0])} & {f(r.iloc[1])} \\\\" for i, r in cs.iterrows()]
s12 = r"""
\begin{table}[!ht]
\centering\footnotesize
\caption{Covariate shift of the per-date predictions: share of municipality-seasons whose indicators lie beyond the 99th
percentile of the Mahalanobis distance of the training window means, and share with at least one indicator outside the
0.5--99.5 percentile range of training.}\label{tab:shift}
\begin{tabular}{lrr}
\toprule
Sowing date & Beyond training P99 ($D^2$) & Indicator outside training range \\
\midrule
""" + "\n".join(lin) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

# ---------------- S13: lower tail and stability by date and mesoregion
ct = pd.read_csv(REV / "cauda_estabilidade.csv")
by = ct.groupby("semeadura")[["mean", "p10", "cv"]].mean().reindex(ORDEM)
cvr = ct.pivot_table(index="meso", columns="semeadura", values="cv").reindex(columns=ORDEM)
lin = [f"State mean yield (kg ha$^{{-1}}$) & " + " & ".join(f(x, 0) for x in by["mean"]) + r" \\",
       f"State P10 yield (kg ha$^{{-1}}$) & " + " & ".join(f(x, 0) for x in by["p10"]) + r" \\", r"\midrule"]
lin += [f"CV, {m.replace(' Paranaense', '')} & " + " & ".join(f(x, 2) for x in cvr.loc[m]) + r" \\" for m in cvr.index]
s13 = r"""
\begin{table}[!ht]
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Lower tail and stability of the predicted municipal yield over the 26 weather seasons (2025 technology level), by
sowing date: state means of the municipal mean and 10th percentile, and coefficient of variation (CV) by mesoregion.}\label{tab:tail}
\begin{tabular}{l""" + "r" * 9 + r"""}
\toprule
 & """ + " & ".join(ORDEM) + r""" \\
\midrule
""" + "\n".join(lin) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

s = s.replace("\\clearpage\\section*{Supplementary figures}", s9 + s10 + s11 + s12 + s13 + "\n\\clearpage\\section*{Supplementary figures}")
s6 = r"""
\begin{figure}[!ht]
\centering
\includegraphics[width=\textwidth]{figuras/fig_correlacao_fase.png}
\caption{Pearson correlation between the yield anomaly and the MZA-FAO indicators of each phenological period, between
seasons (state means) and between municipalities (mean of the within-season correlations), 2003--2025.}
\end{figure}
"""
s = s.replace("\\bibliographystyle{plainnat}", s6 + "\n\\bibliographystyle{plainnat}")
DST.write_text(s, encoding="utf-8")
print("written", DST, len(s))
