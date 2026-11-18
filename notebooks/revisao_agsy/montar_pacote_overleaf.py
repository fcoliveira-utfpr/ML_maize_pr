"""Builds the Overleaf/Elsevier submission package (zip) from ../../artigo_elsevier.
Run from anywhere: python montar_pacote_overleaf.py"""
import re
import zipfile
from pathlib import Path

ART = Path(__file__).resolve().parents[3] / "artigo_elsevier"
OUTZIP = ART.parent / "agsy_submission_overleaf.zip"
ROOT = "agsy_submission"

main = (ART / "main.tex").read_text(encoding="utf-8")
sup = (ART / "supplementary.tex").read_text(encoding="utf-8")
inc = lambda s: re.findall(r"includegraphics\[[^\]]*\]\{([^}]+)\}", s)
figs_main = list(dict.fromkeys(inc(main)))          # order of appearance; first one is the graphical abstract
figs_all = sorted(set(inc(main)) | set(inc(sup)))
# main-text figure order (graphical abstract is also Fig. 1)
body = main[main.index("\\section{Introduction}"):]
order = list(dict.fromkeys(inc(body)))

leia = f"""PACOTE DE SUBMISSAO -- Agricultural Systems (Elsevier)
==========================================================

Overleaf
--------
1. No Overleaf: New Project > Upload Project > escolha este .zip.
2. Menu > Settings: Compiler = pdfLaTeX; Main document = main.tex.
3. Para compilar o material suplementar ou a carta, troque o "Main document" para
   supplementary.tex ou cover_letter.tex (os tres usam a mesma pasta figuras/ e o mesmo references.bib).
   Arquivos de classe da Elsevier (cas-sc.cls, cas-common.sty, cas-model2-names.bst) e a pasta
   thumbnails/ sao necessarios para o main.tex.

Arquivos
--------
main.tex                 manuscrito (cas-sc, resumo estruturado, highlights, graphical abstract)
supplementary.tex        material suplementar (Methods S1-S2, Tables S1-S13, Figs. S1-S6)
cover_letter.tex         carta de apresentacao ao editor
references.bib           bibliografia
figuras/                 figuras usadas pelos .tex (PNG, 300 dpi)
elsevier_upload/         as mesmas figuras do texto principal renomeadas na ordem do artigo, para o
                         upload como arquivos separados no Editorial Manager

Ordem das figuras do texto principal (elsevier_upload/)
-------------------------------------------------------
graphical_abstract.png  <- {figs_main[0]}
""" + "\n".join(f"Figure{i:02d}.png          <- {f}" for i, f in enumerate(order, 1)) + """

Pendencias do autor antes de submeter
-------------------------------------
- coautores, ORCID, financiamento e agradecimentos (nao incluidos de proposito);
- conferir no Guide for Authors os limites do resumo (~380 palavras) e dos highlights (<= 85 caracteres);
- a carta afirma que o manuscrito e original e nao esta em avaliacao em outra revista: confirme.
"""

with zipfile.ZipFile(OUTZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["main.tex", "supplementary.tex", "cover_letter.tex", "references.bib",
              "cas-sc.cls", "cas-common.sty", "cas-model2-names.bst"]:
        z.write(ART / f, f"{ROOT}/{f}")
    for f in sorted((ART / "thumbnails").iterdir()):
        z.write(f, f"{ROOT}/thumbnails/{f.name}")
    for f in figs_all:
        z.write(ART / f, f"{ROOT}/{f}")
    z.write(ART / figs_main[0], f"{ROOT}/elsevier_upload/graphical_abstract.png")
    for i, f in enumerate(order, 1):
        z.write(ART / f, f"{ROOT}/elsevier_upload/Figure{i:02d}.png")
    z.writestr(f"{ROOT}/LEIA-ME.txt", leia)
print(OUTZIP, f"{OUTZIP.stat().st_size / 1e6:.1f} MB")
print(leia)
