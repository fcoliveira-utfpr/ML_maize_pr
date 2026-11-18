# Revision analyses for Agricultural Systems (Nov 2026)

Prototype scripts (to become notebook 10). Run with cwd = `notebooks/`. Outputs: `resultados/revisao/`.

| Script | What | Extra input |
|---|---|---|
| rev1.py | covariate shift (Mahalanobis), forward pricing (hedge ratios), soybean break-even | — |
| rev2.py | PSR policy-level: sowing calendar proxy, indemnity by sowing period (15/03 check), tau by region, adverse selection; writes `dados/psr_milho2safra_pr_municipio_ano.csv` | psr_pr.parquet |
| rev3.py | hybrid retrained with observed calendar (V0–V3) + PSR within-season gradient check | psr_pr.parquet |
| rev4.py | efficient frontier, decision gap vs current calendar, stress-stratified validation, lower tail/CV. Env: PROD_FILE, TAU_REGIONAL=1 | — |
| rev5.py | physics offset beta·[log PPf(date) − window mean] calibrated/checked against PSR and IBGE | psr_pr.parquet |

`psr_pr.parquet` = PSR open microdata (MAPA, dataset sisser3, CSV 2016–2024 + 2025) filtered to state PR. It holds
personal data, so it stays OUT of the repository: re-download with a browser user-agent from
https://dados.agricultura.gov.br/dataset/sisser3 and filter `SG_UF_PROPRIEDADE == "PR"` (see rev2.py for columns).
