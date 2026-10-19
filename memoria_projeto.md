# Projeto Milho 2ª Safra PR — Modelo híbrido MZA-FAO + ML e risco econômico da data de semeadura

## Objetivo

Artigo (alvo: *Journal of Agronomy and Crop Science*, Wiley; alternativa: *Stochastic Environmental
Research and Risk Assessment*, Springer — ambas nas listas CAPES de acesso aberto) que:

1. estima a produtividade do milho 2ª safra nos **399 municípios do Paraná** com um **modelo
   híbrido**: o Modelo da Zona Agroecológica da FAO (MZA-FAO) gera as variáveis físicas e o
   aprendizado de máquina corrige o que a física não representa (ideia central do autor);
2. avalia o **risco econômico de cada data de semeadura** (margem, prejuízo, risco de cauda, seguro
   e ZARC) e recomenda a melhor data por município.

Substitui e aprofunda dois TCCs orientados pelo autor (UTFPR Santa Helena, 2026), em `Artigo_milho/`:
- **Vitor E. C. Lopes** — modelagem híbrida MZA-FAO + RF/ET/XGBoost/LightGBM, 10 municípios do Oeste;
- **Artur J. J. Esbabo** — ZARC + produtividade prevista + risco econômico, PCA + TOPSIS, 50 municípios do Oeste.

## Estrutura do projeto

| Pasta / arquivo | Papel |
|---|---|
| `notebooks/01_clima_xavier.ipynb` | Clima diário 1998-12-01 a 2025-12-31 (BR-DWGD v3.2.4) para as sedes dos 399 municípios, via Earth Engine |
| `notebooks/02_simulacao_mza.ipynb` | MZA-FAO + critérios ZARC: 399 municípios × 27 anos (1999–2025) × 9 datas de semeadura ≈ 97 mil simulações |
| `notebooks/modelo_mza.py` | Simulação de um ciclo (fenologia por graus-dia, Kc, ky, raiz, ZARC), usando `agrometeorologiapy` |
| `notebooks/03_modelo_hibrido.ipynb` | (a fazer) ML híbrido treinado no IBGE/PAM, validação por ano, quantis, análises por fase |
| `notebooks/04_economia.ipynb` | Custos e preços reais (R$ de dez/2025) |
| `notebooks/05_risco_decisao.ipynb` | (a fazer) Margem por ano, SERF, P(prejuízo), CVaR, seguro/ZARC, melhor data |
| `dados/` | Insumos brutos (ver tabela abaixo) |
| `resultados/` | Saídas dos notebooks (CSV/parquet) |
| `figuras/` | Figuras do artigo (300 dpi) |
| `Artigo_milho/` | TCCs de origem (docx) |
| `Wiley_LaTeX_Manuscript_Template_for_the_Journal_of_Agronomy_and_Crop_Science/` | Modelo LaTeX da revista, para a escrita |
| `bibliografia/` | Referências (fora do git) |

### Dados (`dados/`)

| Arquivo | Fonte |
|---|---|
| `ibge_pam_milho_pr.csv` | IBGE/PAM, SIDRA tabela 839 (milho 1ª e 2ª safra por município, 2003–2025; área plantada/colhida, produção, rendimento). 397 municípios; 339 com ≥15 anos de 2ª safra |
| `municipios_pr.csv` | Código IBGE, coordenadas da sede, altitude, DTA (mm/m) e marcador da mesorregião Oeste (50) |
| `nhistorico_94.xls` (aba `milho_saf`) | SEAB/DERAL, custo do milho safrinha por saca, safras 1999/00–2024/25, com mês de referência |
| `milho_2a_safra_serie_historica_2005-2025.xls` | CONAB, custo do milho 2ª safra (85 abas do PR, 6 municípios, 1999–2025) — usado só para a proporção Renda de Fatores / Custo Total |
| `custos.xlsx` | SEAB/DERAL, custo detalhado por item, 9 culturas, 5 datas (mai/2025–mai/2026) |
| `ipea_precos_milho_mensal.csv` | Preço recebido pelo produtor no PR (SEAB/DERAL via Ipeadata, série `DERAL12_PRMI12`) |
| `ipca_mensal.csv` | IPCA mensal, BCB/SGS 433 (índice base dez/1998 = 100) |
| `xavier_diario_pr.csv.gz` / `.parquet` | Gerado pelo notebook 01 (72 MB, fora do git) |

## Pipeline

1. **Clima** (01) → `dados/xavier_diario_pr.csv.gz` (pr, Tmax, Tmin, RH, Rs, u2, ET; 3,95 milhões de linhas, sem falhas).
2. **MZA-FAO** (02) → `resultados/simulacao_mza_pr.parquet`: por (município, ano, data): ciclo,
   datas de floração e maturação, PPf, PA, e por período de Doorenbos & Kassam ISNA, chuva, DEF,
   EXC, Tmed/Tmax/Tmin, Rs, UR, dias de geada (Tmin ≤ 2 °C) e de calor (Tmax ≥ 35 °C); critérios do ZARC.
3. **Modelo híbrido** (03): alvo = rendimento IBGE 2ª safra. O dado observado mistura as datas de
   semeadura, então o treino usa as variáveis do MZA médias da janela de semeadura; a predição por
   data usa as variáveis da data. Saída: distribuição (quantis + intervalos conformais).
4. **Economia** (04) → `resultados/custos_milho_safrinha_real.csv`, `resultados/precos_milho_pr_mensal_real.csv`.
5. **Risco e decisão** (05): margem por ano pareando clima, preço do mês de colheita e custo do mesmo
   ano + Monte Carlo da incerteza do modelo → SERF, P(prejuízo), CVaR 10%, valor do seguro/ZARC.

## Convenções que não podem ser quebradas

- **Tudo indexado pelo ano da colheita.** IBGE/PAM e CONAB já usam esse ano ("SAFRA DA SECA - 2010",
  "2ª SAFRA 2025"); a DERAL usa ano-agrícola ("1999/00" → colheita 2000).
- **Validação deixando um ano inteiro de fora** (e, quando couber, município + ano). Nunca divisão aleatória.
- Produtividade: **somente IBGE/PAM 2ª safra** (decisão do autor). Clima: **somente BR-DWGD (Xavier)**.
- **Economia estrutural, fora do ML**: receita − custo é identidade contábil; não há margem observada
  por município; custo e preço são estaduais. O ML entrega a distribuição da produtividade.
- MZA com a biblioteca do autor **`agrometeorologiapy` ≥ 0.3.0** (PyPI; repositório local
  `~/Python/agrobiblioteca`). Não reimplementar PP/PA/balanço hídrico aqui.
- **Análises por fase fenológica** (variáveis por fase × data, correlação por fase com a anomalia de
  produtividade, SHAP agrupado por fase, critérios ZARC por fase) são parte central do artigo —
  pedido do autor, vieram do TCC do Vitor.
- O MCP (`mcp_maiz/`, notebooks 11/11b) ficou de fora por decisão do autor (exigia pagamento).

## Decisões e bugs importantes (histórico)

### Diagnóstico dos TCCs (out/2026)
- **R² 0,77 do ExtraTrees era vazamento.** `df_wide.csv` tinha 960 linhas = 10 municípios × 16 anos ×
  6 datas, com a mesma produtividade repetida nas 6 datas; o `train_test_split(random_state=42)` pôs
  o mesmo município-ano no treino e no teste. Reavaliado: aleatório R² 0,94; deixando um município
  de fora 0,79; **deixando um ano de fora −0,57** (pior que a média histórica, −0,12). Municípios
  vizinhos compartilham o clima do ano e o modelo só reconhecia o ano.
- **Clima e produtividade desalinhados em 1 ano.** A planilha DERAL de produtividade (Oeste,
  2007–2022) está rotulada um ano depois da colheita: a quebra da safrinha 2021 (seca + geada,
  2.637 kg/ha no PR) aparece como "2022". Com a defasagem, a correlação anual ISNA fases III/IV ×
  produtividade passa de −0,27 para 0,63–0,67; contra o IBGE, de 0,11 para 0,68. Os notebooks
  antigos juntavam pelo ano civil da semeadura. Decisão: abandonar a planilha DERAL, usar o IBGE.
- **Custo também desalinhado**: o código antigo usava o 1º ano da safra DERAL (1999/00 → 1999).
- **PA negativa** no modelo antigo (fator 1 − ky(1 − ETR/ETc) sem limite em zero). A biblioteca já limita.
- **ky na fase errada**: a tabela de culturas (`dados_culturas.csv` do repositório agrometeorologia)
  tem ky 0,4/1,5/0,5/0,2 nas fases FAO-56 de Kc — o 1,5 (floração) caía no desenvolvimento
  vegetativo. Agora: Kc nas fases FAO-56 (17/28/33/22%) e ky nos 5 períodos de D&K.
- O modelo previa só 1,4% de diferença entre datas de semeadura (a produtividade observada não varia
  com a data), e o TOPSIS recomendava 25/03, a data de maior risco climático.
- Dentro de um mesmo ano, o ISNA não explica as diferenças entre municípios (correlação ≈ 0): o clima
  explica a variação entre anos; solo e manejo, a entre municípios — papel do ML no híbrido.

### Clima (01)
- Asset do autor `projects/ee-alexandrexavier/assets/BR-DWGD` (v3.2.4, 1961-01-01 a 2025-12-31, uma
  imagem por dia, bandas u2, Tmin, pr, Tmax, RH, ET, Rs). A coleção `projects/sat-io/open-datasets/BR-DWGD`
  usada no `xavier.ipynb` antigo só vai até 2020 (chuva até 2022).
- Valores inteiros: `real = bruto × BAND_*_MULT + BAND_*_ADD` (propriedades de cada imagem).
- Extração: `toBands()` por mês + `sampleRegions` nos 399 pontos (mantém `cod_ibge`), 8 threads, ~4 s/mês.
- O pacote completo em NetCDF (Dropbox) tem 95 GB — inviável no disco.

### MZA (02, `modelo_mza.py`)
- Fenologia por graus-dia (Tb 10 °C; regra de `agrometeorologiapy.data_maturacao_fisiologica`);
  **CT = 1450 °C dia** dá ~115 dias para semeadura em fevereiro no Oeste (1400 → 92/112/138 dias em
  15/01, 15/02, 15/03). Sensibilidade ±100 a fazer.
- Raiz 0,10 → 0,40 m até o fim do vegetativo; CAD = z × DTA; 30 dias de spin-up antes da semeadura.
- IAF 5, índice de colheita 0,40, umidade 13%, rota C4; Qo calculada com as funções de radiação da biblioteca.
- ZARC num ciclo de referência fixo de 100 dias com as fases da Portaria SPA/MAPA nº 329/2026
  (I 1–20, II 21–50, III 51–80, IV 81–100 dias), balanço próprio. Adverso: ISNA_I ≤ 0,70 ou ISNA_III ≤ 0,55.
- 9 datas: dias 5, 15, 25 de janeiro a março. ~39 ms por simulação; ~25 min em 4 núcleos.

### Economia (04)
- R$ de **dez/2025**, deflação pelo IPCA do **mês** de referência de cada estimativa DERAL.
- Custo DERAL por saca × produtividade de referência do rodapé (60, 80, 75, 70, 80 sc/ha).
- Renda de Fatores / Custo Total da CONAB, ano a ano (média dos municípios); lida em todos os 3
  layouts procurando o rótulo em qualquer coluna (a 1ª versão perdia 36 de 85 abas). A DERAL atual
  (custos.xlsx) dá ~0,30; a CONAB 2025, 0,28.
- **Custo operacional** (total − Renda de Fatores) é o cenário principal: a decisão de semear a
  safrinha após a soja não paga a terra de novo (a DERAL atribui a remuneração da terra cheia,
  R$ 1.491/ha, a cada cultura). Custo total = sensibilidade.
- **Só o transporte externo (~8,7% do custo variável) é proporcional à produção**; semente, adubo,
  defensivo, operações, seguro e juros são por hectare (perdidos numa quebra). O modelo antigo
  tratava todo o variável como proporcional ao kg.
- Preço mensal deflacionado mês a mês; **jul/2023 = R$ 215/saca no Ipeadata é erro de digitação**:
  valores > 2× a mediana dos vizinhos são substituídos pela média dos meses adjacentes.

### Risco e decisão (05) — desenho acordado
- Margem como distribuição (26 anos pareados) + Monte Carlo da incerteza do ML.
- Ranqueamento por **SERF** (Hardaker et al., 2004) numa faixa de aversão a risco; P(prejuízo); CVaR 10%.
- **ZARC ligado à economia pelo seguro**: PROAGRO/seguro subvencionado só para combinações aptas;
  quantificar o valor (R$/ha) de estar dentro do zoneamento.
- PCA removido; TOPSIS no máximo como comparação com o TCC.
- Avaliar os modelos também pela decisão: calibração de P(prejuízo) e arrependimento (R$/ha).

## Ambiente

- `.venv313` (Python 3.13) e kernel Jupyter `milho_pr`. O `.venv` antigo (Python 3.9) está quebrado
  (o `pip` aponta para um interpretador inexistente) e pode ser apagado.
- LightGBM e XGBoost não carregam no Mac (falta `libomp` do sistema): usar os modelos de árvore do
  scikit-learn (ExtraTrees, RandomForest, HistGradientBoosting).
- Earth Engine: projeto Cloud **`mapbiomas-brazil`** (credenciais em `~/.config/earthengine`).
- O repositório do artigo é a pasta `ML_maize_pr/` (criada pelo autor em 18/10/2026 dentro de
  `~/Python/ML_milho_oeste/`). O `.venv313` ficou na pasta de cima
  (`~/Python/ML_milho_oeste/.venv313`); o kernel `milho_pr` aponta para ele com caminho absoluto.
  **Não mover a pasta com notebook rodando**: a simulação MZA de 18/10 se perdeu porque a pasta foi
  movida durante a execução e a saída foi gravada num caminho que deixou de existir.
- Rodar um notebook: `cd notebooks && ~/Python/ML_milho_oeste/.venv313/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=milho_pr <nb>.ipynb`
- Ambiente Windows antigo do autor (dois Pythons 3.13, Anaconda × sistema) não se aplica mais a este pipeline.

## Fluxo git + Colab

O artigo vai para um **repositório novo, `ML_maize_pr`**, a ser criado pelo autor. O `aquacrop_ml`
(remoto atual desta pasta, `origin`) guarda o histórico dos TCCs e **não deve ser renomeado**.
O README já usa o contador de visitas do `ML_maize_pr`. A pasta local continua `ML_milho_oeste`.

O autor também edita pelo Google Colab, que commita direto no GitHub. Antes de commitar: `git fetch`
e `git status`; resolver com `git pull origin main --no-rebase`. Identidade do git local ao
repositório (confirmar com o autor antes de alterar). Push grande por HTTPS: `http.postBuffer`
já configurado. Nada é commitado sem pedido do autor.

## Próximos passos

1. Terminar a simulação MZA (02) e conferir: ciclo por data, PA × IBGE por ano.
2. Notebook 03: comparar MZA calibrado, ML puro e híbrido (resíduo do MZA + solo, altitude, geada);
   validação por ano e município+ano; quantis/conformal; SHAP por fase; análises por fase × data.
3. Notebook 05: risco e decisão, mapas por município.
4. Sensibilidades: CT ± 100, pesos da janela de semeadura, custo total × operacional.
5. Escrita em inglês no modelo LaTeX da revista.
