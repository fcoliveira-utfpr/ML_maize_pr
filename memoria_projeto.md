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

### Premissas do notebook 05 e fontes (levantadas em 18/10/2026)
- **Funrural 1,5%** da receita bruta (1,2% previdência + 0,1% RAT + 0,2% SENAR), produtor pessoa física:
  Lei 8.212/1991, art. 25, redação da Lei 13.606/2018. Antes de 2018 eram 2,3%; o produtor pode optar
  pela folha de salários.
- **Cobertura do seguro 65%** da produtividade esperada: faixa mínima subvencionada pelo PSR (60–65%,
  desde 2016) e nível de ~94% das apólices de milho 2ª safra em 2019 (imprensa citando o MAPA; a CNA
  propôs subir o mínimo para 65%). Produtividade esperada = histórico do produtor ou média do IBGE
  (condições gerais do seguro, MAPA). **Confirmado nos dados do PSR** (abaixo): 75% das apólices em 65%.
- **Dados do PSR/SISSER** (dados abertos do MAPA, dataset `sisser3`, CSV 2016–2024 + 2025; baixar com
  user-agent de navegador, senão o servidor devolve arquivo vazio). Os brutos têm nome e documento
  parcial dos segurados: ficam fora do repositório; só entram agregados sem dado pessoal
  (`dados/psr_milho2safra_pr_resumo_anual.csv`, `dados/psr_milho2safra_pr_cobertura.csv`).
  Milho 2ª safra PR: 132.673 apólices; cobertura 65% (75%) e 70% (19%); taxa de prêmio média 12,8% do
  valor garantido; subvenção federal 35–45% do prêmio → prêmio pago ≈ R$ 250/ha, igual ao item
  PROAGRO/SEGURO da DERAL (R$ 247/ha). 2021: 67,5% das apólices indenizadas. Eventos: seca 23,9 mil,
  geada 7,4 mil. 2025 ainda sem indenizações processadas. Usado no notebook 05 como validação
  independente (fração indenizada × quebras < 65% previstas pelo modelo, por ano).
- **Seguro/PROAGRO só dentro do ZARC**: PSR e PROAGRO exigem plantio conforme o zoneamento (MCR, cap. 16).
- **20 dias da maturação fisiológica à venda**: grão com ~30–35% de umidade na camada preta, colhido
  com 18–22% (Embrapa Milho e Sorgo); secagem no campo de 0,4–0,8 p.p./dia (Iowa State; Purdue) →
  15–25 dias. Rodar sensibilidade com 10 e 30 dias.

### Risco e decisão (05) — desenho acordado
- Margem como distribuição (26 anos pareados) + Monte Carlo da incerteza do ML.
- Ranqueamento por **SERF** (Hardaker et al., 2004) numa faixa de aversão a risco; P(prejuízo); CVaR 10%.
- **ZARC ligado à economia pelo seguro**: PROAGRO/seguro subvencionado só para combinações aptas;
  quantificar o valor (R$/ha) de estar dentro do zoneamento.
- PCA removido; TOPSIS no máximo como comparação com o TCC.
- Avaliar os modelos também pela decisão: calibração de P(prejuízo) e arrependimento (R$/ha).

## Ambiente

- `.venv313` (Python 3.13, dentro do repositório) e kernel Jupyter `milho_pr`; `pip install -r requirements.txt`.
- LightGBM e XGBoost não carregam no Mac (falta `libomp` do sistema): usar os modelos de árvore do
  scikit-learn (ExtraTrees, RandomForest, HistGradientBoosting).
- Earth Engine: projeto Cloud **`mapbiomas-brazil`** (credenciais em `~/.config/earthengine`).
- O repositório do artigo é `~/Python/ML_maize_pr` (clone de github.com/fcoliveira-utfpr/ML_maize_pr).
  O ambiente `.venv313` fica dentro dele (fora do git); o kernel `milho_pr` aponta para ele.
  Pacotes exatos em `requirements.txt`. A pasta antiga `~/Python/ML_milho_oeste` (clone do `aquacrop_ml`)
  não é mais usada.
  **Não mover a pasta com notebook rodando**: a simulação MZA de 18/10 se perdeu porque a pasta foi
  movida durante a execução e a saída foi gravada num caminho que deixou de existir.
- **Rodadas longas: usar `caffeinate -i -s`** (o Mac dormiu de ~02h a ~04h45 em 19/10 e parou APSIM e
  DSSAT sem erro). O DSSAT roda em `~/.dssat_tmp` (caminhos ≤ 80 caracteres); o APSIM em pastas temporárias.
- Rodar um notebook: `cd notebooks && ../.venv313/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=milho_pr <nb>.ipynb`
- Ambiente Windows antigo do autor (dois Pythons 3.13, Anaconda × sistema) não se aplica mais a este pipeline.

## Fluxo git + Colab

O artigo vai para um **repositório novo, `ML_maize_pr`**, a ser criado pelo autor. O `aquacrop_ml`
(remoto atual desta pasta, `origin`) guarda o histórico dos TCCs e **não deve ser renomeado**.
O README já usa o contador de visitas do `ML_maize_pr`.

O autor também edita pelo Google Colab, que commita direto no GitHub. Antes de commitar: `git fetch`
e `git status`; resolver com `git pull origin main --no-rebase`. Identidade do git local ao
repositório (confirmar com o autor antes de alterar). Push grande por HTTPS: `http.postBuffer`
já configurado. Nada é commitado sem pedido do autor.

## Resultados da primeira rodada completa (19/10/2026)

### MZA (02)
- 96.957 simulações (399 × 27 × 9), nenhum ciclo incompleto com janela de 300 dias. A 1ª rodada falhou
  porque a janela era de 220 dias (planalto frio não fechava 1450 °C dia) e o `nbconvert` saiu com
  código 0 escondendo o erro: rodar sempre com `set -o pipefail`.
- Semeadura tardia: ciclo 105 → 158 dias (05/01 → 25/03), geada no enchimento 0,04 → 1,15 dia, PA
  2.933 → ~1.750 kg/ha.

### Modelo híbrido (03) — validação deixando um ano de fora, 7.437 município-anos, 373 municípios
| Modelo | RMSE kg/ha | R² anual estadual |
|---|---|---|
| A média histórica | 1.183 | 0,35 |
| B MZA calibrado (linear: ISNA floração/enchimento + severidade da geada) | 1.140 | 0,53 |
| C ML puro (HGB raso, clima bruto, 11 vars) | 1.140 | 0,54 |
| D híbrido (HGB raso, variáveis do MZA, 10 vars) | **1.086** | **0,64** |
| E híbrido monotônico (física) — **modelo final** | 1.099 | 0,60 |
- **Lição**: ExtraTrees com ~70 variáveis decorava os anos (não superava B). O sinal climático é anual e
  regional (22 anos de treino): o modelo tem de ser raso e parcimonioso. Mesma complexidade, variáveis
  do MZA > clima bruto → é o argumento do híbrido. E escolhido pela coerência física ao prever cada data
  e por captar melhor a quebra de 2021 (todos subestimam 2021: obs. 3.069, E prevê ~3.840).
- Intervalo P10–P90 conformal: cobertura 79,4% (nominal 80%); fraca em 2020–2022 (0,44–0,64).
- SHAP: estático (altitude, DTA) ~39%, enchimento ~29%, floração ~21%; variável top: ISNA no enchimento.

### Risco e decisão (05)
- **ZARC oficial no lugar da réplica**: tábua de risco 2026/2027 (dados abertos MAPA,
  `dados/zarc_milho2safra_pr_2026_2027.csv`), Grupo I (Cod_Ciclo 20), classe AD do solo pela DTA
  (DTA mm/m ÷ 100 = mm/cm; limites IN SPA/MAPA 1/2022 → AD2 19, AD3 51, AD4 48, AD5 247, AD6 34 municípios).
  A réplica do critério hídrico (CAD = 0,40 m × DTA) concorda só 40% com o oficial — pessimista demais
  (classificava Norte/Oeste como inaptos). Fica como teste no notebook, não como critério.
  Códigos da tábua: Cod_Ciclo 20 = Grupo I, 21 = Grupo II; Cod_Solo 11–16 = AD1–AD6; dec1 = 1–10/jan.
  A janela do milho 2ª safra no PR termina em dec8 (25/03 inapto em todo o estado).
- **Risco da lavoura**: o rendimento municipal varia menos que a lavoura; τ (desvio idiossincrático em
  log) calibrado no PSR 2016–2024 = **0,33** (σ médio do município 0,26); r anual com a fração de apólices
  indenizadas = 0,86. Sem isso o seguro saía com valor negativo; com isso, +R$ 47 a +R$ 94/ha (maior no Oeste).
- Melhor data (CE, rr = 2, só datas aptas): 05/01 em 314 municípios — a data mais cedo da grade;
  faixa de fronteira do Oeste prefere 15/03. 27 municípios sem data apta (planalto sul).
  Sensibilidades: cobertura 70% (99% mesma data), ZARC Grupo II (94%), 10/30 dias até a venda (88–90%).
- Brier da P(prejuízo) fora da amostra: todos ~0,165–0,171 contra 0,247 da referência; diferenças
  pequenas entre modelos (o prejuízo depende muito do nível do município, de preço e de custo).

## Segunda rodada (19/10/2026): solo, limite do R² e APSIM

### Solo (notebook 01b)
- **DTA do balanço hídrico** = média da AWC (asset do autor `projects/fcoliveira/assets/AWC_br`, polígonos
  do mapa de solos, m³ m⁻³ × 1000 = mm m⁻¹) sobre a **lavoura temporária** (MapBiomas Coleção 10, 2024,
  classes 39 e 41) de cada município. Correlação 0,95 com a DTA antiga da sede (`dta_ponto_mm_m`).
- **Granulometria por camada** (perfil do APSIM): **MapBiomas Solo Coleção 3** (argila/areia em 10 camadas
  de 10 cm, 0–100 cm, a partir do SoilData), por sugestão do autor no lugar do SoilGrids (global).
  `dados/solo_camadas_pr.csv`. Earth Engine: inicializar com `project="mapbiomas-brazil"` (ou `fcoliveira`).

### Por que o R² municipal é ~0,40 (diagnóstico)
- Decomposição do log-rendimento IBGE (dentro da amostra): tendência + município 0,35; + efeito de ano
  estadual 0,57; + efeito ano × 12 regiões **0,69** = teto mesmo com clima perfeito. 71% dos valores do
  IBGE são múltiplos de 100 kg e 8,5% repetem o ano anterior (estimativas, não levantamento).
- Testes (validação por ano): sem valores repetidos 0,404; + latitude/longitude **0,410** (R² anual 0,66);
  hiperparâmetros 0,40–0,41; janela pelas datas aptas do ZARC 0,36 (pior); tendência por município 0,40;
  peso pela área 0,37; só municípios ≥ 1000 ha 0,30. → adotados: tirar repetidos + lat/lon.
- Choques não climáticos: 2022 superestimado por todos os modelos — hipótese: cigarrinha/enfezamento
  na safrinha 2021/22 (confirmar na literatura).

### APSIM NG sem calibração (notebook 02b, `notebooks/modelo_apsim.py`)
- Ideia do autor: APSIM sem calibrar + ML corrigindo (cf. Shahhosseini et al., 2021, Sci. Rep.).
- .NET 8 SDK instalado em `~/.dotnet` (sem sudo); ApsimX clonado e compilado em `~/Python/apsimx`
  (`dotnet ~/Python/apsimx/bin/Models.dll arquivo.apsimx --cpu-count=4`).
- Base: `Examples/Maize.apsimx`; cultivar genérica `A_100`; 6 pl m⁻²; 200 kg N ha⁻¹ (N não limitante);
  perfil de 7 camadas (0–1,8 m) com LL15/DUL/SAT por Saxton & Rawls (2006) a partir da granulometria,
  água disponível escalada para a AWC do município; MO por camada suposta (2,5% → 0,5%).
- **Bug corrigido**: a regra "Sow using a variable rule" exige chuva acumulada > MinRain (estrito); com
  MinRain = 0 só semeava se chovesse no dia → MinESW = MinRain = −1.
- Variáveis por ciclo: rendimento e biomassa simulados, nº de grãos, IAF máx., dias à floração, ciclo,
  **estresse hídrico acumulado** Σ(1 − Fw) por período (veg/flor/ench; a média do Fw dilui o sinal) e
  Tmin mínima por período. Estádios do Maize NG: 7 = florescimento, 8–10 = enchimento → maturidade.
- Piloto Cascavel: 9 datas × 27 anos em 19,5 s; rendimento simulado 8–10 t/ha (jan) → 1–5 t/ha (25/03).
- Notebook 03: modelos F (híbrido APSIM) e G (MZA + APSIM), com restrições monotônicas; o final é o de
  menor RMSE entre E, F e G.

## Hipótese ampliada (19/10/2026): qual modelo de processos rende o melhor híbrido?

Decisão do autor: comparar **MZA-FAO, AquaCrop, APSIM e DSSAT**, todos **sem calibração** (parâmetros
de fábrica; calibrar contra o rendimento só no ML), como geradores de variáveis do mesmo ML (HGB raso,
monotônico, validação deixando um ano de fora). O vencedor segue para o risco/decisão (notebook 05).
Os notebooks de exemplo de soja do autor (APSIM/AquaCrop/DSSAT, Santa Helena) serviram de base e foram
apagados por ele; trechos salvos no scratchpad da sessão.

- **Comparação justa**: mesma janela para todos os modelos (4 datas: 15/01, 05/02, 25/02, 15/03),
  municípios com histórico de 2ª safra no IBGE. Só o vencedor precisa das 9 datas × 399 municípios
  (MZA e APSIM já têm; AquaCrop/DSSAT seriam completados se vencerem).
- **AquaCrop** (`aquacrop`, AquaCrop-OSPy; `notebooks/modelo_aquacrop.py`): cultura `MaizeGDD`,
  ETo do BR-DWGD, mesmo perfil de solo do APSIM. Variáveis: rendimento e potencial simulados, razão,
  biomassa, Σ(1 − Tr/TrPot) por período (veg < HIstart ≤ flor < HIstart + Flowering ≤ ench), Tmin.
  Sem calibração ele zera o rendimento em ~8/27 anos em Cascavel (estresse de frio na polinização,
  parâmetro de milho de verão temperado) — resultado do estudo, não falha da comparação.
  ~20–25 s por município × data.
- **DSSAT** (`DSSATTools` v3; `notebooks/modelo_dssat.py`): CERES-Maize, cultivar genérica `990003`
  SHORT SEASON, modo sazonal 27 safras, 200 kg N, cultura anterior soja. O binário do pacote é Linux:
  compilar `DSSAT/dssat-csm-os` no Mac (gfortran via `brew install gcc` + cmake) em
  `~/Python/dssat-csm-os/build/bin/dscsm048` e apontar `DSSATTools.run.BIN_PATH`. Estresse = Σ WSPD
  por período (±10 dias da antese).
- `notebooks/rodar_lote.py`: execução paralela com checkpoint por município (AquaCrop e DSSAT).

## Resultado da comparação dos geradores de variáveis (19/10/2026, 06h50)

Validação deixando um ano de fora, 6.806 município-anos (sem valores repetidos), 373 municípios,
janela de 4 datas, mesmo ML (HGB raso; E, F, H, I com restrições monotônicas):

| Modelo | RMSE kg/ha | R² | R² anual | Δ RMSE vs E (IC 95%, bootstrap por anos) |
|---|---|---|---|---|
| A média histórica | 1.207 | 0,26 | 0,31 | +108 (−57; 285) |
| B MZA calibrado (linear) | 1.128 | 0,35 | 0,59 | +37 (−36; 103) |
| C ML puro (clima bruto) | 1.158 | 0,32 | 0,55 | +65 (−23; 153) |
| D MZA híbrido (sem restrição) | **1.084** | **0,40** | **0,66** | −7 (−44; 27) |
| **E MZA híbrido monotônico — final** | 1.092 | 0,39 | 0,63 | — |
| F APSIM híbrido | 1.145 | 0,33 | 0,51 | +51 (−1; 110), P(melhor que E) = 3% |
| H AquaCrop híbrido | 1.218 | 0,25 | 0,42 | +123 (0,1; 254) — pior que a média histórica |
| I DSSAT híbrido | 1.144 | 0,33 | 0,52 | +52 (11; 94) |
| J todos os modelos | 1.130 | 0,35 | 0,56 | +39 (10; 65) |

- **Achado central: o modelo mais simples (MZA-FAO) é o melhor gerador de variáveis.** Os modelos
  complexos sem calibração (APSIM, DSSAT) ficam no nível do MZA linear; o AquaCrop de fábrica piora
  a previsão (zera ~50% das combinações: cultura `MaizeGDD` de verão temperado na safrinha). Juntar
  tudo (J) piora: mais variáveis com 22 anos de treino = sobreajuste.
- **Ressalva para o artigo**: as variáveis do MZA foram escolhidas depois de testes exploratórios nos
  mesmos dados (risco de viés de seleção a favor do MZA); as do APSIM/AquaCrop/DSSAT foram definidas uma
  vez, sem ajuste. Considerar uma seleção de variáveis automática e igual para todos (ex.: dentro da
  validação) como teste de robustez.
- Brier da P(prejuízo) fora da amostra: D 0,165 (melhor), B 0,170, E 0,171, A 0,173, F 0,174,
  H 0,180 — mesma ordem.
- Simulações: APSIM 399 × 27 × 9 (96.957, todas maturaram); DSSAT e AquaCrop 373 × 27 × 4 (40.284 cada).
  DSSAT: ciclo 85–115 d, rendimento ~5 t/ha pouco sensível à data; APSIM: 7,7 → 3,3 t/ha de 05/01 a 25/03.
- O vencedor (E, MZA) já tem 9 datas × 399 municípios → o notebook 05 rodou com ele; resultados de
  decisão iguais em essência à rodada anterior (melhor data 05/01 em 327 municípios; 15/03 em 28 da
  fronteira Oeste; seguro/ZARC +R$ 54 a +R$ 107/ha; τ = 0,33, r com PSR = 0,84).

## Teste: zonas agroclimáticas GYGA no treinamento (19/10/2026)

Ideia do autor: treinar por zonas (artigo "Hydroclimatic and Thermal Characterization of Agroclimatic
Zones in Paraná", repo github.com/fcoliveira-utfpr/climate_parana; zona predominante por município na
Tabela S1). 9 zonas com dados de 2ª safra (7701: 2.017 obs; 6801: 1.588; 7601: 1.528; 7801: 944;
5901/6601/6701 quase sem dados). Mesma validação deixando um ano de fora, modelo E:

| Variante | RMSE kg/ha | Δ vs E (IC 95%, bootstrap por anos) |
|---|---|---|
| E global | 1.092 | — |
| E + zona como variável | 1.092 | +0 (−2; +2) |
| E com um modelo por zona (≥ 300 obs) | 1.098 | +6 (−14; +27) |
| E com tendência tecnológica por zona | 1.089 | −4 (−22; +16) |
| E tendência por zona + zona como variável | 1.090 | −2 (−21; +17) |

- **Zonas não melhoram o desempenho.** O efeito fixo do município já absorve o nível da zona; latitude,
  longitude, altitude e DTA já entram no ML; e as variáveis do MZA já trazem o clima da zona. Um modelo
  por zona divide os dados, mas o gargalo são os 22 anos de treino, não a heterogeneidade espacial.
- **Decisão do autor: zonas descartadas** (não entram no artigo).

## Tentativas de subir o R² (19/10/2026, 3ª rodada) — validação por ano, Δ RMSE vs E

| Variante | R² | R² anual | Δ RMSE (IC 95%) | P(melhor que E) |
|---|---|---|---|---|
| E (atual) | 0,394 | 0,634 | — | — |
| + anomalia do ano anterior (lag1) / média de 3 anos | 0,390 / 0,393 | 0,62 | +3 / +1 | 32% / 44% |
| + variáveis da data cedo (15/01) e tarde (15/03) | 0,398 | 0,655 | −4 (−18; +10) | 71% |
| + Rs, Tmax, excedente, UR por fase | 0,382 | 0,603 | +11 | 19% |
| ensemble D + E (média em log) | **0,406** | 0,650 | −11 (−29; +5) | 91% |
| ensemble B + D + E | 0,409 | 0,641 | −13 (−46; +15) | 82% |

- **Onde está o erro**: 73% do erro² é dentro do ano, entre municípios (ruído do IBGE + manejo local);
  acertar a média estadual de cada ano levaria o R² só a 0,56. Sem 2021–2022, R² = 0,42–0,44.
- **R² por escala** (modelo E / ensemble D+E): município 0,39/0,41; microrregião-ano 0,51/0,53;
  mesorregião-ano 0,58/0,60; estado-ano 0,63/0,65 → reportar o desempenho nas várias escalas.
- Conclusão: com o IBGE municipal não há ganho grande honesto; o R² de 0,77 dos TCCs era vazamento.

## Comparação de algoritmos (19/10/2026) — mesmas variáveis do híbrido (VARS_D), validação por ano

Ajuste aninhado = grade avaliada por 5 blocos de anos dentro do treino de cada dobra externa.
xgboost 3.4.1 e lightgbm 4.7.0 instalados no `.venv313` (exigiu `brew install libomp`).

| Algoritmo | R² mun. | R² anual | Δ RMSE vs E (IC 95%) |
|---|---|---|---|
| SVR RBF aninhado (escolheu C = 0,3, γ = 0,05 em 22/23 dobras) | **0,436** | **0,718** | −38 (−91; +6) |
| SVR RBF fixo (C = 0,3, γ = scale) | 0,415 | 0,708 | −20 |
| HGB / LGBM monotônicos aninhados | 0,405 / 0,403 | 0,63 / 0,62 | −11 / −9 |
| HGB D raso | 0,403 | 0,656 | −7 |
| LGBM / XGB rasos monotônicos | 0,398 / 0,395 | 0,63 | −4 / −1 |
| **E (HGB raso monotônico)** | 0,394 | 0,634 | — |
| RF raso / ET raso / Ridge splines | 0,394 / 0,390 / 0,390 | 0,57–0,65 | ~0 |
| HGB / RF / LGBM / MLP / ET padrão | 0,366–0,377 | 0,57–0,67 | +15 a +23 |
| XGB padrão | 0,294 | 0,669 | +86 |

**Ensemble SVR aninhado + E** (média em log): R² 0,435 (micro 0,550; meso 0,621; estado 0,688),
Δ RMSE −37 (−63; −16), P(melhor) = 100% — único ganho com IC fora do zero e menor variância.
- Hiperparâmetros quase não importam; configurações padrão (como nos TCCs) decoram os anos.
- O SVR não tem restrição monotônica: conferir a resposta à data de semeadura antes de adotar.
- Scripts: scratchpad da sessão (`teste_algoritmos.py`, `teste_r2.py`).

## Modelo final = ensemble E + SVR (19/10/2026, decisão do autor: "a melhor opção")

- Notebook 03: **K** = SVR RBF (C = 0,3, γ = 0,05, ε = 0,05; escolha do ajuste aninhado) sobre as variáveis
  de D **sem EC e sem DEF_floracao**; **L = média (log) de E e K = modelo final**.
- Por que tirar EC e DEF do SVR: r = 0,97 e −0,92 com ISNA_floracao; com elas o SVR repartia o efeito da
  água com sinais opostos e EC ficava com efeito contrário ao físico (dependência parcial, seção 5.1).
  Sem elas: R² 0,436 → 0,427, mas todas as variáveis no sentido físico (ISNA_enchimento com leve
  não monotonicidade no topo, efeito total positivo).
- Resultado (validação por ano): **L R² 0,427, R² anual 0,674, RMSE 1.062; Δ RMSE vs E −30 (−52; −9),
  P = 100%**; K sozinho 0,420. Cobertura P10–P90 79,4%. Brier P(prejuízo): K 0,164, L 0,165, D 0,165, E 0,171.
- SHAP do ensemble = média do TreeSHAP do HGB e do SHAP por permutação do SVR (o SVR prevê só ~1.400
  linhas/s; o PermutationExplainer padrão travou o notebook por 90 min). Grupo "reprodutivo" criado para
  geada/frio_reprod (antes caíam em "estático/ciclo").
- Decisão (05) com L: melhor data 05/01 em 311 municípios, 15/01 em 31, 15/03 em 27 (fronteira Oeste);
  data ótima igual à de E em 86%; τ = 0,33, r com PSR = 0,88; valor do seguro/ZARC +R$ 39 a +R$ 75/ha.

## Por que 27 municípios preferem semear em 15/03 (19/10/2026) — para a discussão

- Quem são: faixa lindeira ao Lago de Itaipu/fronteira do Oeste (Foz, Santa Helena, Marechal C. Rondon,
  Guaíra, Terra Roxa…) **e o Noroeste arenoso** (Umuarama, Paranavaí: Altônia, Icaraíma, Querência do
  Norte, Marilena…). Altitude média 299 m (× 583 m dos que preferem 05/01), DTA 90 × 114 mm/m.
- **Não é o preço**: o índice sazonal favorece a venda cedo (abr 1,006; jul 0,952). Com preço constante
  entre datas, a vantagem de 15/03 nesses municípios cresce (+139 → +323 R$/ha de CE) e 46 municípios
  passam a preferir 15/03.
- **É o calor no florescimento**: nesses municípios a semeadura em 05/01 floresce com Tmax média de 32,3 °C
  e 1,9 dia ≥ 35 °C; em 15/03, 27,2 °C e 0,2 dia. Geada no período reprodutivo fica desprezível (0,29 dia
  em 15/03 × 1,55 no restante do PR). Atribuição no modelo final (troca de uma variável por vez,
  05/01 → 15/03): Δ total +0,087 em log (+9%), dos quais **calor_floracao +0,076**, ISNA no enchimento
  +0,017, geada −0,010. No grupo 05/01, a semeadura tardia perde por geada (−0,036) e ISNA (−0,07).
- Leitura para o artigo: o MZA sozinho preferiria a semeadura cedo (PA 1.932 kg/ha em 05/01 × 1.339 em
  15/03 nesses municípios), porque não penaliza o calor no florescimento; **o ML do híbrido acrescenta o
  estresse térmico** — exemplo concreto do que a correção do ML capta além da física. Ressalvas: efeito
  aprendido da variação entre anos e extrapolado entre datas; risco de geada tardia e colheita mais cara
  no inverno. Robusto entre versões do modelo (com E sozinho já eram 28 municípios).
- Scripts: scratchpad da sessão (`analise_15mar.py`, `atribuicao_15mar.py`).

## Sensibilidades (19/10/2026) — notebook 06, `notebooks/sensibilidade.py` (variável de ambiente SENS)

| Cenário | R² mun. (L) | R² anual | Mesma melhor data que a base | Melhor 05/01 / 15/03 | Valor seguro médio |
|---|---|---|---|---|---|
| base (CT 1450, 4 datas iguais) | 0,427 | 0,674 | — | 311 / 27 | R$ 58 |
| CT 1350 | 0,393 | 0,611 | 78% | 284 / 12 | R$ 65 |
| **CT 1550** | **0,450** | **0,727** | 83% | 310 / 23 | R$ 77 |
| janela_cedo (pesos 0,4/0,3/0,2/0,1) | 0,384 | 0,593 | 91% | 310 / 24 | R$ 83 |
| janela_7datas (15/01–15/03) | 0,430 | 0,669 | 90% | 323 / 26 | R$ 58 |
| custo total | — | — | 98% | — | (margem média R$ 352, P(prejuízo) 0,53) |
| cobertura 70% / ZARC Grupo II / 10–30 dias até a venda | — | — | 99% / 95% / 90% | — | — |

- Recomendação robusta: 05/01 domina em todos os cenários; 15/03 sempre na faixa Oeste/Noroeste.
- **CT 1550 melhora o modelo** (ciclo 115 → 164 d de 05/01 a 15/03): a soma térmica de 1450 pode estar
  curta para os híbridos atuais. Decisão pendente do autor: manter 1450 (literatura) ou adotar 1550
  (escolha pelo próprio R² da validação = ajuste de 1 hiperparâmetro; testar 1650 antes de decidir).
- Pesos concentrados nas datas cedo pioram (0,384): o rendimento municipal reflete semeaduras espalhadas.

## Robustez: seleção automática e igual de variáveis para os 4 geradores (19/10/2026)

Dentro de cada dobra (só anos de treino): top-k por |Spearman| com a anomalia, descartando redundantes
(|r| > 0,9); HGB raso sem restrição para todos; pools só com saídas de processo (MZA 37 candidatas:
PPf, PA, EC, ciclo, ISNA/DEF/EXC/geada/calor/Tmin por fase; APSIM 12; AquaCrop 12; DSSAT 10).

| Gerador | R² k=8 | R² k=12 | Δ RMSE vs MZA (k=8; IC 95%) |
|---|---|---|---|
| MZA | 0,326 | 0,337 | — |
| APSIM | 0,319 | 0,304 | +6 (−39; +51) |
| AquaCrop | 0,245 | 0,245 | +69 (−64; +173) |
| DSSAT | 0,330 | 0,330 | −4 (−44; +33) |

- **Com seleção automática igual, MZA ≈ DSSAT ≈ APSIM**; AquaCrop de fábrica continua o pior.
- A vantagem do MZA na comparação principal (E 0,394 × F/I 0,334) vem das variáveis escolhidas à mão
  (ISNA por fase, PA/PPf, geada/frio/calor reprodutivo) e das restrições monotônicas — e essas variáveis
  foram definidas após testes exploratórios nos mesmos dados (viés de seleção não separável do
  conhecimento agronômico). Para APSIM/DSSAT, à mão e automático dão o mesmo (~0,33).
- **Reenquadrar o achado central**: não "o MZA é o melhor gerador", e sim "o modelo mais simples, sem
  calibração, é tão bom quanto APSIM/DSSAT; a complexidade sem calibração não acrescenta informação";
  o ganho final vem do desenho de variáveis guiado pela agronomia + física no ML (monotonia) + ensemble.
- Script: scratchpad da sessão (`selecao_auto.py`).

## Próximos passos

1. Restrição da colheita da soja (05/01 só com soja superprecoce): **vai para a discussão do artigo**
   (decisão do autor, 19/10/2026), sem limitar a janela no modelo — "o mais cedo possível" é a mensagem.
2. ~~Preferência por 15/03~~ → explicada (calor no florescimento; ver seção acima), vai para a discussão.
3. ~~Sensibilidades~~ → feitas. CT: **mantido 1450 no principal** (literatura); 1550 vai como sensibilidade
   (indício de soma térmica curta para híbridos atuais). Autor dispensou o teste de 1650 (19/10/2026).
4. Opcional (computacional): calibrar a profundidade de raiz/CAD da réplica para reproduzir o ZARC oficial.
5. Escrita em inglês no modelo LaTeX da revista.
