# Extrator IMASUL — CREA-MS (V4 com mapa)

## Instalação
1. Extraia a pasta `imasul_extrator`.
2. Abra o terminal dentro da pasta.
3. Execute: `py -m pip install -r requirements.txt`
4. Execute: `streamlit run app.py`

## Mapa de plantio florestal
Selecione a base `Plantio florestal - Consulta Classes EFlorestal (Polígono)`.
Expanda `Visualizar áreas de plantio florestal no mapa` e clique em `Carregar áreas no mapa`.
A visualização busca geometrias em lotes pequenos e limita a quantidade de polígonos para preservar o desempenho. Use a extração para coletar a base completa.

Os dados e checkpoints ficam em `dados_imasul`, ao lado de `app.py`. Preserve essa pasta para manter a possibilidade de retomada.
