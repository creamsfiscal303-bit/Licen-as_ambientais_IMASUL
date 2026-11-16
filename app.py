"""
Fiscalização de PLANTIO FLORESTAL — IMASUL / CREA-MS (v3: mapa, agentes e rotas)

Fonte: camada "licencas_ambientais" (layer 16) do IMASUL, que traz tipo de empreendimento,
detentor, CNPJ, imóvel, município, datas e a geometria dos imóveis.

Fluxo: filtros (lateral) -> baixar dados -> abas: Dados | Por município | Mapa | Agentes e rotas
"""
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

st.set_page_config(page_title="Fiscalização de Plantio | IMASUL", page_icon="🌱", layout="wide")

# ----------------------------------------------------------------------------------------
# CONFIGURAÇÕES (se o IMASUL mudar algo, é aqui que você mexe)
# ----------------------------------------------------------------------------------------
LAYER_URL = "https://www.pinms.ms.gov.br/arcgis/rest/services/IMASUL/licencas_ambientais/FeatureServer/16"
QUERY_URL = f"{LAYER_URL}/query"
PAGE_SIZE = 2000          # limite do servidor (maxRecordCount) quando só pedimos o ponto central
PAGE_SIZE_GEOM = 500      # páginas menores quando precisamos baixar a forma do imóvel
TIMEOUT = (20, 180)
FUSO = "America/Campo_Grande"
ANO_MIN = 2012
ANO_MAX = datetime.now().year
STATUS_POSSIVEIS = ["Emitida", "Cancelada", "Suspensa", "Gerada"]
PALAVRAS_PADRAO = ("PLANTIO DE FLORESTA", "REFLORESTAMENTO")
SEM_MUNICIPIO = "(SEM MUNICÍPIO)"
SEM_AGENTE = "(sem agente)"

try:
    PASTA_APP = Path(__file__).resolve().parent
except NameError:
    PASTA_APP = Path.cwd()
PASTA_DADOS = PASTA_APP / "dados_fiscalizacao"
PASTA_DADOS.mkdir(exist_ok=True)
ARQ_AGENTES = PASTA_DADOS / "agentes_municipios.csv"

CAMPOS = [
    "objectid", "num_autorizacao", "num_processo", "id_empreendimento",
    "data_expedicao", "data_validade", "validade_licenca", "status_licenca", "tipo_licenca",
    "id_tipo_empreendimento", "tipo_empreendimento",
    "nome_detentor", "cnpj", "nome_imovel", "municipio", "cod_mun",
    "sicar", "car", "car_id", "area_informada", "situacao_cadastro", "coordenadas_imovel",
]
COLUNAS_DATA = ["data_expedicao", "data_validade"]
PALETA = [
    [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
    [240, 50, 230], [128, 128, 0], [0, 128, 128], [170, 110, 40], [128, 0, 0], [0, 0, 128],
]
COR_SEM_AGENTE = [150, 150, 150]


# ----------------------------------------------------------------------------------------
# COMUNICAÇÃO COM O SERVIDOR ARCGIS
# ----------------------------------------------------------------------------------------
class ErroArcGIS(Exception):
    """Erro devolvido pelo próprio servidor (ex.: consulta inválida). Não adianta tentar de novo."""


def arcgis_query(params, tentativas=4):
    ultimo_erro = None
    for t in range(tentativas):
        try:
            resp = requests.post(QUERY_URL, data={**params, "f": "json"}, timeout=TIMEOUT)
            resp.raise_for_status()
            dados = resp.json()
            if "error" in dados:
                err = dados["error"]
                raise ErroArcGIS(f"{err.get('message', 'erro')} {err.get('details', [])}")
            return dados
        except ErroArcGIS:
            raise
        except Exception as exc:
            ultimo_erro = exc
            if t < tentativas - 1:
                time.sleep(min(2 ** t, 8))
    raise RuntimeError(f"Falha de comunicação com o IMASUL: {ultimo_erro}")


def sql_texto(valor):
    return "'" + str(valor).replace("'", "''") + "'"


@st.cache_data(ttl=3600, show_spinner=False)
def listar_tipos():
    estat = [{"statisticType": "count", "onStatisticField": "objectid", "outStatisticFieldName": "total"}]
    dados = arcgis_query({
        "where": "tipo_empreendimento IS NOT NULL",
        "groupByFieldsForStatistics": "tipo_empreendimento",
        "outStatistics": json.dumps(estat),
    })
    tipos = {}
    for f in dados.get("features", []):
        a = f["attributes"]
        nome = a.get("tipo_empreendimento")
        if nome and str(nome).strip():
            tipos[nome] = tipos.get(nome, 0) + int(a.get("total") or 0)
    return dict(sorted(tipos.items(), key=lambda kv: -kv[1]))


@st.cache_data(ttl=300, show_spinner=False)
def contar_registros(where):
    return int(arcgis_query({"where": where, "returnCountOnly": "true"}).get("count", 0))


# ----------------------------------------------------------------------------------------
# GEOMETRIA E ROTAS (funções puras, fáceis de testar)
# ----------------------------------------------------------------------------------------
def centroide_de_aneis(aneis):
    """Centro do maior anel (polígono) do imóvel. Usado só se o servidor não devolver o centroide."""
    melhor, melhor_area = None, -1.0
    for anel in aneis or []:
        pts = [(p[0], p[1]) for p in anel]
        if len(pts) < 3:
            continue
        a2 = cx = cy = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            cruz = x0 * y1 - x1 * y0
            a2 += cruz
            cx += (x0 + x1) * cruz
            cy += (y0 + y1) * cruz
        area = abs(a2) / 2
        if area > melhor_area:
            melhor_area = area
            if a2 != 0:
                melhor = (cx / (3 * a2), cy / (3 * a2))
            else:
                melhor = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return melhor if melhor else (None, None)


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em linha reta entre pontos (km). Aceita números ou arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def ordenar_vizinho_proximo(lat, lon, inicio_lat, inicio_lon):
    """Visita sempre o ponto não visitado mais próximo. Devolve as posições na ordem de visita."""
    n = len(lat)
    visitado = np.zeros(n, dtype=bool)
    ordem = []
    atual = (inicio_lat, inicio_lon)
    for _ in range(n):
        d = haversine_km(atual[0], atual[1], lat, lon)
        d[visitado] = np.inf
        i = int(np.argmin(d))
        ordem.append(i)
        visitado[i] = True
        atual = (lat[i], lon[i])
    return ordem


def montar_rota(df_agente, municipio_base, pontos_por_dia):
    """Roteiro município a município: ordena os municípios pelo mais próximo a partir da base
    e, dentro de cada um, as licenças pelo vizinho mais próximo. Distâncias em LINHA RETA."""
    df = df_agente.dropna(subset=["lat", "lon"]).copy()
    if df.empty:
        return df, municipio_base
    cent = df.groupby("municipio_norm")[["lat", "lon"]].mean()
    if municipio_base not in cent.index:
        municipio_base = df["municipio_norm"].value_counts().index[0]
    inicio = (cent.loc[municipio_base, "lat"], cent.loc[municipio_base, "lon"])

    restantes, ordem_mun, atual = list(cent.index), [], inicio
    while restantes:
        d = [haversine_km(atual[0], atual[1], cent.loc[m, "lat"], cent.loc[m, "lon"]) for m in restantes]
        escolhido = restantes.pop(int(np.argmin(d)))
        ordem_mun.append(escolhido)
        atual = (cent.loc[escolhido, "lat"], cent.loc[escolhido, "lon"])

    partes, atual = [], inicio
    for mun in ordem_mun:
        g = df[df["municipio_norm"] == mun]
        idx = ordenar_vizinho_proximo(g["lat"].to_numpy(), g["lon"].to_numpy(), atual[0], atual[1])
        g = g.iloc[idx]
        atual = (g["lat"].iloc[-1], g["lon"].iloc[-1])
        partes.append(g)
    rota = pd.concat(partes).reset_index(drop=True)
    rota.insert(0, "ordem", np.arange(1, len(rota) + 1))
    rota.insert(1, "dia", (rota["ordem"] - 1) // max(int(pontos_por_dia), 1) + 1)
    lat, lon = rota["lat"].to_numpy(), rota["lon"].to_numpy()
    anterior_lat = np.r_[inicio[0], lat[:-1]]
    anterior_lon = np.r_[inicio[1], lon[:-1]]
    rota["km_da_parada_anterior"] = np.round(haversine_km(anterior_lat, anterior_lon, lat, lon), 1)
    rota["link_google_maps"] = [f"https://www.google.com/maps?q={a:.6f},{b:.6f}" for a, b in zip(lat, lon)]
    return rota, municipio_base


COLUNAS_ROTA = [
    "ordem", "dia", "municipio", "num_autorizacao", "num_processo", "tipo_empreendimento",
    "nome_detentor", "cnpj", "nome_imovel", "data_expedicao", "validade_licenca", "data_validade",
    "status_licenca", "area_informada", "sicar", "lat", "lon", "km_da_parada_anterior", "link_google_maps",
]


# ----------------------------------------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------------------------------------
def parametros_pagina(where, offset, modo):
    p = {
        "where": where, "outFields": ",".join(CAMPOS), "orderByFields": "objectid ASC",
        "resultOffset": offset, "outSR": 4326, "returnZ": "false",
    }
    if modo == "centroide":
        p.update({"returnGeometry": "false", "returnCentroid": "true", "resultRecordCount": PAGE_SIZE})
    else:  # baixa a forma simplificada (~50 m) e calcula o centro aqui
        p.update({"returnGeometry": "true", "maxAllowableOffset": 0.0005, "geometryPrecision": 5,
                  "resultRecordCount": PAGE_SIZE_GEOM})
    return p


def extrair_xy(feicao, modo):
    if modo == "centroide":
        c = feicao.get("centroid") or {}
        return c.get("x"), c.get("y")
    return centroide_de_aneis((feicao.get("geometry") or {}).get("rings"))


def baixar_tudo(where, total, ao_progredir=None):
    """Baixa os registros com o ponto central de cada imóvel. Devolve (linhas, modo_usado)."""
    modo = "centroide"
    linhas, offset = [], 0
    for _ in range(20_000):  # trava de segurança
        try:
            dados = arcgis_query(parametros_pagina(where, offset, modo))
        except ErroArcGIS:
            if modo == "centroide" and offset == 0:
                modo = "geometria"
                continue
            raise
        feicoes = dados.get("features", [])
        if modo == "centroide" and offset == 0 and feicoes and not any("centroid" in f for f in feicoes[:20]):
            modo = "geometria"   # servidor ignorou returnCentroid: refaz a primeira página com a forma
            continue
        for f in feicoes:
            attrs = dict(f["attributes"])
            attrs["lon"], attrs["lat"] = extrair_xy(f, modo)
            linhas.append(attrs)
        offset += len(feicoes)
        if ao_progredir:
            ao_progredir(len(linhas), total)
        tamanho = PAGE_SIZE if modo == "centroide" else PAGE_SIZE_GEOM
        if not feicoes or (len(feicoes) < tamanho and not dados.get("exceededTransferLimit")):
            break
    return linhas, modo


# ----------------------------------------------------------------------------------------
# FILTROS E TRATAMENTO
# ----------------------------------------------------------------------------------------
def montar_where(tipos, ano_ini, ano_fim, status):
    partes = []
    if tipos:
        partes.append("tipo_empreendimento IN (" + ",".join(sql_texto(t) for t in tipos) + ")")
    if (ano_ini, ano_fim) != (ANO_MIN, ANO_MAX):
        partes.append(f"data_expedicao >= DATE '{ano_ini}-01-01' AND data_expedicao < DATE '{ano_fim + 1}-01-01'")
    if status and set(status) != set(STATUS_POSSIVEIS):
        partes.append("status_licenca IN (" + ",".join(sql_texto(s) for s in status) + ")")
    return " AND ".join(f"({p})" for p in partes) if partes else "1=1"


def esta_vazio(serie):
    return serie.isna() | (serie.astype(str).str.strip() == "")


def normalizar_municipio(serie):
    s = serie.fillna("").astype(str).str.strip().str.upper()
    return s.where(s != "", SEM_MUNICIPIO)


def preparar_dataframe(linhas):
    df = pd.DataFrame(linhas, columns=CAMPOS + ["lon", "lat"])
    for col in COLUNAS_DATA:
        ms = pd.to_numeric(df[col], errors="coerce")
        df[col] = pd.to_datetime(ms, unit="ms", utc=True).dt.tz_convert(FUSO).dt.date
    ano = pd.to_datetime(df["data_expedicao"], errors="coerce").dt.year.astype("Int64")
    df.insert(df.columns.get_loc("data_expedicao") + 1, "ano_expedicao", ano)
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["area_informada"] = pd.to_numeric(df["area_informada"], errors="coerce")
    df["municipio_norm"] = normalizar_municipio(df["municipio"])
    return df


def remover_duplicadas(df):
    chave = [c for c in ["num_autorizacao", "num_processo", "id_empreendimento",
                         "tipo_empreendimento", "sicar", "car"] if c in df.columns]
    return df.drop_duplicates(subset=chave, keep="first")


# ----------------------------------------------------------------------------------------
# AGENTES FISCAIS (município -> agente), guardado em CSV na pasta dados_fiscalizacao
# ----------------------------------------------------------------------------------------
def carregar_atribuicoes():
    if not ARQ_AGENTES.exists():
        return {}
    try:
        t = pd.read_csv(ARQ_AGENTES, sep=";", dtype=str, encoding="utf-8-sig").fillna("")
        return {m.strip().upper(): a.strip() for m, a in zip(t["municipio"], t["agente"]) if a.strip()}
    except Exception:
        return {}


def salvar_atribuicoes(mapa):
    t = pd.DataFrame(sorted(mapa.items()), columns=["municipio", "agente"])
    t.to_csv(ARQ_AGENTES, sep=";", index=False, encoding="utf-8-sig")


def ler_csv_atribuicoes(arquivo):
    t = pd.read_csv(arquivo, sep=None, engine="python", dtype=str, encoding="utf-8-sig").fillna("")
    t.columns = [c.strip().lower() for c in t.columns]
    if not {"municipio", "agente"} <= set(t.columns):
        raise ValueError("O CSV precisa ter as colunas 'municipio' e 'agente'.")
    return {m.strip().upper(): a.strip() for m, a in zip(t["municipio"], t["agente"]) if m.strip() and a.strip()}


# ----------------------------------------------------------------------------------------
# EXPORTAÇÃO
# ----------------------------------------------------------------------------------------
def nome_aba(nome, usados):
    base = re.sub(r"[\[\]:*?/\\]", "-", str(nome)).strip()[:31] or "Aba"
    cand, i = base, 2
    while cand.lower() in usados:
        sufixo = f"_{i}"
        cand = base[:31 - len(sufixo)] + sufixo
        i += 1
    usados.add(cand.lower())
    return cand


def formatar_planilha(writer, colunas_data=COLUNAS_DATA):
    for ws in writer.book.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            largura = min(max((len(str(c.value or "")) for c in list(col)[:300]), default=10) + 2, 45)
            ws.column_dimensions[col[0].column_letter].width = max(12, largura)
        for cel in ws[1]:
            if cel.value in colunas_data:
                for linha in ws.iter_rows(min_row=2, min_col=cel.column, max_col=cel.column):
                    linha[0].number_format = "DD/MM/YYYY"


def meta_df(where, total_servidor, linhas_brutas, linhas_arquivo):
    return pd.DataFrame([
        ["Fonte", "IMASUL - camada licencas_ambientais (imóveis rurais)"],
        ["Serviço ArcGIS REST", LAYER_URL],
        ["Consulta aplicada", where],
        ["Data/hora da extração (UTC)", datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M:%S")],
        ["Registros no servidor para esta consulta", total_servidor],
        ["Linhas baixadas", linhas_brutas],
        ["Linhas no arquivo", linhas_arquivo],
        ["Coordenadas", "Ponto central aproximado do imóvel (SIRGAS 2000/WGS84). Distâncias em linha reta, não por estrada."],
        ["Observação", "Dados públicos do serviço consultado. Confirmar na fonte oficial antes de uso em procedimento fiscalizatório."],
    ], columns=["Item", "Informação"])


@st.cache_data(show_spinner="Gerando Excel...")
def gerar_excel(df, where, total_servidor, linhas_brutas):
    saida = io.BytesIO()
    with pd.ExcelWriter(saida, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Licencas")
        meta_df(where, total_servidor, linhas_brutas, len(df)).to_excel(w, index=False, sheet_name="Metadados")
        formatar_planilha(w)
    return saida.getvalue()


def excel_abas(abas, resumo, meta):
    """abas: {nome do agente: DataFrame da rota}. Uma aba por agente + Resumo + Metadados."""
    saida = io.BytesIO()
    usados = set()
    with pd.ExcelWriter(saida, engine="openpyxl") as w:
        resumo.to_excel(w, index=False, sheet_name=nome_aba("Resumo", usados))
        for nome, tabela in abas.items():
            tabela.to_excel(w, index=False, sheet_name=nome_aba(nome, usados))
        meta.to_excel(w, index=False, sheet_name=nome_aba("Metadados", usados))
        formatar_planilha(w)
    return saida.getvalue()


def rota_para_exportar(rota):
    t = rota.copy()
    t["lat"], t["lon"] = t["lat"].round(6), t["lon"].round(6)
    return t[[c for c in COLUNAS_ROTA if c in t.columns]]


# ----------------------------------------------------------------------------------------
# MAPA
# ----------------------------------------------------------------------------------------
def cores_dos_agentes(agentes):
    return {a: PALETA[i % len(PALETA)] for i, a in enumerate(sorted(agentes))}


def preparar_para_mapa(df, cores):
    m = df.dropna(subset=["lat", "lon"]).copy()
    m["cor"] = m["agente"].map(lambda a: cores.get(a, COR_SEM_AGENTE))
    for c in ["nome_detentor", "nome_imovel", "municipio_norm", "num_autorizacao", "status_licenca",
              "validade_licenca", "agente"]:
        m[c] = m[c].fillna("").astype(str)
    m["tipo_curto"] = m["tipo_empreendimento"].fillna("").astype(str).str.slice(0, 70)
    m["expedicao"] = m["data_expedicao"].astype(str).replace("NaT", "")
    m["lat"], m["lon"] = m["lat"].astype(float), m["lon"].astype(float)
    return m[["lon", "lat", "cor", "agente", "municipio_norm", "nome_detentor", "nome_imovel",
              "num_autorizacao", "tipo_curto", "expedicao", "status_licenca", "validade_licenca"]]


TOOLTIP = {
    "html": "<b>{municipio_norm}</b><br/>{nome_detentor}<br/>Imóvel: {nome_imovel}<br/>"
            "Licença: {num_autorizacao} ({status_licenca})<br/>{tipo_curto}<br/>"
            "Expedição: {expedicao} · {validade_licenca}<br/>Agente: {agente}",
    "style": {"backgroundColor": "#222", "color": "white", "fontSize": "12px"},
}


def vista_inicial(m, zoom=5.3):
    if m.empty:
        return pdk.ViewState(latitude=-20.5, longitude=-54.6, zoom=5.3)
    return pdk.ViewState(latitude=float(m["lat"].mean()), longitude=float(m["lon"].mean()), zoom=zoom)


def desenhar_mapa(m, modo="Pontos", rota=None):
    camadas = []
    if modo == "Mapa de calor":
        camadas.append(pdk.Layer("HeatmapLayer", data=m[["lon", "lat"]], get_position="[lon, lat]",
                                 radius_pixels=40, opacity=0.8))
    else:
        camadas.append(pdk.Layer("ScatterplotLayer", data=m, get_position="[lon, lat]", get_fill_color="cor",
                                 get_radius=600, radius_min_pixels=3, radius_max_pixels=12, pickable=True))
    if rota is not None and len(rota) > 1:
        caminho = [{"path": rota[["lon", "lat"]].astype(float).values.tolist()}]
        camadas.insert(0, pdk.Layer("PathLayer", data=caminho, get_path="path", get_color=[60, 60, 60],
                                    width_min_pixels=2))
    deck = pdk.Deck(layers=camadas, initial_view_state=vista_inicial(m), tooltip=TOOLTIP, map_style="light")
    st.pydeck_chart(deck, use_container_width=True)


# ========================================================================================
# TELA
# ========================================================================================
st.title("🌱 Fiscalização de Plantio Florestal — IMASUL")
st.caption("CREA-MS · Dados abertos do IMASUL (ArcGIS REST) · Mapa, agentes por município e roteiros")
st.warning("Esta camada não necessariamente representa todas as licenças concedidas pelo IMASUL. "
           "Valide os dados na fonte oficial antes de qualquer uso fiscalizatório.")

try:
    tipos_disponiveis = listar_tipos()
except Exception as exc:
    st.error(f"Não foi possível consultar o IMASUL agora: {exc}")
    st.stop()

with st.sidebar:
    st.header("Filtros do IMASUL")
    padrao = [t for t in tipos_disponiveis if any(p in t.upper() for p in PALAVRAS_PADRAO)]
    tipos_escolhidos = st.multiselect(
        "Tipo de empreendimento", options=list(tipos_disponiveis), default=padrao,
        format_func=lambda t: f"{t[:90]}{'…' if len(t) > 90 else ''}  ({tipos_disponiveis[t]:,})",
        help="Já vem marcado: plantio de floresta e reflorestamento.",
    )
    ano_ini, ano_fim = st.slider("Ano de expedição", ANO_MIN, ANO_MAX, (ANO_MIN, ANO_MAX))
    status = st.multiselect("Situação da licença", STATUS_POSSIVEIS, default=STATUS_POSSIVEIS)

if not tipos_escolhidos:
    st.info("Escolha ao menos um tipo de empreendimento na barra lateral.")
    st.stop()
if not status:
    st.info("Escolha ao menos uma situação de licença.")
    st.stop()

where = montar_where(tipos_escolhidos, ano_ini, ano_fim, status)
try:
    total = contar_registros(where)
except Exception as exc:
    st.error(f"Consulta recusada pelo servidor: {exc}")
    st.code(where)
    st.stop()

st.metric("Registros no servidor para estes filtros", f"{total:,}")
with st.expander("Consulta enviada ao servidor"):
    st.code(where, language="sql")
if total == 0:
    st.info("Nenhum registro para estes filtros.")
    st.stop()

if st.button("⬇️ Baixar dados do IMASUL (com localização)", type="primary", use_container_width=True):
    barra = st.progress(0.0, text="Iniciando...")
    try:
        linhas, modo = baixar_tudo(
            where, total,
            lambda feitos, tot: barra.progress(min(1.0, feitos / max(tot, 1)), text=f"{feitos:,} de {tot:,} registros"),
        )
        st.session_state["resultado"] = {"where": where, "linhas": linhas, "total": total, "modo": modo}
    except Exception as exc:
        st.error(f"Download interrompido: {exc}. Clique no botão de novo para recomeçar.")
    barra.empty()

resultado = st.session_state.get("resultado")
if not resultado:
    st.stop()
if resultado["where"] != where:
    st.info("Os filtros mudaram desde o último download. Clique em 'Baixar dados' para atualizar.")
    st.stop()

df = preparar_dataframe(resultado["linhas"])
brutas = len(df)

with st.sidebar:
    st.header("Filtros locais")
    unica = st.checkbox("Uma linha por licença (remove polígonos repetidos)", value=True,
                        help="Se o mesmo número de licença cobrir imóveis sem SICAR/CAR preenchido, "
                             "eles serão tratados como um só.")
    opcoes_validade = sorted(df["validade_licenca"].dropna().astype(str).unique())
    validade_sel = st.multiselect("Validade da licença", opcoes_validade, default=opcoes_validade)
    busca = st.text_input("🔎 Texto em detentor, imóvel ou município")

if unica:
    df = remover_duplicadas(df)
if opcoes_validade and set(validade_sel) != set(opcoes_validade):
    df = df[df["validade_licenca"].astype(str).isin(validade_sel)]
if busca.strip():
    mascara = pd.Series(False, index=df.index)
    for c in ["nome_detentor", "nome_imovel", "municipio"]:
        mascara |= df[c].fillna("").astype(str).str.contains(busca.strip(), case=False, regex=False)
    df = df[mascara]

atrib = carregar_atribuicoes()
df = df.copy()
df["agente"] = df["municipio_norm"].map(atrib).fillna(SEM_AGENTE)

sem_coord = int(df["lat"].isna().sum())
if sem_coord:
    st.warning(f"{sem_coord:,} registro(s) sem coordenadas: aparecem na tabela, mas não no mapa nem nas rotas.")
if resultado.get("modo") == "geometria":
    st.caption("ℹ️ O servidor não devolveu o ponto central; ele foi calculado a partir da forma simplificada do imóvel.")

aba_dados, aba_mun, aba_mapa, aba_agentes = st.tabs(["📋 Dados", "📊 Por município", "🗺️ Mapa", "👮 Agentes e rotas"])

# ------------------------------------------------------------------ DADOS
with aba_dados:
    m1, m2, m3 = st.columns(3)
    m1.metric("Linhas baixadas", f"{brutas:,}")
    m2.metric("Linhas na tabela", f"{len(df):,}")
    m3.metric("Licenças distintas (nº autorização)", f"{df['num_autorizacao'].nunique():,}")
    if len(df):
        st.markdown("**Preenchimento dos dados de identificação** (licenças antigas costumam vir em branco)")
        campos_id = ["nome_detentor", "cnpj", "nome_imovel", "municipio", "sicar"]
        st.dataframe(pd.DataFrame({
            "campo": campos_id,
            "% preenchido": [round(100 * (~esta_vazio(df[c])).mean(), 1) for c in campos_id],
        }), hide_index=True, use_container_width=True)
        st.markdown("**Licenças por ano de expedição**")
        st.bar_chart(df["ano_expedicao"].value_counts().sort_index())
    exportavel = df.drop(columns=["municipio_norm"])
    st.dataframe(exportavel.head(1000), use_container_width=True, hide_index=True, height=420)
    if len(df) > 1000:
        st.caption("Prévia limitada a 1.000 linhas. Os arquivos para download têm todas.")
    st.download_button(
        "⬇️ Baixar Excel (.xlsx)", data=gerar_excel(exportavel, where, resultado["total"], brutas),
        file_name=f"imasul_plantio_{datetime.now():%Y%m%d_%H%M}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    st.download_button(
        "⬇️ Baixar CSV", data=exportavel.to_csv(index=False, sep=";", encoding="utf-8-sig"),
        file_name=f"imasul_plantio_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv", use_container_width=True)

# ------------------------------------------------------------------ POR MUNICÍPIO
with aba_mun:
    por_mun = (df.groupby("municipio_norm")
                 .agg(licencas=("objectid", "size"), area_informada_total=("area_informada", "sum"))
                 .sort_values("licencas", ascending=False).reset_index())
    por_mun["agente"] = por_mun["municipio_norm"].map(atrib).fillna(SEM_AGENTE)
    por_mun["area_informada_total"] = por_mun["area_informada_total"].round(1)
    n_sem = int(por_mun.loc[por_mun["municipio_norm"] == SEM_MUNICIPIO, "licencas"].sum())
    c1, c2 = st.columns(2)
    c1.metric("Municípios com licenças", f"{(por_mun['municipio_norm'] != SEM_MUNICIPIO).sum():,}")
    c2.metric("Licenças sem município informado", f"{n_sem:,}")
    if n_sem:
        st.caption("Registros sem município (comum em licenças antigas) ficam no grupo '(SEM MUNICÍPIO)'.")
    st.markdown("**20 municípios com mais licenças**")
    st.bar_chart(por_mun.head(20).set_index("municipio_norm")["licencas"])
    st.dataframe(por_mun.rename(columns={"municipio_norm": "município"}), hide_index=True,
                 use_container_width=True, height=420)

# ------------------------------------------------------------------ MAPA
with aba_mapa:
    agentes_existentes = sorted(set(df["agente"]))
    cores = cores_dos_agentes([a for a in agentes_existentes if a != SEM_AGENTE])
    cA, cB = st.columns([1, 2])
    modo_mapa = cA.radio("Visualização", ["Pontos", "Mapa de calor"], horizontal=True)
    agentes_mapa = cB.multiselect("Agentes no mapa", agentes_existentes, default=agentes_existentes)
    mapa_df = preparar_para_mapa(df[df["agente"].isin(agentes_mapa)], cores)
    st.caption(f"{len(mapa_df):,} pontos. Cada ponto é o centro aproximado do imóvel licenciado; cor = agente responsável pelo município.")
    desenhar_mapa(mapa_df, modo_mapa)
    if len(cores):
        st.markdown(" ".join(
            f"<span style='color:rgb({c[0]},{c[1]},{c[2]})'>●</span> {a}" for a, c in cores.items()
        ), unsafe_allow_html=True)

# ------------------------------------------------------------------ AGENTES E ROTAS
with aba_agentes:
    st.subheader("1) Agentes e municípios")
    st.caption(f"A atribuição fica salva em `{ARQ_AGENTES.name}` (pasta dados_fiscalizacao) e vale para qualquer filtro.")
    agentes_salvos = sorted(set(atrib.values()))
    texto_agentes = st.text_area("Agentes fiscais (um nome por linha)", value="\n".join(agentes_salvos), height=120)
    lista_agentes = sorted({a.strip() for a in texto_agentes.splitlines() if a.strip()})

    with st.expander("Importar atribuição de um CSV (colunas: municipio;agente)"):
        arq = st.file_uploader("CSV", type=["csv"], key="upload_atrib")
        if arq is not None and st.button("Aplicar importação"):
            try:
                novo = ler_csv_atribuicoes(arq)
                salvar_atribuicoes({**atrib, **novo})
                st.success(f"{len(novo)} município(s) importados.")
                st.rerun()
            except Exception as exc:
                st.error(f"Não foi possível importar: {exc}")

    editor_base = (df.groupby("municipio_norm").size().sort_values(ascending=False).rename("licencas").reset_index())
    editor_base = editor_base[editor_base["municipio_norm"] != SEM_MUNICIPIO]
    editor_base["agente"] = editor_base["municipio_norm"].map(atrib).fillna("")
    editor_base = editor_base.rename(columns={"municipio_norm": "municipio"})
    opcoes_agente = [""] + sorted(set(lista_agentes) | set(editor_base["agente"]) - {""})
    editado = st.data_editor(
        editor_base, hide_index=True, use_container_width=True, height=300, key="editor_atrib",
        disabled=["municipio", "licencas"],
        column_config={"agente": st.column_config.SelectboxColumn("agente", options=opcoes_agente)},
    )
    if st.button("💾 Salvar atribuição"):
        novo_mapa = dict(atrib)
        for mun, ag in zip(editado["municipio"], editado["agente"]):
            if ag and str(ag).strip():
                novo_mapa[mun] = str(ag).strip()
            else:
                novo_mapa.pop(mun, None)
        salvar_atribuicoes(novo_mapa)
        st.success("Atribuição salva.")
        st.rerun()

    sem_agente = df[df["agente"] == SEM_AGENTE]
    if len(sem_agente):
        st.warning(f"{len(sem_agente):,} licença(s) em {sem_agente['municipio_norm'].nunique():,} município(s) ainda sem agente.")

    st.divider()
    st.subheader("2) Roteiro de visitas por agente")
    agentes_com_dados = sorted(a for a in set(df["agente"]) if a != SEM_AGENTE)
    if not agentes_com_dados:
        st.info("Atribua ao menos um município a um agente (acima) para montar os roteiros.")
    else:
        r1, r2, r3 = st.columns(3)
        agente_sel = r1.selectbox("Agente", agentes_com_dados)
        dados_agente = df[df["agente"] == agente_sel]
        municipios_agente = dados_agente["municipio_norm"].value_counts().index.tolist()
        base_sel = r2.selectbox("Município-base (ponto de partida)", municipios_agente)
        por_dia = r3.number_input("Visitas por dia", min_value=1, max_value=50, value=8)
        rota, base_usada = montar_rota(dados_agente, base_sel, por_dia)
        if rota.empty:
            st.info("Nenhuma licença com coordenadas para este agente.")
        else:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Visitas", f"{len(rota):,}")
            k2.metric("Municípios", f"{rota['municipio_norm'].nunique():,}")
            k3.metric("Dias estimados", f"{int(rota['dia'].max()):,}")
            k4.metric("Km em linha reta", f"{rota['km_da_parada_anterior'].sum():,.0f}")
            st.caption("Ordem: municípios do mais próximo ao mais distante a partir da base; dentro de cada município, "
                       "sempre a próxima licença mais próxima. Distâncias em LINHA RETA — não consideram estradas.")
            rota_map = preparar_para_mapa(rota, cores)
            rota_map["agente"] = rota_map["agente"] + " · visita " + rota["ordem"].astype(str) + " · dia " + rota["dia"].astype(str)
            desenhar_mapa(rota_map, "Pontos", rota=rota)
            saida = rota_para_exportar(rota)
            st.dataframe(saida, hide_index=True, use_container_width=True, height=320)
            st.download_button(
                f"⬇️ Baixar roteiro de {agente_sel} (Excel)",
                data=excel_abas({agente_sel: saida},
                                pd.DataFrame([[agente_sel, len(rota), rota['municipio_norm'].nunique(),
                                               int(rota['dia'].max()), round(rota['km_da_parada_anterior'].sum(), 1)]],
                                             columns=["agente", "visitas", "municipios", "dias", "km_linha_reta"]),
                                meta_df(where, resultado["total"], brutas, len(saida))),
                file_name=f"roteiro_{re.sub(r'[^A-Za-z0-9]+', '_', agente_sel)}_{datetime.now():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

        st.divider()
        st.markdown("**Pacote com todos os agentes** (uma aba por agente; base = município com mais licenças de cada um)")
        if st.button("📦 Gerar pacote de todos os agentes"):
            with st.spinner("Montando roteiros..."):
                abas, resumo = {}, []
                for ag in agentes_com_dados:
                    g = df[df["agente"] == ag]
                    rt, base_ag = montar_rota(g, g["municipio_norm"].value_counts().index[0], por_dia)
                    if rt.empty:
                        continue
                    abas[ag] = rota_para_exportar(rt)
                    resumo.append([ag, base_ag, len(rt), rt["municipio_norm"].nunique(),
                                   int(rt["dia"].max()), round(rt["km_da_parada_anterior"].sum(), 1)])
                resumo_df = pd.DataFrame(resumo, columns=["agente", "base", "visitas", "municipios", "dias", "km_linha_reta"])
                st.session_state["pacote"] = excel_abas(abas, resumo_df, meta_df(where, resultado["total"], brutas, len(df)))
        if st.session_state.get("pacote"):
            st.download_button("⬇️ Baixar pacote (Excel)", data=st.session_state["pacote"],
                               file_name=f"roteiros_agentes_{datetime.now():%Y%m%d}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
