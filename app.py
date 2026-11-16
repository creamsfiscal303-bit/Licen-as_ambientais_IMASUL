"""
Fiscalização de PLANTIO FLORESTAL — IMASUL / CREA-MS (v4: mapa, agentes, rotas e conferência de ART)

Fonte: camada "licencas_ambientais" (layer 16) do IMASUL, que traz tipo de empreendimento,
detentor, CNPJ, imóvel, município, datas e a geometria dos imóveis.

Fluxo: filtros (lateral) -> baixar dados -> abas: Dados | Por município | Mapa | Conferência de ART | Agentes e rotas
"""
import hashlib
import io
import json
import re
import unicodedata
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
    "status_licenca", "situacao_art", "area_informada", "sicar", "lat", "lon", "km_da_parada_anterior", "link_google_maps",
]


# ----------------------------------------------------------------------------------------
# CONFERÊNCIA DE ART (Decisão CEA/MS 1333/2015)
# Unidade de conferência = "empreendimento": mesmo CPF/CNPJ no mesmo município, no ano de referência.
# O resultado é INDÍCIO para priorizar vistoria, nunca conclusão.
# ----------------------------------------------------------------------------------------
S_OK = "Com ART no ano"
S_POSS = "Possível ART (conferir)"
S_OUTRO_MUN = "ART em outro município"
S_OUTRO_ANO = "ART só em outros anos"
S_SEM = "Sem ART localizada"
S_NAO_VERIF = "Não verificável (sem CPF/CNPJ)"
S_NAO_CONF = "(não conferida)"
ORDEM_STATUS = [S_SEM, S_OUTRO_MUN, S_OUTRO_ANO, S_POSS, S_NAO_VERIF, S_OK, S_NAO_CONF]
CORES_STATUS = {
    S_OK: [46, 160, 67], S_POSS: [240, 200, 0], S_OUTRO_MUN: [245, 130, 48], S_OUTRO_ANO: [160, 90, 200],
    S_SEM: [215, 25, 28], S_NAO_VERIF: [110, 110, 110], S_NAO_CONF: [190, 190, 190],
}


def sem_acentos(txt):
    return "".join(c for c in unicodedata.normalize("NFD", str(txt)) if unicodedata.category(c) != "Mn")


def norm_texto(valor):
    """MAIÚSCULAS, sem acento e sem pontuação: 'Três Lagoas' -> 'TRES LAGOAS'."""
    if valor is None or pd.isna(valor):
        return ""
    return re.sub(r"[^A-Z0-9]+", " ", sem_acentos(valor).upper()).strip()


def norm_doc(valor):
    """Só os dígitos do CPF (11) ou CNPJ (14). Vazio se estiver mascarado ou incompleto."""
    if valor is None or pd.isna(valor):
        return ""
    t = str(valor).strip()
    if "*" in t:
        return ""
    d = re.sub(r"\D", "", re.sub(r"\.0+$", "", t))
    if len(d) in (9, 10):
        d = d.zfill(11)      # Excel costuma comer o zero à esquerda
    elif len(d) in (12, 13):
        d = d.zfill(14)
    return d if len(d) in (11, 14) else ""


SEM_MUN_KEY = norm_texto(SEM_MUNICIPIO)


def extrair_ano(serie):
    """Aceita datas (dd/mm/aaaa, aaaa-mm-dd...) ou uma coluna que já seja só o ano."""
    s = serie.astype(str).str.strip()
    ano = pd.to_numeric(s.where(s.str.fullmatch(r"(19|20)\d{2}")), errors="coerce")
    datas = pd.to_datetime(s.where(ano.isna()), dayfirst=True, errors="coerce", format="mixed")
    return ano.fillna(datas.dt.year)


@st.cache_data(show_spinner=False)
def ler_planilha(conteudo, nome):
    """Devolve {nome da aba: DataFrame}. Tudo como texto, para não perder zeros de CPF/CNPJ."""
    if nome.lower().endswith(".csv"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                return {"CSV": pd.read_csv(io.BytesIO(conteudo), sep=None, engine="python", dtype=str, encoding=enc)}
            except UnicodeDecodeError:
                continue
        raise ValueError("Não consegui ler o CSV (codificação desconhecida).")
    return pd.read_excel(io.BytesIO(conteudo), sheet_name=None, dtype=str)


def adivinhar_coluna(colunas, palavras):
    """Posição (começando em 1; 0 = nenhuma) da primeira coluna cujo nome contém alguma palavra."""
    normais = [norm_texto(c) for c in colunas]
    for p in palavras:
        for i, c in enumerate(normais):
            if norm_texto(p) in c:
                return i + 1
    return 0


def preparar_arts(tab, col_doc, col_nome, col_mun, col_data, col_ativ, incluir=(), excluir=(), col_num=None):
    a = pd.DataFrame(index=tab.index)
    a["doc"] = tab[col_doc].map(norm_doc) if col_doc else ""
    a["nome_norm"] = tab[col_nome].map(norm_texto) if col_nome else ""
    a["mun_key"] = tab[col_mun].map(norm_texto) if col_mun else ""
    a["ano"] = extrair_ano(tab[col_data]) if col_data else np.nan
    a["ativ"] = tab[col_ativ].map(norm_texto) if col_ativ else ""
    texto = lambda col: tab[col].fillna("").astype(str).str.strip() if col else ""
    a["num"], a["nome_orig"], a["mun_orig"] = texto(col_num), texto(col_nome), texto(col_mun)
    info = {"linhas_lidas": len(tab)}
    incluir = [norm_texto(x) for x in incluir if norm_texto(x)]
    excluir = [norm_texto(x) for x in excluir if norm_texto(x)]
    if col_ativ and incluir:
        ok = a["ativ"].apply(lambda t: any(p in t for p in incluir))
        info["descartadas_fora_do_filtro"] = int((~ok).sum())
        a = a[ok]
    if col_ativ and excluir:
        ruim = a["ativ"].apply(lambda t: any(p in t for p in excluir))
        info["descartadas_por_exclusao"] = int(ruim.sum())
        a = a[~ruim]
    info["arts_consideradas"] = len(a)
    info["sem_ano_valido"] = int(a["ano"].isna().sum())
    info["sem_cpf_cnpj_valido"] = int((a["doc"] == "").sum())
    return a, info


def montar_empreendimentos(df):
    """Agrupa as licenças por (CPF/CNPJ ou nome do detentor) + município."""
    t = df.copy()
    t["doc"] = t["cnpj"].map(norm_doc)
    t["nome_norm"] = t["nome_detentor"].map(norm_texto)
    t["mun_key"] = t["municipio_norm"].map(norm_texto)
    t["chave"] = np.where(t["doc"] != "", "DOC:" + t["doc"],
                          np.where(t["nome_norm"] != "", "NOME:" + t["nome_norm"], "OBJ:" + t["objectid"].astype(str)))
    primeiro_nome = lambda s: next((x for x in s if isinstance(x, str) and x.strip()), "")
    g = t.groupby(["chave", "mun_key"], sort=False).agg(
        doc=("doc", "first"), nome_norm=("nome_norm", "first"), detentor=("nome_detentor", primeiro_nome),
        municipio=("municipio_norm", "first"), agente=("agente", "first"), licencas=("objectid", "size"),
        area_informada_total=("area_informada", "sum"), tipo_empreendimento=("tipo_empreendimento", "first"),
        lat=("lat", "mean"), lon=("lon", "mean"), objectids=("objectid", list),
    ).reset_index().drop(columns="chave")
    g["area_informada_total"] = g["area_informada_total"].round(1)
    return g


def conferir_arts(emp, art, ano_ref, usar_raiz=False):
    """Para cada empreendimento devolve situação, observação e a ART encontrada (evidência para o agente)."""
    def kd(d):
        return d[:8] if (usar_raiz and len(d) == 14) else d

    art_ano = art[art["ano"] == ano_ref]
    doc_mun_ano, doc_ano, nome_mun_ano, nome_ano = {}, {}, {}, {}
    for r in art_ano.itertuples():
        ev = (r.num, r.nome_orig, r.doc, r.mun_orig)
        if r.doc:
            doc_mun_ano.setdefault((kd(r.doc), r.mun_key), ev)
            doc_ano.setdefault(kd(r.doc), ev)
        if r.nome_norm:
            nome_mun_ano.setdefault((r.nome_norm, r.mun_key), ev)
            nome_ano.setdefault(r.nome_norm, ev)
    doc_todos = {kd(d) for d in art["doc"] if d}
    vazio = ("", "", "", "")

    saida = []
    for doc, nome, mun in zip(emp["doc"], emp["nome_norm"], emp["mun_key"]):
        d = kd(doc) if doc else ""
        sem_mun = mun in ("", SEM_MUN_KEY)
        if d and not sem_mun and (d, mun) in doc_mun_ano:
            saida.append((S_OK, "CPF/CNPJ e município encontrados em ART do ano.", *doc_mun_ano[(d, mun)]))
        elif nome and not sem_mun and (nome, mun) in nome_mun_ano:
            saida.append((S_POSS, "ART do ano com mesmo nome e município; CPF/CNPJ ausente ou diferente. Conferir.",
                          *nome_mun_ano[(nome, mun)]))
        elif sem_mun and d and d in doc_ano:
            saida.append((S_POSS, "Há ART do ano para este CPF/CNPJ, mas a licença não informa o município.", *doc_ano[d]))
        elif sem_mun and nome and nome in nome_ano:
            saida.append((S_POSS, "Há ART do ano com este nome, mas a licença não informa o município.", *nome_ano[nome]))
        elif d and d in doc_ano:
            saida.append((S_OUTRO_MUN, "Há ART do ano com o mesmo CPF/CNPJ, mas em outro município "
                                       "(a Decisão 1333 pede uma ART por município).", *doc_ano[d]))
        elif d and d in doc_todos:
            saida.append((S_OUTRO_ANO, "Há ART com este CPF/CNPJ apenas em outros anos.", *vazio))
        elif d:
            saida.append((S_SEM, "Nenhuma ART com este CPF/CNPJ na planilha enviada.", *vazio))
        else:
            saida.append((S_NAO_VERIF, "A licença não informa CPF/CNPJ do detentor e nenhuma ART bate pelo nome.", *vazio))
    return pd.DataFrame(saida, index=emp.index, columns=[
        "situacao_art", "observacao", "art_numero", "art_contratante", "art_cpf_cnpj", "art_municipio"])


def executar_conferencia(df, art, ano_ref, usar_raiz=False):
    emp = montar_empreendimentos(df)
    res = conferir_arts(emp, art, ano_ref, usar_raiz)
    emp = pd.concat([emp, res], axis=1)
    por_objectid = {}
    for oids, situacao in zip(emp["objectids"], emp["situacao_art"]):
        for o in oids:
            por_objectid[o] = situacao
    return emp, por_objectid


def emp_para_exportar(emp, incluir_doc_art=True):
    t = emp.copy()
    t["situacao_art"] = pd.Categorical(t["situacao_art"], categories=ORDEM_STATUS, ordered=True)
    t = t.sort_values(["situacao_art", "licencas"], ascending=[True, False])
    t["lat"], t["lon"] = t["lat"].round(6), t["lon"].round(6)
    t["link_google_maps"] = [f"https://www.google.com/maps?q={a:.6f},{b:.6f}" if pd.notna(a) else ""
                             for a, b in zip(t["lat"], t["lon"])]
    t = t.rename(columns={"doc": "cpf_cnpj"})
    cols = ["situacao_art", "observacao", "art_numero", "art_contratante", "art_cpf_cnpj", "art_municipio",
            "cpf_cnpj", "detentor", "municipio", "agente", "licencas",
            "area_informada_total", "tipo_empreendimento", "lat", "lon", "link_google_maps"]
    if not incluir_doc_art:
        cols.remove("art_cpf_cnpj")
    return t[[c for c in cols if c in t.columns]].astype({"situacao_art": str})


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
MUNICIPIOS_MS = "Água Clara, Alcinópolis, Amambai, Anastácio, Anaurilândia, Angélica, Antônio João, Aparecida do Taboado, Aquidauana, Aral Moreira, Bandeirantes, Bataguassu, Batayporã, Bela Vista, Bodoquena, Bonito, Brasilândia, Caarapó, Camapuã, Campo Grande, Caracol, Cassilândia, Chapadão do Sul, Corguinho, Coronel Sapucaia, Corumbá, Costa Rica, Coxim, Deodápolis, Dois Irmãos do Buriti, Douradina, Dourados, Eldorado, Fátima do Sul, Figueirão, Glória de Dourados, Guia Lopes da Laguna, Iguatemi, Inocência, Itaporã, Itaquiraí, Ivinhema, Japorã, Jaraguari, Jardim, Jateí, Juti, Ladário, Laguna Carapã, Maracaju, Miranda, Mundo Novo, Naviraí, Nioaque, Nova Alvorada do Sul, Nova Andradina, Novo Horizonte do Sul, Paraíso das Águas, Paranaíba, Paranhos, Pedro Gomes, Ponta Porã, Porto Murtinho, Ribas do Rio Pardo, Rio Brilhante, Rio Negro, Rio Verde de Mato Grosso, Rochedo, Santa Rita do Pardo, São Gabriel do Oeste, Selvíria, Sete Quedas, Sidrolândia, Sonora, Tacuru, Taquarussu, Terenos, Três Lagoas, Vicentina".split(", ")   # os 79 municípios de MS (grafia oficial)


def carregar_atribuicoes():
    """{município normalizado: agente}. A comparação ignora acento, maiúscula e pontuação."""
    if not ARQ_AGENTES.exists():
        return {}
    try:
        t = pd.read_csv(ARQ_AGENTES, sep=";", dtype=str, encoding="utf-8-sig").fillna("")
        return {norm_texto(m): a.strip() for m, a in zip(t["municipio"], t["agente"]) if m.strip() and a.strip()}
    except Exception:
        return {}


def salvar_atribuicoes(mapa):
    """Grava o CSV com a grafia oficial dos municípios (quando reconhecida)."""
    nomes = {norm_texto(m): m for m in MUNICIPIOS_MS}
    linhas = sorted(((nomes.get(k, k), a) for k, a in mapa.items()), key=lambda x: norm_texto(x[0]))
    pd.DataFrame(linhas, columns=["municipio", "agente"]).to_csv(ARQ_AGENTES, sep=";", index=False, encoding="utf-8-sig")


def ler_csv_atribuicoes(arquivo):
    """Aceita CSV com colunas municipio e agente (UTF-8 ou Latin-1, separado por ; ou ,)."""
    conteudo = arquivo.getvalue() if hasattr(arquivo, "getvalue") else open(arquivo, "rb").read()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            t = pd.read_csv(io.BytesIO(conteudo), sep=None, engine="python", dtype=str, encoding=enc).fillna("")
            break
        except UnicodeDecodeError:
            continue
    t.columns = [norm_texto(c).lower() for c in t.columns]
    if "municipio" not in t.columns or "agente" not in t.columns:
        raise ValueError("O CSV precisa ter as colunas 'municipio' e 'agente'.")
    return {norm_texto(m): a.strip() for m, a in zip(t["municipio"], t["agente"]) if m.strip() and a.strip()}


def agente_do_municipio(serie_municipios, atrib):
    """Procura o agente de cada município (calcula uma vez por nome distinto)."""
    por_nome = {m: atrib.get(norm_texto(m)) for m in serie_municipios.dropna().unique()}
    return serie_municipios.map(por_nome)


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


def meta_df(where, total_servidor, linhas_brutas, linhas_arquivo, extras=None):
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
    ] + (extras or []), columns=["Item", "Informação"])


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


def preparar_para_mapa(df, cores, coluna="agente"):
    m = df.dropna(subset=["lat", "lon"]).copy()
    m["cor"] = m[coluna].map(lambda a: cores.get(a, COR_SEM_AGENTE))
    if "situacao_art" not in m.columns:
        m["situacao_art"] = ""
    for c in ["nome_detentor", "nome_imovel", "municipio_norm", "num_autorizacao", "status_licenca",
              "validade_licenca", "agente", "situacao_art"]:
        m[c] = m[c].fillna("").astype(str)
    m["tipo_curto"] = m["tipo_empreendimento"].fillna("").astype(str).str.slice(0, 70)
    m["expedicao"] = m["data_expedicao"].astype(str).replace("NaT", "")
    m["lat"], m["lon"] = m["lat"].astype(float), m["lon"].astype(float)
    return m[["lon", "lat", "cor", "agente", "municipio_norm", "nome_detentor", "nome_imovel",
              "num_autorizacao", "tipo_curto", "expedicao", "status_licenca", "validade_licenca", "situacao_art"]]


TOOLTIP = {
    "html": "<b>{municipio_norm}</b><br/>{nome_detentor}<br/>Imóvel: {nome_imovel}<br/>"
            "Licença: {num_autorizacao} ({status_licenca})<br/>{tipo_curto}<br/>"
            "Expedição: {expedicao} · {validade_licenca}<br/>Agente: {agente}<br/>ART: {situacao_art}",
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
df["agente"] = agente_do_municipio(df["municipio_norm"], atrib).fillna(SEM_AGENTE)
estado_art = st.session_state.get("art_conferencia")
df["situacao_art"] = df["objectid"].map(estado_art["por_objectid"] if estado_art else {}).fillna(S_NAO_CONF)
assinatura_df = hashlib.md5(",".join(map(str, sorted(df["objectid"]))).encode()).hexdigest()

sem_coord = int(df["lat"].isna().sum())
if sem_coord:
    st.warning(f"{sem_coord:,} registro(s) sem coordenadas: aparecem na tabela, mas não no mapa nem nas rotas.")
if resultado.get("modo") == "geometria":
    st.caption("ℹ️ O servidor não devolveu o ponto central; ele foi calculado a partir da forma simplificada do imóvel.")

aba_dados, aba_mun, aba_mapa, aba_art, aba_agentes = st.tabs(
    ["📋 Dados", "📊 Por município", "🗺️ Mapa", "🧾 Conferência de ART", "👮 Agentes e rotas"])

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
    por_mun["agente"] = agente_do_municipio(por_mun["municipio_norm"], atrib).fillna(SEM_AGENTE)
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
    cores_agente = cores_dos_agentes([a for a in agentes_existentes if a != SEM_AGENTE])
    cA, cB, cC = st.columns([1, 1, 2])
    modo_mapa = cA.radio("Visualização", ["Pontos", "Mapa de calor"], horizontal=False)
    opcoes_cor = ["Agente"] + (["Situação da ART"] if estado_art else [])
    colorir_por = cB.radio("Colorir por", opcoes_cor, horizontal=False)
    agentes_mapa = cC.multiselect("Agentes no mapa", agentes_existentes, default=agentes_existentes)
    if colorir_por == "Situação da ART":
        coluna_cor, cores_mapa = "situacao_art", CORES_STATUS
        presentes = [s for s in ORDEM_STATUS if s in set(df["situacao_art"])]
    else:
        coluna_cor, cores_mapa = "agente", cores_agente
        presentes = list(cores_agente)
    mapa_df = preparar_para_mapa(df[df["agente"].isin(agentes_mapa)], cores_mapa, coluna_cor)
    st.caption(f"{len(mapa_df):,} pontos. Cada ponto é o centro aproximado do imóvel licenciado.")
    desenhar_mapa(mapa_df, modo_mapa)
    st.markdown(" ".join(
        f"<span style='color:rgb({cores_mapa[a][0]},{cores_mapa[a][1]},{cores_mapa[a][2]})'>●</span> {a}"
        for a in presentes if a in cores_mapa
    ), unsafe_allow_html=True)

# ------------------------------------------------------------------ CONFERÊNCIA DE ART
with aba_art:
    st.subheader("Conferência de ART — Decisão CEA/MS nº 1333/2015")
    with st.expander("Regra aplicada e limites desta conferência"):
        st.markdown(
            "**O que a Decisão 1333/2015 determina (resumo):**\n"
            "- Uma ART individualizada **por contrato** de prestação de serviços na cultura florestal e **por CPF ou CNPJ**, "
            "desde que as áreas do contrato (contíguas ou não) estejam **no mesmo município e comarca**.\n"
            "- Essa ART pode abranger todas as atividades florestais: preparo de solo, plantio, tratos culturais, corte, "
            "carregamento e transporte.\n"
            "- **Viveiro de mudas** exige ART exclusiva; **inventário florestal e serviços ambientais** têm ARTs separadas.\n"
            "- Recolhimento de 1º a 31 de janeiro, com validade até 31 de dezembro do mesmo ano.\n"
            "- Vale para empresas do setor florestal, fornecedores de madeira e prestadores de serviço.\n\n"
            "**Como o app confere:** agrupa as licenças por CPF/CNPJ do detentor + município (um *empreendimento*) "
            "e procura, na planilha, ART do **ano de referência** com o mesmo CPF/CNPJ e município.\n\n"
            "**Limites:** (1) a licença do IMASUL traz CNPJ, mas **não traz CPF** — produtor pessoa física só é "
            "conferido por nome; (2) o detentor da licença nem sempre é quem executa a obra (ex.: dono da terra que "
            "arrenda para empresa florestal), e a decisão trata de quem executa; (3) nomes, CNPJ de matriz/filial e "
            "grafia de município podem divergir. Por isso o resultado é **indício para priorizar vistoria**, nunca conclusão."
        )
    arq_art = st.file_uploader("Planilha de ARTs (CSV ou XLSX)", type=["csv", "xlsx"], key="upload_art")
    if arq_art is None:
        st.info("Envie a planilha de ARTs exportada do sistema do CREA-MS. Depois você indica qual coluna é qual.")
    else:
        try:
            folhas = ler_planilha(arq_art.getvalue(), arq_art.name)
        except Exception as exc:
            folhas = {}
            st.error(f"Não foi possível ler a planilha: {exc}")
        if folhas:
            nome_folha = st.selectbox("Aba da planilha", list(folhas)) if len(folhas) > 1 else next(iter(folhas))
            tab_art = folhas[nome_folha]
            st.caption(f"{len(tab_art):,} linhas · {len(tab_art.columns)} colunas")
            st.dataframe(tab_art.head(5), use_container_width=True, hide_index=True)
            colunas = list(tab_art.columns)
            nenhuma = "(nenhuma)"

            def seletor(rotulo, palavras, ajuda=None):
                return st.selectbox(rotulo, [nenhuma] + colunas, index=adivinhar_coluna(colunas, palavras), help=ajuda)

            st.markdown("**Quais colunas da planilha são estas?** (o app tenta adivinhar; confira)")
            k1, k2, k3 = st.columns(3)
            with k1:
                c_doc = seletor("CPF/CNPJ do contratante", ["CPF CNPJ", "CNPJ", "CPF", "DOCUMENTO"],
                                "Principal chave de comparação. Se estiver mascarado (***), não serve.")
                c_nome = seletor("Nome do contratante", ["NOME CONTRATANTE", "NOME", "RAZAO", "CONTRATANTE"])
            with k2:
                c_mun = seletor("Município da obra/serviço", ["MUNICIPIO OBRA", "MUNICIPIO", "CIDADE"])
                c_data = seletor("Data (ou ano) da ART", ["DATA REGISTRO", "DATA DE REGISTRO", "DATA", "ANO"],
                                 "Usada para saber o ano da ART.")
            with k3:
                c_ativ = seletor("Atividade / descrição (opcional)", ["ATIVIDADE", "DESCRICAO", "OBRA SERVICO"],
                                 "Permite descartar ARTs de viveiro ou inventário e manter só as florestais.")
                c_num = seletor("Número da ART (opcional)", ["NUMERO ART", "N ART", "NUM ART", "ART"],
                                "Aparece no resultado para o agente conferir a ART encontrada.")
            p1, p2, p3 = st.columns(3)
            ano_ref = p1.selectbox("Ano de referência", list(range(ANO_MAX, 2014, -1)))
            usar_raiz = p2.checkbox("Aceitar matriz/filial (8 primeiros dígitos do CNPJ)", value=False,
                                    help="Desligado = segue a decisão ao pé da letra (CNPJ próprio e individual).")
            txt_incluir = p3.text_input("Só considerar ARTs cuja atividade contenha (vírgula; vazio = todas)", value="",
                                        disabled=(c_ativ == nenhuma))
            txt_excluir = st.text_input("Desconsiderar ARTs cuja atividade contenha (vírgula)", value="VIVEIRO, INVENTARIO",
                                        disabled=(c_ativ == nenhuma),
                                        help="Pela Decisão 1333, viveiro e inventário têm ART própria e não cobrem a produção florestal.")
            if c_ativ != nenhuma:
                with st.expander("Ver valores mais comuns da coluna de atividade (ajuda a escolher as palavras)"):
                    st.dataframe(tab_art[c_ativ].fillna("").value_counts().head(25).rename("ARTs").reset_index(),
                                 hide_index=True, use_container_width=True)
            faltando = [n for n, c in [("CPF/CNPJ", c_doc), ("município", c_mun), ("data/ano", c_data)] if c == nenhuma]
            if faltando:
                st.warning("Sem " + ", ".join(faltando) + " a conferência fica fraca ou impossível. Indique essas colunas se existirem.")
            if st.button("🔍 Conferir ARTs", type="primary", disabled=(c_data == nenhuma or (c_doc == nenhuma and c_nome == nenhuma))):
                art_prep, info_art = preparar_arts(
                    tab_art, None if c_doc == nenhuma else c_doc, None if c_nome == nenhuma else c_nome,
                    None if c_mun == nenhuma else c_mun, None if c_data == nenhuma else c_data,
                    None if c_ativ == nenhuma else c_ativ,
                    [x.strip() for x in txt_incluir.split(",") if x.strip()],
                    [x.strip() for x in txt_excluir.split(",") if x.strip()],
                    col_num=None if c_num == nenhuma else c_num,
                )
                emp_res, por_objectid = executar_conferencia(df, art_prep, ano_ref, usar_raiz)
                st.session_state["art_conferencia"] = {
                    "emp": emp_res, "por_objectid": por_objectid, "ano": ano_ref, "arquivo": arq_art.name,
                    "info": info_art, "assinatura": assinatura_df, "raiz": usar_raiz,
                }
                st.rerun()

    estado = st.session_state.get("art_conferencia")
    if estado:
        st.divider()
        st.markdown(f"### Resultado — ART de {estado['ano']} · arquivo `{estado['arquivo']}`")
        if estado["assinatura"] != assinatura_df:
            st.warning("Os filtros mudaram desde esta conferência. Clique em 'Conferir ARTs' de novo para atualizar.")
        i = estado["info"]
        st.caption(f"{i['linhas_lidas']:,} linhas lidas · {i['arts_consideradas']:,} ARTs consideradas · "
                   f"{i['sem_ano_valido']:,} sem ano válido · {i['sem_cpf_cnpj_valido']:,} sem CPF/CNPJ válido"
                   + (f" · {i['descartadas_por_exclusao']:,} descartadas (viveiro/inventário)" if 'descartadas_por_exclusao' in i else ""))
        emp = estado["emp"]
        resumo = (emp.groupby("situacao_art").agg(empreendimentos=("licencas", "size"), licencas=("licencas", "sum"))
                     .reindex([s for s in ORDEM_STATUS if s in set(emp["situacao_art"])]).reset_index()
                     .rename(columns={"situacao_art": "situação"}))
        r1, r2 = st.columns([1, 2])
        r1.dataframe(resumo, hide_index=True, use_container_width=True)
        cruz = pd.crosstab(emp["agente"], emp["situacao_art"]).reindex(columns=[s for s in ORDEM_STATUS if s in set(emp["situacao_art"])])
        r2.markdown("**Empreendimentos por agente e situação**")
        r2.dataframe(cruz, use_container_width=True)
        f1, f2 = st.columns(2)
        sit_ver = f1.multiselect("Situação", [s for s in ORDEM_STATUS if s in set(emp["situacao_art"])],
                                 default=[s for s in (S_SEM, S_OUTRO_MUN, S_OUTRO_ANO, S_POSS) if s in set(emp["situacao_art"])])
        ag_ver = f2.multiselect("Agente", sorted(set(emp["agente"])), default=sorted(set(emp["agente"])))
        incl_doc = st.checkbox("Mostrar CPF/CNPJ da ART encontrada no resultado e no Excel", value=True,
                               help="Desmarque se o arquivo for circular fora da fiscalização (dado pessoal, LGPD).")
        tabela_emp = emp_para_exportar(emp, incl_doc)
        tabela_emp = tabela_emp[tabela_emp["situacao_art"].isin(sit_ver) & tabela_emp["agente"].isin(ag_ver)]
        st.dataframe(tabela_emp, hide_index=True, use_container_width=True, height=380)
        lic_export = df.drop(columns=["municipio_norm"])
        extras = [["Conferência de ART", f"Decisão CEA/MS 1333/2015 · ano {estado['ano']} · arquivo {estado['arquivo']} · "
                                          f"matriz/filial {'aceita' if estado['raiz'] else 'não aceita'}"],
                  ["Aviso", "Situação de ART é INDÍCIO para priorizar vistoria. Confirmar no sistema do CREA antes de qualquer ato."]]
        st.download_button(
            "⬇️ Baixar conferência (Excel)",
            data=excel_abas({"Empreendimentos": emp_para_exportar(emp, incl_doc), "Licencas": lic_export}, resumo,
                            meta_df(where, resultado["total"], brutas, len(lic_export), extras)),
            file_name=f"conferencia_art_{estado['ano']}_{datetime.now():%Y%m%d_%H%M}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        st.caption("Dica: na aba 'Mapa' escolha 'Colorir por: Situação da ART' e, em 'Agentes e rotas', filtre o roteiro por situação.")

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
    editor_base["agente"] = agente_do_municipio(editor_base["municipio_norm"], atrib).fillna("")
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
                novo_mapa[norm_texto(mun)] = str(ag).strip()
            else:
                novo_mapa.pop(norm_texto(mun), None)
        salvar_atribuicoes(novo_mapa)
        st.success("Atribuição salva.")
        st.rerun()

    canon = {norm_texto(m) for m in MUNICIPIOS_MS}
    fora_da_lista = sorted(m for m in df["municipio_norm"].unique() if m != SEM_MUNICIPIO and norm_texto(m) not in canon)
    if fora_da_lista:
        st.warning("Municípios do IMASUL com grafia que não bate com a lista oficial de MS (não receberão agente): "
                   + ", ".join(fora_da_lista[:15]))
    sem_agente = df[df["agente"] == SEM_AGENTE]
    if len(sem_agente):
        st.warning(f"{len(sem_agente):,} licença(s) em {sem_agente['municipio_norm'].nunique():,} município(s) ainda sem agente.")

    st.divider()
    st.subheader("2) Roteiro de visitas por agente")
    df_rota = df
    if estado_art:
        presentes_art = [s for s in ORDEM_STATUS if s in set(df["situacao_art"])]
        padrao_art = [s for s in (S_SEM, S_OUTRO_MUN, S_OUTRO_ANO, S_POSS) if s in presentes_art] or presentes_art
        sit_rota = st.multiselect("Roteiro só com licenças nesta situação de ART", presentes_art, default=padrao_art,
                                  help="Vem da aba 'Conferência de ART'. Deixe todas marcadas para ignorar o filtro.")
        df_rota = df[df["situacao_art"].isin(sit_rota)]
    agentes_com_dados = sorted(a for a in set(df_rota["agente"]) if a != SEM_AGENTE)
    if not agentes_com_dados:
        st.info("Atribua ao menos um município a um agente (acima) para montar os roteiros.")
    else:
        r1, r2, r3 = st.columns(3)
        agente_sel = r1.selectbox("Agente", agentes_com_dados)
        dados_agente = df_rota[df_rota["agente"] == agente_sel]
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
            rota_map = preparar_para_mapa(rota, cores_agente)
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
                    g = df_rota[df_rota["agente"] == ag]
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
