"""
Extrator de licenças de PLANTIO FLORESTAL — IMASUL / CREA-MS (versão simplificada)

Fonte única: camada "licencas_ambientais" (layer 16), que já traz tipo de empreendimento,
detentor, CNPJ, imóvel, município e data de expedição. Não precisa cruzar com outra camada.
"""
import io
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Licenças de Plantio Florestal | IMASUL", page_icon="🌱", layout="wide")

# ----------------------------------------------------------------------------------------
# CONFIGURAÇÕES (se o IMASUL mudar algo, é aqui que você mexe)
# ----------------------------------------------------------------------------------------
LAYER_URL = "https://www.pinms.ms.gov.br/arcgis/rest/services/IMASUL/licencas_ambientais/FeatureServer/16"
QUERY_URL = f"{LAYER_URL}/query"
PAGE_SIZE = 2000          # limite do servidor (maxRecordCount)
TIMEOUT = (20, 180)       # (segundos para conectar, segundos para receber)
FUSO = "America/Campo_Grande"
ANO_MIN = 2012            # primeira data de expedição existente na camada
ANO_MAX = datetime.now().year
STATUS_POSSIVEIS = ["Emitida", "Cancelada", "Suspensa", "Gerada"]
PALAVRAS_PADRAO = ("PLANTIO DE FLORESTA", "REFLORESTAMENTO")  # pré-seleção dos tipos

# Campos baixados (sem geometria, para ser rápido)
CAMPOS = [
    "objectid", "num_autorizacao", "num_processo", "id_empreendimento",
    "data_expedicao", "data_validade", "validade_licenca", "status_licenca", "tipo_licenca",
    "id_tipo_empreendimento", "tipo_empreendimento",
    "nome_detentor", "cnpj", "nome_imovel", "municipio", "cod_mun",
    "sicar", "car", "car_id", "area_informada", "situacao_cadastro", "coordenadas_imovel",
]
COLUNAS_DATA = ["data_expedicao", "data_validade"]


# ----------------------------------------------------------------------------------------
# COMUNICAÇÃO COM O SERVIDOR ARCGIS
# ----------------------------------------------------------------------------------------
class ErroArcGIS(Exception):
    """Erro devolvido pelo próprio servidor (ex.: consulta inválida). Não adianta tentar de novo."""


def arcgis_query(params, tentativas=4):
    """Faz a consulta usando POST (aceita consultas longas) e tenta de novo se a rede falhar."""
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
    """Coloca aspas e protege apóstrofos: O'Brien -> 'O''Brien'."""
    return "'" + str(valor).replace("'", "''") + "'"


@st.cache_data(ttl=3600, show_spinner=False)
def listar_tipos():
    """Pede ao servidor a lista de tipos com a contagem de cada um (agrupamento no servidor)."""
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


def baixar_tudo(where, total, ao_progredir=None):
    """Baixa os registros em páginas de 2000 (resultOffset), em ordem estável."""
    linhas, offset = [], 0
    for _ in range(10_000):  # trava de segurança contra loop infinito
        dados = arcgis_query({
            "where": where,
            "outFields": ",".join(CAMPOS),
            "returnGeometry": "false",
            "orderByFields": "objectid ASC",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
        })
        feicoes = dados.get("features", [])
        linhas.extend(f["attributes"] for f in feicoes)
        offset += len(feicoes)
        if ao_progredir:
            ao_progredir(len(linhas), total)
        if not feicoes or (len(feicoes) < PAGE_SIZE and not dados.get("exceededTransferLimit")):
            break
    return linhas


# ----------------------------------------------------------------------------------------
# FILTROS E TRATAMENTO DOS DADOS
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


def preparar_dataframe(linhas):
    df = pd.DataFrame(linhas, columns=CAMPOS)
    for col in COLUNAS_DATA:  # milissegundos -> data local de MS
        ms = pd.to_numeric(df[col], errors="coerce")
        df[col] = pd.to_datetime(ms, unit="ms", utc=True).dt.tz_convert(FUSO).dt.date
    ano = pd.to_datetime(df["data_expedicao"], errors="coerce").dt.year.astype("Int64")
    df.insert(df.columns.get_loc("data_expedicao") + 1, "ano_expedicao", ano)
    return df


def remover_duplicadas(df):
    """Uma licença pode aparecer em vários polígonos. Mantém uma linha por licença/imóvel."""
    chave = [c for c in ["num_autorizacao", "num_processo", "id_empreendimento",
                         "tipo_empreendimento", "sicar", "car"] if c in df.columns]
    return df.drop_duplicates(subset=chave, keep="first")


@st.cache_data(show_spinner="Gerando Excel...")
def gerar_excel(df, where, total_servidor, linhas_brutas):
    saida = io.BytesIO()
    meta = pd.DataFrame([
        ["Fonte", "IMASUL - camada licencas_ambientais (imóveis rurais)"],
        ["Serviço ArcGIS REST", LAYER_URL],
        ["Consulta aplicada", where],
        ["Data/hora da extração (UTC)", datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M:%S")],
        ["Registros no servidor para esta consulta", total_servidor],
        ["Linhas baixadas", linhas_brutas],
        ["Linhas no arquivo (após filtros locais)", len(df)],
        ["Observação", "Dados públicos do serviço consultado. Confirmar na fonte oficial antes de uso em procedimento fiscalizatório."],
    ], columns=["Item", "Informação"])
    with pd.ExcelWriter(saida, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Licencas")
        meta.to_excel(w, index=False, sheet_name="Metadados")
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                largura = min(max((len(str(c.value or "")) for c in list(col)[:300]), default=10) + 2, 45)
                ws.column_dimensions[col[0].column_letter].width = max(12, largura)
        ws = w.book["Licencas"]
        for cel_cabecalho in ws[1]:
            if cel_cabecalho.value in COLUNAS_DATA:
                for linha in ws.iter_rows(min_row=2, min_col=cel_cabecalho.column, max_col=cel_cabecalho.column):
                    linha[0].number_format = "DD/MM/YYYY"
    return saida.getvalue()


# ----------------------------------------------------------------------------------------
# TELA
# ----------------------------------------------------------------------------------------
st.title("🌱 Licenças de Plantio Florestal — IMASUL")
st.caption("CREA-MS · Dados abertos do IMASUL (ArcGIS REST) · Exportação XLSX/CSV")
st.warning("Esta camada não necessariamente representa todas as licenças concedidas pelo IMASUL. "
           "Valide os dados na fonte oficial antes de qualquer uso fiscalizatório.")

try:
    tipos_disponiveis = listar_tipos()
except Exception as exc:
    st.error(f"Não foi possível consultar o IMASUL agora: {exc}")
    st.stop()

with st.sidebar:
    st.header("Filtros")
    padrao = [t for t in tipos_disponiveis if any(p in t.upper() for p in PALAVRAS_PADRAO)]
    tipos_escolhidos = st.multiselect(
        "Tipo de empreendimento", options=list(tipos_disponiveis), default=padrao,
        format_func=lambda t: f"{t[:90]}{'…' if len(t) > 90 else ''}  ({tipos_disponiveis[t]:,})",
        help="Já vem marcado: plantio de floresta e reflorestamento. "
             "Adicione 'corte ou extração em floresta plantada' se quiser a colheita também.",
    )
    ano_ini, ano_fim = st.slider("Ano de expedição", ANO_MIN, ANO_MAX, (ANO_MIN, ANO_MAX))
    status = st.multiselect("Situação da licença", STATUS_POSSIVEIS, default=STATUS_POSSIVEIS)
    unica = st.checkbox("Uma linha por licença (remove polígonos repetidos)", value=True,
                        help="Se o mesmo número de licença cobrir imóveis sem SICAR/CAR preenchido, "
                             "eles serão tratados como um só.")

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

if st.button("⬇️ Baixar dados do IMASUL", type="primary", use_container_width=True):
    barra = st.progress(0.0, text="Iniciando...")
    try:
        linhas = baixar_tudo(
            where, total,
            lambda feitos, tot: barra.progress(min(1.0, feitos / max(tot, 1)), text=f"{feitos:,} de {tot:,} registros"),
        )
        st.session_state["resultado"] = {"where": where, "linhas": linhas, "total": total}
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
if unica:
    df = remover_duplicadas(df)

busca = st.text_input("🔎 Buscar por texto em detentor, imóvel ou município (opcional)")
if busca.strip():
    mascara = pd.Series(False, index=df.index)
    for c in ["nome_detentor", "nome_imovel", "municipio"]:
        mascara |= df[c].fillna("").astype(str).str.contains(busca.strip(), case=False, regex=False)
    df = df[mascara]

st.subheader("Resultado")
m1, m2, m3 = st.columns(3)
m1.metric("Linhas baixadas", f"{brutas:,}")
m2.metric("Linhas na tabela", f"{len(df):,}")
m3.metric("Licenças distintas (nº autorização)", f"{df['num_autorizacao'].nunique():,}")

if len(df):
    st.markdown("**Preenchimento dos dados de identificação** (licenças antigas costumam vir em branco)")
    campos_id = ["nome_detentor", "cnpj", "nome_imovel", "municipio", "sicar"]
    qualidade = pd.DataFrame({
        "campo": campos_id,
        "% preenchido": [round(100 * (~esta_vazio(df[c])).mean(), 1) for c in campos_id],
    })
    st.dataframe(qualidade, hide_index=True, use_container_width=True)
    st.markdown("**Licenças por ano de expedição**")
    st.bar_chart(df["ano_expedicao"].value_counts().sort_index())

st.dataframe(df.head(1000), use_container_width=True, hide_index=True, height=420)
if len(df) > 1000:
    st.caption("Prévia limitada a 1.000 linhas. Os arquivos para download têm todas.")

st.download_button(
    "⬇️ Baixar Excel (.xlsx)",
    data=gerar_excel(df, where, resultado["total"], brutas),
    file_name=f"imasul_plantio_{datetime.now():%Y%m%d_%H%M}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
st.download_button(
    "⬇️ Baixar CSV",
    data=df.to_csv(index=False, sep=";", encoding="utf-8-sig"),
    file_name=f"imasul_plantio_{datetime.now():%Y%m%d_%H%M}.csv",
    mime="text/csv",
    use_container_width=True,
)
