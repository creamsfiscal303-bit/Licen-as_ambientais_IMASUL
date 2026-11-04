import hashlib
import io
import json
import re
import sqlite3
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Extrator IMASUL | CREA-MS", page_icon="🌱", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "dados_imasul"
DATA_DIR.mkdir(exist_ok=True)
TIMEOUT = (20, 120)
DEFAULT_BATCH = 100

SOURCES = {
    "Licenças ambientais - GLA (Ponto)": {
        "key": "gla",
        "url": "https://www.pinms.ms.gov.br/arcgis/rest/services/IMASUL/gla_licencas_p/FeatureServer/0",
    },
    "Licenças ambientais - imóveis rurais (Polígono)": {
        "key": "rural",
        "url": "https://www.pinms.ms.gov.br/arcgis/rest/services/IMASUL/licencas_ambientais/FeatureServer/16",
    },
    "Plantio florestal - Consulta Classes EFlorestal (Polígono)": {
        "key": "eflorestal",
        "url": "https://www.pinms.ms.gov.br/arcgis/rest/services/IMASUL/API_Monitoramento_Imasul_V2/FeatureServer/1",
    },
}

def arcgis_get(url, params, attempts=4):
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                err = payload["error"]
                raise RuntimeError(f"ArcGIS: {err.get('message', 'erro na consulta')} | detalhes: {err.get('details', [])}")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(str(last_error))

@st.cache_data(ttl=300, show_spinner=False)
def get_metadata(layer_url):
    return arcgis_get(layer_url, {"f": "json"})

def field_options(metadata):
    result = []
    for f in metadata.get("fields", []):
        name = f.get("name")
        alias = f.get("alias") or name
        typ = f.get("type", "")
        if name and name.lower() not in {"shape", "shape__area", "shape__length"}:
            result.append({"name": name, "alias": alias, "type": typ})
    return result

def detect_emission_field(fields):
    preferred = ["datalicenc", "data_expedicao", "data_emissao", "dataemissao", "data_licenca",
                 "data_licenc", "dataexpedicao", "dt_emissao", "dt_licenca"]
    by_name = {f["name"].lower(): f for f in fields}
    for name in preferred:
        if name in by_name and by_name[name]["type"] == "esriFieldTypeDate":
            return by_name[name]["name"]
    for f in fields:
        s = (f["name"] + " " + f["alias"]).lower()
        if f["type"] == "esriFieldTypeDate" and any(x in s for x in ["emiss", "expedi", "licenc", "licença"]):
            if not any(x in s for x in ["valid", "process", "abert", "cadastro"]):
                return f["name"]
    return None

def make_date_where(date_field, period_mode, year, month):
    if not date_field or period_mode == "Todos os períodos":
        return "1=1"
    start = date(int(year), 1 if period_mode == "Ano" else int(month), 1)
    if period_mode == "Ano":
        end = date(int(year) + 1, 1, 1)
    else:
        end = date(int(year) + (1 if int(month) == 12 else 0),
                   1 if int(month) == 12 else int(month) + 1, 1)
    return f"{date_field} >= DATE '{start.isoformat()}' AND {date_field} < DATE '{end.isoformat()}'"

def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"

@st.cache_data(ttl=300, show_spinner=False)
def get_distinct_values(layer_url, field_name, base_where):
    """Obtém os valores distintos no servidor, respeitando o filtro temporal."""
    endpoint = f"{layer_url}/query"
    params = {
        "where": base_where,
        "outFields": field_name,
        "returnGeometry": "false",
        "returnDistinctValues": "true",
        "returnIdsOnly": "false",
        "f": "json",
        "resultRecordCount": 2000,
    }
    try:
        payload = arcgis_get(endpoint, params)
        values = set()
        for feature in payload.get("features", []):
            value = feature.get("attributes", {}).get(field_name)
            if value is not None and str(value).strip():
                values.add(str(value).strip())
        # Se a consulta distinta foi concluída, usa os valores retornados.
        if values or payload.get("features") == []:
            return sorted(values, key=lambda x: x.casefold())
    except Exception:
        pass

    # Alternativa para serviços ArcGIS que falham com returnDistinctValues:
    # baixa IDs e consulta o campo em blocos, deduplicando localmente.
    ids_payload = arcgis_get(endpoint, {
        "where": base_where, "returnIdsOnly": "true",
        "returnGeometry": "false", "f": "json"
    })
    object_ids = ids_payload.get("objectIds") or []
    values = set()
    for start in range(0, len(object_ids), 500):
        block = object_ids[start:start + 500]
        block_payload = arcgis_get(endpoint, {
            "objectIds": ",".join(map(str, block)),
            "outFields": field_name, "returnGeometry": "false", "f": "json"
        })
        for feature in block_payload.get("features", []):
            value = feature.get("attributes", {}).get(field_name)
            if value is not None and str(value).strip():
                values.add(str(value).strip())
    return sorted(values, key=lambda x: x.casefold())

def make_full_where(date_where, type_field, selected_type):
    parts = []
    if date_where and date_where != "1=1":
        parts.append(f"({date_where})")
    if type_field and selected_type and selected_type != "Todos":
        parts.append(f"{type_field} = {sql_string(selected_type)}")
    return " AND ".join(parts) if parts else "1=1"

def job_signature(source_key, where):
    return hashlib.sha256(f"{source_key}|{where}".encode("utf-8")).hexdigest()[:20]

def db_connect(path):
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS job_meta (key TEXT PRIMARY KEY, value TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS pending_ids (objectid INTEGER PRIMARY KEY, done INTEGER NOT NULL DEFAULT 0)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS records (objectid INTEGER PRIMARY KEY, attributes_json TEXT NOT NULL)""")
    conn.commit()
    return conn

def set_meta(conn, key, value):
    conn.execute("INSERT INTO job_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, str(value)))
    conn.commit()

def get_meta(conn, key, default=""):
    row = conn.execute("SELECT value FROM job_meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default

def initialize_job(conn, layer_url, where, source_name):
    if get_meta(conn, "ids_loaded") == "1":
        return
    payload = arcgis_get(f"{layer_url}/query", {
        "where": where, "returnIdsOnly": "true", "returnGeometry": "false", "f": "json"
    })
    ids = payload.get("objectIds") or []
    for start in range(0, len(ids), 5000):
        conn.executemany("INSERT OR IGNORE INTO pending_ids(objectid,done) VALUES(?,0)",
                         [(int(x),) for x in ids[start:start+5000]])
        conn.commit()
    set_meta(conn, "ids_loaded", "1")
    set_meta(conn, "total_ids", len(ids))
    set_meta(conn, "source_name", source_name)
    set_meta(conn, "where", where)
    set_meta(conn, "created_at_utc", datetime.now(timezone.utc).isoformat())

def pending_count(conn):
    return conn.execute("SELECT COUNT(*) FROM pending_ids WHERE done=0").fetchone()[0]

def completed_count(conn):
    return conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

def extract_batch(conn, layer_url, batch_size):
    rowids = [r[0] for r in conn.execute(
        "SELECT objectid FROM pending_ids WHERE done=0 ORDER BY objectid LIMIT ?", (batch_size,)
    ).fetchall()]
    if not rowids:
        return 0

    def fetch(ids):
        payload = arcgis_get(f"{layer_url}/query", {
            "objectIds": ",".join(map(str, ids)),
            "outFields": "*",
            "returnGeometry": "false",
            "f": "json",
        }, attempts=3)
        features = payload.get("features", [])
        result = []
        for feature in features:
            attrs = feature.get("attributes", {})
            oid = attrs.get("OBJECTID", attrs.get("objectid"))
            if oid is None:
                # Fall back to the corresponding requested ID only when one feature is returned.
                if len(ids) == 1:
                    oid = ids[0]
                else:
                    continue
            result.append((int(oid), attrs))
        return result

    try:
        results = fetch(rowids)
    except Exception:
        if len(rowids) > 1:
            mid = len(rowids) // 2
            return extract_specific_ids(conn, layer_url, rowids[:mid]) + extract_specific_ids(conn, layer_url, rowids[mid:])
        raise

    with conn:
        for oid, attrs in results:
            conn.execute("INSERT OR REPLACE INTO records(objectid,attributes_json) VALUES(?,?)",
                         (oid, json.dumps(attrs, ensure_ascii=False)))
            conn.execute("UPDATE pending_ids SET done=1 WHERE objectid=?", (oid,))
    returned = {oid for oid, _ in results}
    missing = [oid for oid in rowids if oid not in returned]
    if missing:
        with conn:
            for oid in missing:
                conn.execute("UPDATE pending_ids SET done=2 WHERE objectid=?", (oid,))
    return len(results)

def extract_specific_ids(conn, layer_url, ids):
    if not ids:
        return 0
    try:
        payload = arcgis_get(f"{layer_url}/query", {
            "objectIds": ",".join(map(str, ids)), "outFields": "*",
            "returnGeometry": "false", "f": "json"
        }, attempts=3)
        features = payload.get("features", [])
        got = 0
        with conn:
            returned = set()
            for ft in features:
                attrs = ft.get("attributes", {})
                oid = attrs.get("OBJECTID", attrs.get("objectid"))
                if oid is None and len(ids) == 1:
                    oid = ids[0]
                if oid is not None:
                    oid = int(oid)
                    returned.add(oid)
                    conn.execute("INSERT OR REPLACE INTO records(objectid,attributes_json) VALUES(?,?)",
                                 (oid, json.dumps(attrs, ensure_ascii=False)))
                    conn.execute("UPDATE pending_ids SET done=1 WHERE objectid=?", (oid,))
                    got += 1
        missing = [x for x in ids if x not in returned]
        if missing and len(ids) > 1:
            mid = len(missing) // 2
            got += extract_specific_ids(conn, layer_url, missing[:mid])
            got += extract_specific_ids(conn, layer_url, missing[mid:])
        elif missing:
            # Keep single ID pending so a later resume can retry it.
            pass
        return got
    except Exception:
        if len(ids) > 1:
            mid = len(ids) // 2
            return extract_specific_ids(conn, layer_url, ids[:mid]) + extract_specific_ids(conn, layer_url, ids[mid:])
        raise

def load_dataframe(conn):
    rows = conn.execute("SELECT attributes_json FROM records ORDER BY objectid").fetchall()
    records = [json.loads(r[0]) for r in rows]
    df = pd.DataFrame(records)
    if not df.empty:
        for col in df.columns:
            if col.lower().startswith("data") or any(x in col.lower() for x in ["emissao", "expedicao", "validade", "licenc"]):
                numeric = pd.to_numeric(df[col], errors="coerce")
                valid = numeric.dropna()
                if len(valid) and valid.between(100000000000, 3000000000000).mean() > 0.8:
                    df[col] = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True).dt.strftime("%d/%m/%Y")
    return df


LICENSE_LAYER_URL = SOURCES["Licenças ambientais - imóveis rurais (Polígono)"]["url"]

def norm_key(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())

def query_license_rows(where, out_fields):
    """Consulta registros da camada de licenças usando paginação por IDs."""
    endpoint = f"{LICENSE_LAYER_URL}/query"
    ids_payload = arcgis_get(endpoint, {
        "where": where, "returnIdsOnly": "true", "returnGeometry": "false", "f": "json"
    })
    ids = ids_payload.get("objectIds") or []
    rows = []
    for start in range(0, len(ids), 300):
        block = ids[start:start + 300]
        payload = arcgis_get(endpoint, {
            "objectIds": ",".join(map(str, block)),
            "outFields": ",".join(out_fields),
            "returnGeometry": "false", "f": "json"
        })
        rows.extend(ft.get("attributes", {}) for ft in payload.get("features", []))
    return rows

def enrich_planting_dataframe(plant_df):
    """Enriquece plantios com dados da camada de licenças, sem inferência por nome."""
    if plant_df.empty:
        return plant_df.copy(), {"matched": 0, "unmatched": 0, "multiple": 0, "cpf_field": False}

    lic_meta = get_metadata(LICENSE_LAYER_URL)
    lic_fields = {f.get("name", "").lower(): f.get("name") for f in lic_meta.get("fields", [])}
    wanted_aliases = ["objectid", "id_empreendimento", "num_processo", "numero_processo",
                      "nome_detentor", "cnpj", "cpf", "nome_imovel", "municipio",
                      "data_expedicao", "validade_licenca", "status_licenca",
                      "tipo_empreendimento", "tipo_licenca", "area_informada", "car", "sicar"]
    out_fields = [lic_fields[x] for x in wanted_aliases if x in lic_fields]
    if "objectid" not in lic_fields:
        out_fields.insert(0, "OBJECTID")
    oid_field = lic_fields.get("objectid", "OBJECTID")
    id_field = lic_fields.get("id_empreendimento")
    process_field = lic_fields.get("num_processo") or lic_fields.get("numero_processo")
    license_rows = []

    # Primeira chave: ID do empreendimento, em blocos para respeitar limites do serviço.
    plant_ids = []
    if "id_empreendimento" in plant_df.columns and id_field:
        for value in plant_df["id_empreendimento"].dropna().unique().tolist():
            try:
                plant_ids.append(int(value))
            except (TypeError, ValueError):
                pass
    for start in range(0, len(plant_ids), 150):
        block = plant_ids[start:start + 150]
        if block:
            license_rows.extend(query_license_rows(f"{id_field} IN ({','.join(map(str, block))})", out_fields))

    # Segunda chave: número do processo, apenas para registros ainda sem ID correspondente.
    existing_ids = {norm_key(r.get(id_field)) for r in license_rows} if id_field else set()
    process_values = []
    if process_field and "numero_processo" in plant_df.columns:
        for value in plant_df["numero_processo"].dropna().unique().tolist():
            val = str(value).strip()
            if val and len(val) <= 100:
                process_values.append(val)
    if process_values:
        # Processo é consultado separadamente; duplicatas são removidas por OBJECTID.
        for start in range(0, len(process_values), 40):
            block = process_values[start:start + 40]
            clauses = ",".join(sql_string(x) for x in block)
            license_rows.extend(query_license_rows(f"{process_field} IN ({clauses})", out_fields))

    unique = {}
    for row in license_rows:
        key = norm_key(row.get(oid_field)) or json.dumps(row, sort_keys=True, ensure_ascii=False)
        unique[key] = row
    license_rows = list(unique.values())

    by_id, by_process = {}, {}
    for row in license_rows:
        if id_field and norm_key(row.get(id_field)):
            by_id.setdefault(norm_key(row.get(id_field)), []).append(row)
        if process_field and norm_key(row.get(process_field)):
            by_process.setdefault(norm_key(row.get(process_field)), []).append(row)

    enriched = []
    matched = multiple = 0
    cpf_field = bool(lic_fields.get("cpf"))
    for _, plant in plant_df.iterrows():
        item = plant.to_dict()
        candidates = []
        match_method = "Sem correspondência"
        pid = norm_key(item.get("id_empreendimento"))
        proc = norm_key(item.get("numero_processo"))
        if pid and by_id.get(pid):
            candidates = by_id[pid]
            match_method = "ID do empreendimento"
        elif proc and by_process.get(proc):
            candidates = by_process[proc]
            match_method = "Número do processo"
        def date_sort_key(row):
            value = row.get(lic_fields.get("data_expedicao", ""))
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
                return int(parsed.timestamp()) if not pd.isna(parsed) else 0
        candidates = sorted(candidates, key=date_sort_key, reverse=True)
        chosen = candidates[0] if candidates else {}
        if candidates:
            matched += 1
            if len(candidates) > 1:
                multiple += 1
        def val(field):
            actual = lic_fields.get(field)
            return chosen.get(actual) if actual else None
        item.update({
            "correspondencia_licenca": "Confirmada" if candidates else "Não localizada",
            "criterio_correspondencia": match_method,
            "quantidade_licencas_correspondentes": len(candidates),
            "propriedade_nome_imovel": val("nome_imovel"),
            "proprietario_detentor": val("nome_detentor"),
            "cpf": val("cpf") if cpf_field else None,
            "cnpj": val("cnpj"),
            "municipio_licenca": val("municipio"),
            "data_expedicao_licenca": val("data_expedicao"),
            "validade_licenca": val("validade_licenca"),
            "situacao_licenca": val("status_licenca"),
            "tipo_licenca": val("tipo_licenca"),
            "numero_processo_licenca": chosen.get(process_field) if process_field else None,
            "id_empreendimento_licenca": chosen.get(id_field) if id_field else None,
            "car_licenca": val("car") or val("sicar"),
        })
        enriched.append(item)
    result = pd.DataFrame(enriched)
    if "data_expedicao_licenca" in result.columns:
        raw = pd.to_numeric(result["data_expedicao_licenca"], errors="coerce")
        is_ms = raw.notna() & raw.between(100000000000, 3000000000000)
        parsed = pd.to_datetime(raw.where(is_ms), unit="ms", errors="coerce", utc=True)
        other = pd.to_datetime(result["data_expedicao_licenca"].where(~is_ms), errors="coerce", dayfirst=True, utc=True)
        result["data_expedicao_licenca"] = parsed.fillna(other).dt.strftime("%d/%m/%Y")
        result["ano_expedicao_licenca"] = pd.to_datetime(result["data_expedicao_licenca"], format="%d/%m/%Y", errors="coerce").dt.year
    return result, {"matched": matched, "unmatched": len(plant_df) - matched, "multiple": multiple, "cpf_field": cpf_field}

def is_forest_planting_class(value):
    """Filtro conservador: inclui classes explicitamente relacionadas a plantio/silvicultura."""
    text = str(value or "").casefold()
    positive = ("plantio florestal", "floresta plantada", "florestas plantadas", "silvicultura",
                "reflorestamento", "implantação de floresta", "implantacao de floresta")
    return any(term in text for term in positive)

def excel_bytes(df, source_name, layer_url, where, total, done, pending, failed):
    output = io.BytesIO()
    metadata = pd.DataFrame([
        ["Fonte/base", source_name],
        ["Serviço ArcGIS REST", layer_url],
        ["Consulta aplicada", where],
        ["Data/hora da extração (UTC)", datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M:%S")],
        ["IDs encontrados", total],
        ["Registros extraídos", done],
        ["Pendentes", pending],
        ["IDs sem retorno/falha", failed],
        ["Observação", "Dados públicos retornados pelo serviço consultado. Confirmar na fonte oficial antes de uso em procedimento fiscalizatório."],
    ], columns=["Item", "Informação"])
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sheet_name = "PlantioFlorestal" if "eflorestal" in layer_url.lower() else "Licencas"
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        metadata.to_excel(writer, index=False, sheet_name="Metadados")
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = min(max((len(str(c.value or "")) for c in list(col)[:300]), default=10) + 2, 45)
                ws.column_dimensions[col[0].column_letter].width = max(12, width)
    output.seek(0)
    return output.getvalue()

st.title("🌱 Extrator de Licenças Ambientais e Plantio Florestal — IMASUL")
st.caption("CREA-MS · Plantio florestal e cruzamento cadastral com licenças · Retomada automática · Exportação XLSX")
st.warning("As camadas são bases distintas e não necessariamente representam todas as licenças concedidas pelo IMASUL. Valide os dados na fonte oficial.")

with st.sidebar:
    st.header("Configuração")
    source_name = st.selectbox("Base de dados", list(SOURCES.keys()))
source = SOURCES[source_name]
layer_url = source["url"]

try:
    metadata = get_metadata(layer_url)
    fields = field_options(metadata)
    st.success(f"Serviço acessível · Limite informado: {metadata.get('maxRecordCount', 'não informado')} registros por consulta")
except Exception as exc:
    st.error(f"Não foi possível consultar os metadados: {exc}")
    st.stop()

with st.expander("Campos técnicos identificados no serviço"):
    st.dataframe(pd.DataFrame(fields), use_container_width=True, hide_index=True)

st.subheader("Filtros da extração")
date_fields = [f for f in fields if f["type"] == "esriFieldTypeDate"]
preferred = detect_emission_field(fields)
date_field_names = [f["name"] for f in date_fields]
if date_field_names:
    default_idx = date_field_names.index(preferred) if preferred in date_field_names else 0
    chosen_date_field = st.selectbox(
        "Campo de data (preferencialmente emissão/expedição da licença)",
        date_field_names, index=default_idx,
        format_func=lambda n: next((f"{n} — {x['alias']}" for x in fields if x["name"] == n), n)
    )
else:
    chosen_date_field = None
    st.info("Esta camada não possui campo de data reconhecido. Não será possível aplicar filtro temporal.")

if date_field_names and source["key"] != "eflorestal":
    period_mode = st.radio("Período", ["Todos os períodos", "Ano", "Ano e mês"], horizontal=True)
else:
    period_mode = "Todos os períodos"
    if source["key"] == "eflorestal":
        st.caption("A camada de plantio não publica data. O ano será aplicado posteriormente pela data de expedição da licença vinculada, quando houver correspondência.")
    else:
        st.caption("Esta camada não possui campo de data publicado; a pesquisa será feita sem filtro temporal.")
c1, c2 = st.columns(2)
with c1:
    year = st.selectbox("Ano", list(range(datetime.now().year, 1999, -1)), disabled=(period_mode == "Todos os períodos" or not date_field_names or source["key"] == "eflorestal"))
with c2:
    month_names = ["Janeiro","Fevereiro","Março","Abril","Maio","Junho","Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]
    month_label = st.selectbox("Mês", month_names, disabled=(period_mode != "Ano e mês" or not date_field_names or source["key"] == "eflorestal"))
    month = month_names.index(month_label) + 1

date_where = make_date_where(chosen_date_field, period_mode, year, month)

# The rural licensing layer exposes tipo_empreendimento. If another layer has
# a comparable field, offer it as well, with its real field name and alias.
type_candidates = [f for f in fields if f["name"].lower() in {"tipo_empreendimento", "tipoempreendimento"}]
if not type_candidates:
    type_candidates = [f for f in fields if "tipo" in f["name"].lower() and
                       ("empreend" in f["name"].lower() or "atividade" in f["name"].lower())]
type_field = type_candidates[0]["name"] if type_candidates else None

selected_type = "Todos"
if type_field:
    type_alias = next((f["alias"] for f in fields if f["name"] == type_field), type_field)
    st.markdown("**Filtro por tipo de empreendimento**")
    try:
        with st.spinner("Carregando opções de tipos de empreendimento..."):
            type_values = get_distinct_values(layer_url, type_field, date_where)
            if source["key"] == "eflorestal":
                planting_values = [v for v in type_values if is_forest_planting_class(v)]
                if planting_values:
                    type_values = planting_values
                else:
                    st.warning("Não foi possível identificar automaticamente classes explicitamente de plantio/silvicultura. Para evitar incluir corte, extração ou queima, o filtro foi mantido sem opções específicas até validação da classificação oficial.")
                    type_values = []
        selected_type = st.selectbox(
            f"{type_alias} — selecione o que deseja fiscalizar",
            ["Todos"] + type_values,
            help=("Lista restrita a classes com indicação explícita de plantio, floresta plantada, silvicultura ou reflorestamento." if source["key"] == "eflorestal" else "As opções são obtidas diretamente da camada selecionada e respeitam o período escolhido.")
        )
        st.caption(f"{len(type_values):,} tipos/opções encontrados na fonte para o período selecionado.")
    except Exception as exc:
        st.warning(f"Não foi possível carregar a lista de tipos distintos: {exc}. A extração poderá ser feita sem esse filtro.")
else:
    st.info("Esta camada não possui campo de tipo de empreendimento identificado. O filtro específico ficará disponível na camada que contém esse atributo.")

batch_size = st.select_slider("Tamanho dos lotes (menor = mais conservador)", options=[50, 100, 200, 300], value=DEFAULT_BATCH)
where = make_full_where(date_where, type_field, selected_type)
signature = job_signature(source["key"], where)
db_path = DATA_DIR / f"job_{signature}.sqlite"
st.caption(f"Consulta enviada ao serviço: `{where}`")
st.caption("Cada combinação de base, período e tipo de empreendimento mantém seu próprio ponto de retomada.")
conn = db_connect(db_path)

try:
    if st.button("🆕 Iniciar / preparar extração com estes filtros", type="primary", use_container_width=True):
        initialize_job(conn, layer_url, where, source_name)
        st.rerun()

    if get_meta(conn, "ids_loaded") == "1":
        total = int(get_meta(conn, "total_ids", "0"))
        done = completed_count(conn)
        pending = pending_count(conn)
        failed = conn.execute("SELECT COUNT(*) FROM pending_ids WHERE done=2").fetchone()[0]
        st.subheader("Estado da extração")
        a,b,c,d = st.columns(4)
        a.metric("IDs encontrados", f"{total:,}")
        b.metric("Extraídos", f"{done:,}")
        c.metric("Pendentes", f"{pending:,}")
        d.metric("Sem retorno/falha", f"{failed:,}")
        if pending > 0:
            if st.button("▶️ Continuar extração de onde parou", use_container_width=True):
                progress = st.progress(0, text="Retomando extração...")
                status = st.empty()
                initial_pending = pending
                while pending_count(conn) > 0:
                    before_pending = pending_count(conn)
                    try:
                        got = extract_batch(conn, layer_url, batch_size)
                    except Exception as exc:
                        status.error(f"Interrupção: {exc}. Os lotes já salvos foram preservados. Clique em continuar novamente.")
                        break
                    current_pending = pending_count(conn)
                    current_done = completed_count(conn)
                    progress.progress(min(1.0, (initial_pending-current_pending)/max(initial_pending,1)),
                                      text=f"Registros salvos: {current_done:,} · Pendentes: {current_pending:,}")
                    status.write(f"Último lote: {got} registros · Total salvo: {current_done:,}")
                    if got == 0 and current_pending == before_pending:
                        status.warning("Nenhum avanço neste lote. A extração foi pausada para evitar repetição infinita.")
                        break
                st.rerun()

        if pending == 0:
            st.success("Extração concluída: não há IDs pendentes.")
        df = load_dataframe(conn)
        if not df.empty:
            export_df = df.copy()
            if source["key"] == "eflorestal":
                st.subheader("Cruzamento com licenças e identificação do imóvel")
                try:
                    with st.spinner("Consultando a camada de licenças e cruzando os empreendimentos..."):
                        export_df, match_stats = enrich_planting_dataframe(df)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Plantios com licença vinculada", f"{match_stats['matched']:,}")
                    m2.metric("Sem correspondência", f"{match_stats['unmatched']:,}")
                    m3.metric("Com múltiplas licenças", f"{match_stats['multiple']:,}")
                    if not match_stats["cpf_field"]:
                        st.info("A camada de licenças consultada não publica campo de CPF. O CNPJ será preenchido quando disponível; CPF ficará em branco, sem inferência.")
                    if "ano_expedicao_licenca" in export_df.columns:
                        years = sorted([int(x) for x in export_df["ano_expedicao_licenca"].dropna().unique()], reverse=True)
                        if years:
                            year_filter = st.selectbox("Ano de expedição da licença vinculada", ["Todos os anos"] + years)
                            if year_filter != "Todos os anos":
                                export_df = export_df[export_df["ano_expedicao_licenca"] == int(year_filter)].copy()
                        else:
                            st.warning("Nenhuma data de expedição foi recuperada nas correspondências encontradas.")
                    st.caption("Correspondência confirmada somente por ID do empreendimento ou número exato do processo. Correspondências por nome não são realizadas automaticamente.")
                except Exception as exc:
                    st.error(f"Falha no cruzamento com a camada de licenças: {exc}")
                    export_df = df.copy()
            st.subheader("Prévia dos registros extraídos")
            st.dataframe(export_df.head(1000), use_container_width=True, hide_index=True, height=420)
            xlsx = excel_bytes(export_df, source_name, layer_url, where, total, len(export_df), pending_count(conn), failed)
            st.download_button("⬇️ Baixar Excel (.xlsx)", data=xlsx,
                               file_name=f"imasul_{source['key']}_{signature}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
            csv = export_df.to_csv(index=False, sep=";", encoding="utf-8-sig")
            st.download_button("⬇️ Baixar CSV", data=csv,
                               file_name=f"imasul_{source['key']}_{signature}.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.info("Ainda não há registros extraídos para esta consulta.")
    else:
        st.info("Configure os filtros e clique em 'Iniciar / preparar extração'. A fila de IDs será salva localmente para permitir retomada.")
except Exception as exc:
    st.error(f"Erro na preparação da extração: {exc}")
finally:
    conn.close()

st.divider()
st.caption("Os dados e checkpoints ficam em `dados_imasul`, ao lado do app.py. Preserve essa pasta para manter a possibilidade de retomada.")
st.caption("O cruzamento cadastral usa ID do empreendimento e, como segunda chave, número exato do processo. A camada de licenças consultada publica CNPJ, nome do detentor e nome do imóvel; CPF só será preenchido se houver campo publicado na fonte.")
