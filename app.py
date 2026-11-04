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
import folium
from streamlit.components.v1 import html as st_html
from shapely.geometry import Polygon, mapping

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

def esri_rings_to_geojson(rings):
    """Converte anéis Esri em geometria GeoJSON, agrupando furos por contenção."""
    exteriors, holes = [], []
    for ring in rings or []:
        if len(ring) < 4:
            continue
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        # No padrão Esri, anéis externos são orientados no sentido horário.
        signed = sum((ring[i][0] * ring[i+1][1] - ring[i+1][0] * ring[i][1]) for i in range(len(ring)-1))
        (exteriors if signed < 0 else holes).append(poly)
    polygons = []
    for outer in exteriors:
        inner = [h.exterior.coords[:] for h in holes if outer.contains(h.representative_point())]
        polygons.append(Polygon(outer.exterior.coords[:], inner))
    if not polygons and holes:
        polygons = [Polygon(h.exterior.coords[:]) for h in holes]
    if not polygons:
        return None
    from shapely.geometry import MultiPolygon
    geom = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    return mapping(geom)

@st.cache_data(ttl=300, show_spinner=False)
def get_map_features(layer_url, where, limit=500):
    """Busca geometria em lotes pequenos para pré-visualização; não substitui a extração completa."""
    endpoint = f"{layer_url}/query"
    ids_payload = arcgis_get(endpoint, {
        "where": where, "returnIdsOnly": "true", "returnGeometry": "false", "f": "json"
    })
    ids = (ids_payload.get("objectIds") or [])[:limit]
    features = []
    for start in range(0, len(ids), 25):
        block = ids[start:start+25]
        payload = arcgis_get(endpoint, {
            "objectIds": ",".join(map(str, block)), "outFields": "*",
            "returnGeometry": "true", "outSR": 4326, "f": "json"
        })
        for item in payload.get("features", []):
            geom = item.get("geometry") or {}
            gj = esri_rings_to_geojson(geom.get("rings"))
            if not gj:
                continue
            props = item.get("attributes", {})
            # Limita atributos no popup para evitar excesso de conteúdo.
            clean = {str(k): (str(v)[:250] if v is not None else "") for k, v in props.items()
                     if k.lower() not in {"shape", "shape__area", "shape__length"}}
            features.append({"type": "Feature", "geometry": gj, "properties": clean})
    return features, len(ids), len(ids_payload.get("objectIds") or [])

def show_forest_map(layer_url, where):
    st.subheader("🗺️ Mapa das áreas de plantio florestal")
    limit = st.select_slider("Quantidade máxima de polígonos para visualizar", [100, 250, 500, 1000], value=500)
    if st.button("🌳 Carregar áreas no mapa", type="primary"):
        try:
            with st.spinner("Consultando geometrias no serviço do IMASUL..."):
                features, shown, total = get_map_features(layer_url, where, limit)
            if not features:
                st.warning("A consulta não retornou geometrias para os filtros selecionados.")
                return
            all_coords = []
            for ft in features:
                geom = ft["geometry"]
                coords = geom.get("coordinates", [])
                def walk(x):
                    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (int, float)):
                        all_coords.append(x)
                    elif isinstance(x, (list, tuple)):
                        for y in x: walk(y)
                walk(coords)
            center = [sum(x[1] for x in all_coords)/len(all_coords), sum(x[0] for x in all_coords)/len(all_coords)] if all_coords else [-20.5, -54.5]
            fmap = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron", control_scale=True)
            tip_map = [("nome_empreendimento", "Empreendimento"), ("tipo_empreendimento", "Tipo"),
                       ("situacao", "Situação"), ("numero_processo", "Processo"), ("numero_car", "CAR")]
            tip_fields = [k for k, _ in tip_map if k in features[0]["properties"]]
            tip_aliases = [label for k, label in tip_map if k in features[0]["properties"]]
            gj = folium.GeoJson({"type":"FeatureCollection", "features":features}, name="Plantio florestal",
                style_function=lambda feat: {"color":"#176b3a", "weight":1.2, "fillColor":"#43a66b", "fillOpacity":0.38},
                highlight_function=lambda feat: {"weight":3, "fillOpacity":0.65},
                tooltip=folium.GeoJsonTooltip(fields=tip_fields, aliases=tip_aliases, localize=True, sticky=False) if tip_fields else None,
                popup=folium.GeoJsonPopup(fields=list(features[0]["properties"].keys())[:15], labels=True, max_width=450))
            gj.add_to(fmap)
            folium.LayerControl().add_to(fmap)
            st.caption(f"Exibindo {len(features):,} polígonos de uma consulta com {total:,} registros. Limite visual: {limit:,}. A visualização é uma amostra; use a extração para obter a base completa.")
            st_html(fmap.get_root().render(), height=650, scrolling=True)
        except Exception as exc:
            st.error(f"Falha ao carregar o mapa: {exc}")

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
st.caption("CREA-MS · Pesquisa de licenças e empreendimentos florestais · Filtros disponíveis por camada · Retomada automática · Exportação XLSX")
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

if date_field_names:
    period_mode = st.radio("Período", ["Todos os períodos", "Ano", "Ano e mês"], horizontal=True)
else:
    period_mode = "Todos os períodos"
    st.caption("Esta camada não possui campo de data publicado; a pesquisa será feita sem filtro temporal.")
c1, c2 = st.columns(2)
with c1:
    year = st.selectbox("Ano", list(range(datetime.now().year, 1999, -1)), disabled=(period_mode == "Todos os períodos" or not date_field_names))
with c2:
    month_names = ["Janeiro","Fevereiro","Março","Abril","Maio","Junho","Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]
    month_label = st.selectbox("Mês", month_names, disabled=(period_mode != "Ano e mês" or not date_field_names))
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
        selected_type = st.selectbox(
            f"{type_alias} — selecione o que deseja fiscalizar",
            ["Todos"] + type_values,
            help="As opções são obtidas diretamente da camada selecionada e respeitam o período escolhido."
        )
        st.caption(f"{len(type_values):,} tipos/opções encontrados na fonte para o período selecionado.")
    except Exception as exc:
        st.warning(f"Não foi possível carregar a lista de tipos distintos: {exc}. A extração poderá ser feita sem esse filtro.")
else:
    st.info("Esta camada não possui campo de tipo de empreendimento identificado. O filtro específico ficará disponível na camada que contém esse atributo.")

batch_size = st.select_slider("Tamanho dos lotes (menor = mais conservador)", options=[50, 100, 200, 300], value=DEFAULT_BATCH)
where = make_full_where(date_where, type_field, selected_type)
if source["key"] == "eflorestal":
    with st.expander("🗺️ Visualizar áreas de plantio florestal no mapa", expanded=True):
        show_forest_map(layer_url, where)
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
            st.subheader("Prévia dos registros extraídos")
            st.dataframe(df.head(1000), use_container_width=True, hide_index=True, height=420)
            xlsx = excel_bytes(df, source_name, layer_url, where, total, len(df), pending_count(conn), failed)
            st.download_button("⬇️ Baixar Excel (.xlsx)", data=xlsx,
                               file_name=f"imasul_{source['key']}_{signature}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
            csv = df.to_csv(index=False, sep=";", encoding="utf-8-sig")
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
st.caption("O Excel inclui a aba Metadados com fonte, consulta, data/hora e estado da coleta. A camada EFlorestal não publica campo de data nem CPF/CNPJ na estrutura consultada; use a base de licenças para complementar a identificação quando houver correspondência.")
