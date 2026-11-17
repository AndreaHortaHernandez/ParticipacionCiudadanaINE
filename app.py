"""App de riesgo de abstención por sección electoral (INE).

Uso:  streamlit run app.py
Requiere las salidas de las fases 1-3 en data/processed/ y, para los mapas,
la cartografía ligera de `python -m src.geo` (data/processed/geo/).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agent import (CATALOGO, CAUSAS, FACTOR_ESTATAL, FACTORES_BASE, INERCIA,
                       RECOMENDACION_ESTATAL, candidatas, causas, planificar,
                       recomendacion_estatal, resumen)
from src.ingest import CLAVE, PROC_DIR
from src.risk_model import DESCRIPCION_FACTOR, GRUPOS

GEO_DIR = os.path.join(PROC_DIR, "geo")
ANIOS = [2027, 2024, 2021]
ENTIDADES = {
    1: "Aguascalientes", 2: "Baja California", 3: "Baja California Sur", 4: "Campeche",
    5: "Coahuila", 6: "Colima", 7: "Chiapas", 8: "Chihuahua", 9: "Ciudad de México",
    10: "Durango", 11: "Guanajuato", 12: "Guerrero", 13: "Hidalgo", 14: "Jalisco",
    15: "México", 16: "Michoacán", 17: "Morelos", 18: "Nayarit", 19: "Nuevo León",
    20: "Oaxaca", 21: "Puebla", 22: "Querétaro", 23: "Quintana Roo", 24: "San Luis Potosí",
    25: "Sinaloa", 26: "Sonora", 27: "Tabasco", 28: "Tamaulipas", 29: "Tlaxcala",
    30: "Veracruz", 31: "Yucatán", 32: "Zacatecas",
}
NOMBRE_FACTOR = {
    "historial_bajo": "Historial de la sección", "desmovilizacion_intermedias": "Desmovilización en intermedias",
    "caida_reciente": "Caída reciente", "volatilidad": "Volatilidad", "jovenes": "Jóvenes",
    "envejecimiento": "Estructura de edad", "genero": "Brecha de género",
    "territorio_acceso": "Tipo de sección y crecimiento del padrón", "contexto_municipal": "Contexto municipal", "contexto_vecinal": "Contexto vecinal",
    "contexto_estatal": "Contexto estatal", "concurrencia_local": "Menos elecciones locales concurrentes",
    "calidad_dato": "Calidad del dato", "ninguno": "Ninguno", "ninguna": "Ninguna",
    INERCIA: "Baja participación crónica (causa por identificar)",
}
NOMBRE_MEDIDA = {m: d["nombre"] for m, d in CATALOGO.items()}

# --------------------------------------------------------------------------
# Identidad visual (un solo lugar para cambiar colores; ver también .streamlit/config.toml)
# --------------------------------------------------------------------------
# Inspirada en la plataforma de Conteos Censales del INE: morado + magenta sobre blanco/lavanda.
PALETA = {
    "primario": "#D4006A",      # magenta (títulos, acentos, riesgo alto)
    "primario_osc": "#9C0050",
    "secundario": "#5C2483",    # morado institucional
    "morado_osc": "#3B1B57",
    "lavanda": "#ECE5F1",
    "lavanda_media": "#CDB9D8",
    "texto": "#3B1B57",
    "gris": "#6B6B78",
    "fondo_suave": "#F6F2F8",
    "borde": "#CDB9D8",
    "bajo": "#B9A6C6",          # factores que reducen el riesgo
}
ESCALA_RIESGO = ["#F3EEF6", "#CDB9D8", "#9E6BB8", "#6E2C91", "#3B1B57"]
SECUENCIA = ["#5C2483", "#D4006A", "#9E6BB8", "#E86AA6", "#3B1B57", "#B5006A",
             "#CDB9D8", "#7B3FA0", "#F2A6CF", "#8C6D9F", "#6B6B78", "#2B1745"]
COLOR_NIVEL = {"Alto": PALETA["primario"], "Medio": "#9E6BB8", "Bajo": "#D9C3E3"}
FUENTE = "Poppins, 'Segoe UI', Arial, sans-serif"
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = SECUENCIA
px.defaults.color_continuous_scale = ESCALA_RIESGO

st.set_page_config(page_title="Riesgo de abstención", page_icon="🗳️", layout="wide")
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
  html, body, [class*="css"], .stMarkdown, .stText, button, input, label, p, h1, h2, h3, h4 {{
      font-family: {FUENTE}; }}
  h2, h3 {{ color:{PALETA['primario']} !important; font-weight:600 !important; }}
  h4 {{ color:{PALETA['secundario']} !important; font-weight:600 !important; }}

  /* Encabezado tipo plataforma INE */
  .banda {{ background:#FFFFFF; padding:8px 4px 14px 4px; margin:-1.2rem 0 1rem 0;
            border-bottom:4px solid; border-image:linear-gradient(90deg,{PALETA['secundario']},{PALETA['primario']}) 1; }}
  .banda h1 {{ color:{PALETA['secundario']}; font-size:2.1rem; font-weight:700; margin:0; padding:0; line-height:1.15; }}
  .banda h1 span {{ color:{PALETA['primario']}; }}
  .banda p {{ color:{PALETA['primario']}; margin:6px 0 0 0; font-size:1.05rem; font-weight:500; }}

  /* Barra lateral morada como los filtros del tablero del INE */
  section[data-testid="stSidebar"] {{ background:{PALETA['secundario']}; }}
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * ,
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{ color:#FFFFFF !important; }}
  section[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background:#FFFFFF; border-radius:8px; }}

  /* Indicadores como las insignias moradas del INE */
  div[data-testid="stMetric"] {{ background:{PALETA['secundario']}; border-radius:22px; padding:16px 18px;
            text-align:center; box-shadow:0 2px 6px rgba(59,27,87,.18); }}
  div[data-testid="stMetric"] * {{ color:#FFFFFF !important; justify-content:center; }}
  div[data-testid="stMetricValue"] {{ font-weight:700; }}

  /* Navegación principal como pestañas */
  div.st-key-vista div[role="radiogroup"] {{
      gap:6px; border-bottom:2px solid {PALETA['lavanda_media']}; padding-bottom:6px; }}
  div.st-key-vista label {{ background:{PALETA['lavanda']}; border-radius:18px 18px 0 0; padding:6px 14px !important;
      margin-right:4px; }}
  div.st-key-vista label:has(input:checked) {{ background:{PALETA['secundario']}; }}
  div.st-key-vista label:has(input:checked) p {{ color:#FFFFFF !important; }}
  div.st-key-vista label p {{ color:{PALETA['secundario']}; font-weight:600; }}
  div.st-key-vista label > div:first-child {{ display:none; }}
  /* Pestañas */
  button[data-baseweb="tab"] p {{ font-size:1rem; font-weight:600; color:{PALETA['secundario']}; }}
  button[data-baseweb="tab"][aria-selected="true"] p {{ color:{PALETA['primario']}; }}

  /* Bloques punteados lavanda, como las tarjetas del tablero */
  .conclusion {{ background:{PALETA['fondo_suave']}; border:2px dotted {PALETA['lavanda_media']};
            border-left:6px solid {PALETA['primario']}; padding:18px 22px; border-radius:14px;
            font-size:1.03rem; line-height:1.6; color:{PALETA['texto']}; }}
  .conclusion b {{ color:{PALETA['secundario']}; }}
  .paso {{ display:inline-block; background:{PALETA['primario']}; color:#fff; border-radius:50%;
           width:30px; height:30px; text-align:center; line-height:30px; font-weight:700; margin-right:10px; }}
  .retos {{ display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 18px 0; }}
  .reto {{ flex:1; min-width:220px; border-radius:22px; padding:16px 18px; }}
  .reto h5 {{ margin:0 0 6px 0; font-weight:700; font-size:1.02rem; }}
  .reto p {{ margin:0; font-size:.92rem; line-height:1.45; }}
  .reto.r1 {{ background:{PALETA['lavanda']}; color:{PALETA['morado_osc']}; }}
  .reto.r2 {{ background:{PALETA['lavanda_media']}; color:{PALETA['morado_osc']}; }}
  .reto.r3 {{ background:{PALETA['secundario']}; color:#FFFFFF; }}
  .reto.r3 h5, .reto.r3 p {{ color:#FFFFFF; }}
  div.stButton > button[kind="primary"] {{ background:{PALETA['primario']}; border:none; border-radius:24px;
            padding:10px 28px; font-weight:600; font-size:1.05rem; }}
  div.stButton > button[kind="primary"]:hover {{ background:{PALETA['primario_osc']}; }}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
def _ruta(nombre: str) -> str:
    return os.path.join(PROC_DIR, nombre)


@st.cache_data(show_spinner="Cargando riesgo…")
def cargar_riesgo(anio: int) -> pd.DataFrame:
    r = pd.read_parquet(_ruta(f"riesgo_{anio}.parquet"))
    shap_cols = [c for c in r.columns if c.startswith("shap_")]
    cols = CLAVE + ["MPIOCVE", "LN_prev", "rural", "mixta", "nacional_supuesto", "pred_rel",
                    "participacion_pred", "riesgo", "riesgo_min", "riesgo_max", "prob_riesgo_alto",
                    "factor_1", "factor_1_pp", "factor_2",
                    "factor_2_pp", "factor_3", "factor_3_pp", "factor_dominante"] + shap_cols
    r = r[[c for c in cols if c in r.columns]].copy()
    # Las claves vienen como enteros pequeños (int16): se pasan a int64 para poder combinarlas
    r = r.astype({c: "int64" for c in CLAVE})
    r["EDONOM"] = r["EDOCVE"].map(ENTIDADES)
    r["tipo"] = np.select([r.get("rural", 0) == 1, r.get("mixta", 0) == 1], ["rural", "mixta"], "urbana")

    p = _perfiles()
    if p is not None:
        r = r.merge(p.rename(columns={"MPIOCVE": "MPIOCVE_perfil"}), on=CLAVE, how="left")
        r["MPIOCVE"] = r["MPIOCVE"].fillna(r.pop("MPIOCVE_perfil"))
    # Algunas secciones nuevas no traen municipio: -1 = sin municipio
    r["MPIOCVE"] = r["MPIOCVE"].fillna(-1).astype("int64")
    real = _participacion_real(anio)
    if real is not None:
        r = r.merge(real, on=CLAVE, how="left")
    r = r.join(causas(r))  # causa atendible principal y nivel base (ver src/agent.py)
    r["clave"] = r["EDOCVE"] * 100000 + r["SECCION"]
    # Sin pd.NA ni tipos "nullable": Plotly/Streamlit no los serializan (TypeError NAType)
    for c in r.columns:
        if (c.startswith(("factor_", "causa_")) and not c.endswith("_pp")) or c == "factor_dominante":
            r[c] = r[c].astype(object).where(r[c].notna(), "ninguna").astype(str)
        elif c in ("MPIONOM", "perfil_nombre"):
            r[c] = r[c].astype(object).where(r[c].notna(), "Sin dato").astype(str)
        elif pd.api.types.is_numeric_dtype(r[c]) and c not in CLAVE + ["MPIOCVE", "clave"]:
            r[c] = r[c].astype("float64")
    return r


def _perfiles() -> pd.DataFrame | None:
    f = _ruta("perfiles_seccion_2024.parquet")
    if not os.path.exists(f):
        return None
    p = pd.read_parquet(f, columns=CLAVE + ["MPIOCVE", "MPIONOM", "perfil", "perfil_nombre"])
    return p.astype({c: "int64" for c in CLAVE})


def _participacion_real(anio: int) -> pd.DataFrame | None:
    f = _ruta("panel_seccion.parquet")
    if anio > 2024 or not os.path.exists(f):
        return None
    p = pd.read_parquet(f, columns=["eleccion"] + CLAVE + ["participacion"],
                        filters=[("eleccion", "==", anio)])
    p = p.drop(columns="eleccion").astype({c: "int64" for c in CLAVE})
    return p.rename(columns={"participacion": "participacion_real"})


@st.cache_data
def cargar_json(nombre: str) -> dict | None:
    f = _ruta(nombre)
    if not os.path.exists(f):
        return None
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource(show_spinner="Cargando cartografía…")
def cargar_geo(nombre: str) -> dict | None:
    f = os.path.join(GEO_DIR, nombre)
    if not os.path.exists(f):
        return None
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner="El agente evalúa las acciones posibles…")
def cargar_candidatas(anio: int) -> pd.DataFrame:
    return candidatas(cargar_riesgo(anio))


def pond(df: pd.DataFrame, col: str, peso: str = "LN_prev") -> float:
    m = df[col].notna() & df[peso].notna()
    return float(np.average(df.loc[m, col], weights=df.loc[m, peso])) if m.any() else np.nan


ALTO_MAPA = 640


@st.cache_resource(show_spinner="Preparando el mapa…", max_entries=12)
def geo_secciones(edo: int, mpio: int):
    """Secciones de la entidad (o solo del municipio) + encuadre (centro, zoom). Se calcula una vez
    por zona y no depende de la elección: el filtro usa el municipio que trae la cartografía."""
    geo = cargar_geo(f"secciones_{edo:02d}.geojson")
    if geo is None:
        return None, None, None
    if mpio:
        geo = {"type": "FeatureCollection",
               "features": [f for f in geo["features"] if int(f["properties"].get("MPIOCVE", -1)) == mpio]}
    centro, zoom = _encuadre(geo, ancho_px=1100, alto_px=ALTO_MAPA - 40)
    return geo, centro, zoom


@st.cache_resource(show_spinner="Preparando el mapa…", max_entries=12)
def geo_municipios(edo: int):
    """Municipios de la entidad: ~20 veces menos polígonos que las secciones, carga casi inmediata."""
    geo = cargar_geo(f"municipios_{edo:02d}.geojson")
    if geo is None:
        return None, None, None
    centro, zoom = _encuadre(geo, ancho_px=1100, alto_px=ALTO_MAPA - 40)
    return geo, centro, zoom


def _moda(s: pd.Series) -> str:
    m = s.mode()
    return str(m.iat[0]) if len(m) else "Sin dato"


@st.cache_resource
def geo_pais():
    geo = cargar_geo("entidades.geojson")
    if geo is None:
        return None, None, None
    centro, zoom = _encuadre(geo, ancho_px=1100, alto_px=ALTO_MAPA - 40)
    return geo, centro, zoom


SIN_ACCION = "Sin acción del agente"
ETIQUETAS_MAPA = {"SECCION": "Sección", "MPIONOM": "Municipio", "riesgo": "Riesgo",
                  "factor": "Causa principal", "EDONOM": "Entidad", "secciones": "Secciones",
                  "LN": "Lista nominal", "perfil_nombre": "Perfil", "medida_agente": "Acción del agente",
                  "votantes": "Votantes adicionales esperados", "acciones": "Acciones del agente"}


def mapa(df, geo, centro, zoom, clave, color, hover, titulo, continuo=True, barra="Riesgo", formato=".0%"):
    """Coroplético en UNA sola capa: la geometría se envía una vez aunque el color
    sea por categorías (px crea una capa por categoría y repite toda la geometría)."""
    Traza = getattr(go, "Choroplethmap", None)
    nuevo = Traza is not None
    Traza = Traza or go.Choroplethmapbox
    cd = pd.DataFrame({h: (df[h].map(lambda x: f"{x:.1%}") if h == "riesgo" else
                           df[h].map(lambda x: f"{x:,.0f}") if h in ("LN", "secciones", "votantes", "acciones")
                           else df[h].astype(str))
                       for h in hover})
    plantilla = f"<b>{ETIQUETAS_MAPA.get(hover[0], hover[0])} %{{customdata[0]}}</b><br>" + "<br>".join(
        f"{ETIQUETAS_MAPA.get(h, h)}: %{{customdata[{i}]}}" for i, h in enumerate(hover) if i > 0) + "<extra></extra>"
    if continuo:
        z = df[color].astype(float)
        escala = [[i / (len(ESCALA_RIESGO) - 1), c] for i, c in enumerate(ESCALA_RIESGO)]
        extra = dict(colorbar=dict(title=barra, tickformat=formato))
    else:
        # "Sin acción" al final y en gris claro, para que resalte lo que sí eligió el agente
        cats = sorted(c for c in df[color].astype(str).unique() if c != SIN_ACCION)
        if (df[color] == SIN_ACCION).any():
            cats.append(SIN_ACCION)
        n = max(len(cats), 1)
        z = df[color].astype(str).map({c: i for i, c in enumerate(cats)})
        escala = []
        for i, cat in enumerate(cats):
            col = "#E9E4EE" if cat == SIN_ACCION else SECUENCIA[i % len(SECUENCIA)]
            escala += [[i / n, col], [(i + 1) / n, col]]
        extra = dict(zmin=-0.5, zmax=n - 0.5,
                     colorbar=dict(tickvals=list(range(n)), ticktext=cats, title=None, len=0.9))
    # Solo se envían al navegador los polígonos que tienen dato
    ids = set(df[clave].tolist())
    geo = {"type": "FeatureCollection",
           "features": [f for f in geo["features"] if f["properties"].get(clave) in ids]}
    # Con miles de polígonos el contorno blanco cuesta más que lo que aporta
    linea = 0.4 if len(df) <= 3000 else 0
    fig = go.Figure(Traza(geojson=geo, locations=df[clave], featureidkey=f"properties.{clave}", z=z,
                          colorscale=escala, marker_opacity=0.8, marker_line_width=linea,
                          marker_line_color="#FFFFFF", customdata=cd.to_numpy(),
                          hovertemplate=plantilla, **extra))
    capa = dict(style="carto-positron", center=centro, zoom=zoom)
    fig.update_layout(**({"map": capa} if nuevo else {"mapbox": capa}))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=ALTO_MAPA, title=titulo,
                      font=dict(family=FUENTE, color=PALETA["texto"]),
                      title_font=dict(size=16, color=PALETA["primario"]))
    return fig


def _coords(g, paso=4):
    t, c = g["type"], g["coordinates"]
    if t == "Polygon":
        return c[0][::paso]
    if t == "MultiPolygon":
        return [p for poly in c for p in poly[0][::paso]]
    return []


def _encuadre(geo, ancho_px: int, alto_px: int) -> tuple[dict, float]:
    """Centro y zoom (Web Mercator, teselas de 512 px) para que quepan todas las geometrías."""
    xs = [p for f in geo["features"] for p in _coords(f["geometry"])]
    if not xs:
        return {"lat": 23.6, "lon": -102.5}, 3.8
    lon, lat = np.array(xs, dtype=float).T
    lon0, lon1, lat0, lat1 = lon.min(), lon.max(), lat.min(), lat.max()

    def y(phi):  # latitud -> coordenada Mercator
        return np.log(np.tan(np.pi / 4 + np.radians(phi) / 2))
    dx = max(np.radians(lon1 - lon0), 1e-4)
    dy = max(y(lat1) - y(lat0), 1e-4)
    z = min(np.log2(ancho_px * 2 * np.pi / (512 * dx)), np.log2(alto_px * 2 * np.pi / (512 * dy)))
    lat_c = float(np.degrees(2 * np.arctan(np.exp((y(lat0) + y(lat1)) / 2)) - np.pi / 2))
    return {"lat": lat_c, "lon": float((lon0 + lon1) / 2)}, float(np.clip(z - 0.25, 3, 15))


# --------------------------------------------------------------------------
# Pantalla de resultados
# --------------------------------------------------------------------------
PRESUPUESTO_DEFECTO = 10  # % del costo de todas las acciones candidatas (se ajusta en la barra lateral)


def _fig(fig, alto=360):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50, b=10), legend_title=None,
                      font=dict(color=PALETA["texto"], family=FUENTE),
                      title_font=dict(size=16, color=PALETA["primario"]))
    return fig


@st.cache_data(show_spinner="El agente arma la estrategia…", max_entries=32)
def _plan_zona(anio: int, edo: int, mpio: int, pct: float) -> tuple[pd.DataFrame, dict]:
    cand = cargar_candidatas(anio)
    if edo:
        cand = cand[cand["EDOCVE"] == edo]
    if mpio:
        cand = cand[cand["MPIOCVE"] == mpio]
    if cand.empty:
        return cand, {}
    presupuesto = cand["costo"].sum() * pct / 100
    plan = planificar(cand, presupuesto)
    return plan, resumen(plan, presupuesto)


def _acciones_por_seccion(plan: pd.DataFrame) -> pd.DataFrame:
    """Lo que eligió el agente, una fila por sección: medida principal, acciones y votantes."""
    if plan.empty or "seleccionada" not in plan:
        return pd.DataFrame(columns=CLAVE + ["MPIOCVE", "medida_agente", "acciones", "votantes"])
    s = plan[plan["seleccionada"]].sort_values("prioridad")
    g = s.groupby(CLAVE, sort=False)
    out = pd.DataFrame({"MPIOCVE": g["MPIOCVE"].first(), "medida_agente": g["medida"].first().map(NOMBRE_MEDIDA),
                        "acciones": g.size(), "votantes": g["votantes"].sum()})
    return out.reset_index()


def traza_agente(fila: pd.Series, plan: pd.DataFrame, pct: float) -> None:
    """El ciclo del agente para UNA sección, con sus números: percibe, diagnostica, evalúa, decide."""
    ln = float(fila["LN_prev"]) if pd.notna(fila["LN_prev"]) else 0.0
    st.markdown(f"<span class='paso'>1</span>**Percibe.** El modelo predice un riesgo de "
                f"**{fila['riesgo']:.1%}** con {ln:,.0f} electores: ≈ **{fila['riesgo'] * ln:,.0f} "
                "abstencionistas** esperados.", unsafe_allow_html=True)

    base = float(fila.get("inercia_pp", 0) or 0)
    aportes = sorted(((NOMBRE_FACTOR.get(c, c), float(fila.get(f"shap_{c}", 0) or 0) * 100) for c in CAUSAS),
                     key=lambda x: -x[1])
    positivas = [(n, pp) for n, pp in aportes if pp > 0]
    diag = (", ".join(f"**{n.lower()}** (+{pp:.1f} pp)" for n, pp in positivas[:3]) if positivas else
            "ninguna causa atendible: el agente la trata como **baja participación crónica (causa por identificar)**")
    st.markdown(f"<span class='paso'>2</span>**Diagnostica.** Nivel base (historial y contexto) = "
                f"{base:.1f} pp: dice *cuánto* riesgo hay, no por qué. Causas atendibles: {diag}.",
                unsafe_allow_html=True)

    acc = plan[(plan["EDOCVE"] == fila["EDOCVE"]) & (plan["SECCION"] == fila["SECCION"])] if len(plan) else plan
    if acc.empty:
        st.markdown("<span class='paso'>3</span>**Evalúa.** No hay acciones posibles: ningún factor eleva el "
                    "riesgo de esta sección.", unsafe_allow_html=True)
        st.success("Decisión: no intervenir.")
        return
    st.markdown("<span class='paso'>3</span>**Evalúa** las medidas que atienden sus causas:",
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Medida": acc["medida"].map(NOMBRE_MEDIDA), "Atiende": acc["factor"].map(NOMBRE_FACTOR),
        "Aporte (pp)": acc["contribucion_pp"].round(1), "Efecto esperado (pp)": acc["ganancia_pp"].round(2),
        "Votantes extra": acc["votantes"].round(1), "Costo": acc["costo"].round(1),
        "Utilidad": acc["utilidad"].round(3)}), hide_index=True, width="stretch")

    sel = plan[plan["seleccionada"]]
    corte = float(sel["utilidad"].min()) if len(sel) else float("inf")
    lineas = []
    for _, a in acc.iterrows():
        nombre = NOMBRE_MEDIDA.get(a["medida"], a["medida"])
        if a["seleccionada"]:
            lineas.append(f"✅ **{nombre}**: entra. Utilidad {a['utilidad']:.3f} ≥ corte {corte:.3f}; "
                          f"es la acción #{int(a['prioridad']):,} de {len(plan):,} en la zona.")
        elif a["orden_en_seccion"] > 1 and a["utilidad"] >= corte:
            lineas.append(f"❌ **{nombre}**: no entra porque la medida de su causa principal no entró.")
        else:
            lineas.append(f"❌ **{nombre}**: no entra. Utilidad {a['utilidad']:.3f} < corte {corte:.3f}: el "
                          f"presupuesto del {pct}% se acaba antes (es la #{int(a['prioridad']):,}). "
                          "Con más presupuesto entraría.")
    st.markdown("<span class='paso'>4</span>**Decide.**<br>" + "<br>".join(lineas), unsafe_allow_html=True)


def pantalla_resultados(vista, datos, anio, zona, edo, mpio, pct):
    st.markdown(
        "<div class='retos'>"
        "<div class='reto r1'><h5>1 · Perfiles ciudadanos</h5><p>Agrupa las secciones en 5 perfiles "
        "según cómo han participado de 2009 a 2024.</p></div>"
        "<div class='reto r2'><h5>2 · Predicción de riesgo</h5><p>Estima la abstención de cada sección "
        "y explica qué la causa.</p></div>"
        "<div class='reto r3'><h5>3 · Estrategia y visualización</h5><p>Un agente propone qué hacer y "
        "dónde, y lo muestra en mapas y gráficas.</p></div></div>",
        unsafe_allow_html=True)
    st.markdown(
        "<span class='paso'>1</span>Elige la <b>elección</b> y la <b>zona</b> (entidad y municipio) "
        "en el panel de la izquierda.<br><br>"
        "<span class='paso'>2</span>Presiona <b>Ver resultados</b>: el sistema analiza las secciones, "
        "predice la abstención, explica sus causas y propone una estrategia.",
        unsafe_allow_html=True)
    st.write("")
    if st.button("▶  Ver resultados", type="primary"):
        st.session_state["ver_resultados"] = True
    if not st.session_state.get("ver_resultados"):
        return
    if vista.empty:
        st.info("No hay secciones con los filtros elegidos.")
        return

    v = vista.copy()
    # Niveles de riesgo relativos al país en esa elección
    p50, p80 = datos["riesgo"].quantile([0.5, 0.8])
    v["nivel"] = np.select([v["riesgo"] >= p80, v["riesgo"] >= p50], ["Alto", "Medio"], "Bajo")
    ln = v["LN_prev"].sum()
    abst = float((v["riesgo"] * v["LN_prev"]).sum())
    riesgo_zona, riesgo_pais = pond(v, "riesgo"), pond(datos, "riesgo")
    n_alto = int((v["nivel"] == "Alto").sum())
    plan, res = _plan_zona(anio, edo, mpio, pct)

    # Causas atendibles: aporte positivo medio (pp, ponderado por LN). El nivel base
    # (historial y contexto) se reporta aparte: dice cuánto riesgo hay, no por qué.
    w = v["LN_prev"].fillna(0) + 1e-9

    def _aporte(f):
        col = f"shap_{f}"
        return float(np.average(v[col].clip(lower=0).fillna(0) * 100, weights=w)) if col in v else 0.0
    aportes = pd.Series({NOMBRE_FACTOR.get(f, f): _aporte(f) for f in CAUSAS}).sort_values(ascending=False)
    base_pp = sum(_aporte(f) for f in FACTORES_BASE)
    estatal_pp = _aporte(FACTOR_ESTATAL)
    causa1 = aportes.index[0] if len(aportes) and aportes.iloc[0] > 0 else "—"
    estados_conc = recomendacion_estatal(v)

    st.subheader(f"Resultados · {zona} · elección {anio}")

    # ---- Conclusión automática
    comp = "mayor" if riesgo_zona > riesgo_pais else "menor"
    texto = (f"En <b>{zona}</b> se analizaron <b>{len(v):,} secciones</b> con <b>{ln:,.0f} electores</b>. "
             f"Para {anio} se esperan <b>{abst:,.0f} abstencionistas</b> (riesgo medio "
             f"<b>{riesgo_zona:.1%}</b>, {comp} que el promedio nacional de {riesgo_pais:.1%}). "
             f"<b>{n_alto:,}</b> {'sección está' if n_alto == 1 else 'secciones están'} ({n_alto / len(v):.0%}) en <b>riesgo alto</b>. "
             f"Más allá de su historial, la causa que más eleva el riesgo es <b>{causa1.lower()}</b>.")
    if len(estados_conc):
        nombres_e = ", ".join(ENTIDADES[int(e)] for e in estados_conc["EDOCVE"])
        texto += (f" Además, en <b>{nombres_e}</b> habrá menos elecciones locales el mismo día que en la anterior, lo que "
                  f"eleva el riesgo ≈{estados_conc['aporte_pp'].mean():.1f} pp: conviene una campaña con el OPLE.")
    if res:
        top_med = res["por_medida"][0]["nombre"] if res["por_medida"] else "—"
        texto += (f" Con un presupuesto del {pct}%, el agente propone "
                  f"<b>{res['acciones']:,} acciones</b> en <b>{res['secciones_intervenidas']:,} secciones</b>, "
                  f"con ≈<b>{res['votantes_adicionales_esperados']:,.0f} votantes adicionales</b>; "
                  f"la medida principal es <i>{top_med}</i>.")
    st.markdown(f"<div class='conclusion'>{texto}</div>", unsafe_allow_html=True)
    st.write("")

    # ---- Indicadores
    k = st.columns(5)
    k[0].metric("Secciones analizadas", f"{len(v):,}")
    k[1].metric("Electores", f"{ln:,.0f}")
    k[2].metric("Participación esperada", f"{1 - riesgo_zona:.1%}")
    k[3].metric("Abstencionistas esperados", f"{abst:,.0f}")
    k[4].metric("Secciones en riesgo alto", f"{n_alto:,}", f"{n_alto / len(v):.0%} del total", delta_color="off")

    # ---- Riesgo
    st.markdown("#### ¿Cuánto riesgo hay?")
    a, b = st.columns(2)
    niv = v["nivel"].value_counts().reindex(["Alto", "Medio", "Bajo"]).fillna(0).reset_index()
    niv.columns = ["nivel", "secciones"]
    a.plotly_chart(_fig(px.pie(niv, names="nivel", values="secciones", hole=0.55, color="nivel",
                               color_discrete_map=COLOR_NIVEL,
                               title="Secciones por nivel de riesgo (umbral: país)")), width="stretch")
    cnt, bordes = np.histogram(v["riesgo"].dropna(), bins=40)
    hd = pd.DataFrame({"riesgo": (bordes[:-1] + bordes[1:]) / 2, "secciones": cnt})
    h = px.bar(hd, x="riesgo", y="secciones", title="Distribución del riesgo de abstención",
               color_discrete_sequence=[PALETA["primario"]])
    h.update_traces(marker_line_width=0)
    h.update_layout(bargap=0.02)
    h.add_vline(x=riesgo_pais, line_dash="dash", line_color=PALETA["secundario"],
                annotation_text="promedio nacional")
    h.update_xaxes(tickformat=".0%", title="riesgo")
    h.update_yaxes(title="secciones")
    b.plotly_chart(_fig(h), width="stretch")

    # ---- Causas
    st.markdown("#### ¿Por qué? Causas del riesgo")
    st.caption(f"El historial y el contexto de la zona explican en promedio {base_pp:.1f} pp del riesgo "
               "(nivel base). Abajo, las causas sobre las que sí se puede actuar.")
    a, b = st.columns(2)
    c = aportes[aportes > 0].sort_values().reset_index()
    c.columns = ["causa", "pp"]
    if estatal_pp > 0:
        c = pd.concat([pd.DataFrame({"causa": ["Menos elecciones locales concurrentes (estatal)"], "pp": [estatal_pp]}), c])
        c = c.sort_values("pp")
    f1 = px.bar(c, x="pp", y="causa", orientation="h", color_discrete_sequence=[PALETA["primario"]],
                title="Aporte medio de cada causa al riesgo (pp, SHAP)")
    f1.update_layout(yaxis_title=None, xaxis_title="pp")
    a.plotly_chart(_fig(f1), width="stretch")
    fd = v["causa_principal"].map(NOMBRE_FACTOR).value_counts().head(8).reset_index()
    fd.columns = ["causa", "secciones"]
    b.plotly_chart(_fig(px.pie(fd, names="causa", values="secciones", hole=0.45,
                               title="Causa principal en cada sección")), width="stretch")

    # ---- Quiénes
    st.markdown("#### ¿Dónde y en quiénes?")
    a, b = st.columns(2)
    if "perfil_nombre" in v:
        pf = (v[v["perfil_nombre"] != "Sin dato"].groupby("perfil_nombre")
              .agg(secciones=("riesgo", "size"), riesgo=("riesgo", "mean")).reset_index())
        f2 = px.bar(pf.sort_values("riesgo"), x="riesgo", y="perfil_nombre", orientation="h",
                    color="riesgo", color_continuous_scale=ESCALA_RIESGO, hover_data=["secciones"],
                    title="Riesgo medio por perfil de votante")
        f2.update_layout(yaxis_title=None, coloraxis_showscale=False)
        f2.update_xaxes(tickformat=".0%")
        a.plotly_chart(_fig(f2), width="stretch")
    tp = v.groupby("tipo").agg(secciones=("riesgo", "size"), riesgo=("riesgo", "mean")).reset_index()
    f3 = px.bar(tp, x="tipo", y="riesgo", text=tp["secciones"].map(lambda n: f"{n:,} secc."),
                color_discrete_sequence=[PALETA["secundario"]], title="Riesgo medio por tipo de sección")
    f3.update_yaxes(tickformat=".0%", title=None)
    f3.update_xaxes(title=None)
    b.plotly_chart(_fig(f3), width="stretch")

    # ---- Estrategia
    st.markdown("#### Estrategia recomendada por el agente")
    if not res:
        st.info("No hay acciones candidatas para esta zona.")
    else:
        k = st.columns(4)
        k[0].metric("Acciones", f"{res['acciones']:,}")
        k[1].metric("Secciones intervenidas", f"{res['secciones_intervenidas']:,}")
        k[2].metric("Ciudadanos alcanzados", f"{res['ciudadanos_alcanzados']:,}")
        k[3].metric("Votantes adicionales esperados", f"{res['votantes_adicionales_esperados']:,.0f}")
        pm = pd.DataFrame(res["por_medida"])
        a, b = st.columns([3, 2])
        if not pm.empty:
            f4 = px.bar(pm.sort_values("votantes"), x="votantes", y="nombre", orientation="h",
                        color_discrete_sequence=[PALETA["primario"]], hover_data=["acciones", "costo"],
                        title="Votantes adicionales esperados por medida")
            f4.update_layout(yaxis_title=None)
            a.plotly_chart(_fig(f4, 400), width="stretch")
            b.plotly_chart(_fig(px.pie(pm, names="nombre", values="costo", hole=0.5,
                                       title="Reparto del presupuesto"), 400), width="stretch")
        if len(estados_conc):
            st.markdown("**Recomendación estatal:** " + RECOMENDACION_ESTATAL + " Entidades: "
                        + ", ".join(f"{ENTIDADES[int(e)]} (+{pp:.1f} pp)"
                                    for e, pp in zip(estados_conc["EDOCVE"], estados_conc["aporte_pp"])) + ".")
        sel = plan[plan["seleccionada"]].head(15).copy()
        sel["Medida"] = sel["medida"].map(NOMBRE_MEDIDA)
        sel["Causa que atiende"] = sel["factor"].map(NOMBRE_FACTOR)
        sel["Entidad"] = sel["EDOCVE"].map(ENTIDADES)
        st.markdown("**Primeras 15 acciones por prioridad**")
        st.dataframe(sel[["prioridad", "Entidad", "SECCION", "Medida", "Causa que atiende", "riesgo",
                          "votantes"]], hide_index=True, width="stretch",
                     column_config={"riesgo": st.column_config.ProgressColumn("riesgo", format="%.2f",
                                                                              min_value=0, max_value=1),
                                    "votantes": st.column_config.NumberColumn("votantes extra", format="%.1f")})
        st.caption(f"Presupuesto: {pct}% del costo de todas las acciones posibles (ajústalo en la barra "
                   "lateral). En 🤖 Agente se ve paso a paso por qué eligió cada sección. "
                   "Efectos y costos son supuestos ilustrativos.")

    # ---- Validación con dato real
    if "participacion_real" in v and v["participacion_real"].notna().any():
        st.markdown("#### ¿Acertó el modelo? (dato real de la elección)")
        m = v.dropna(subset=["participacion_real"])
        err = (m["participacion_pred"] - m["participacion_real"]).abs().mean() * 100
        rho = m["participacion_pred"].corr(m["participacion_real"], method="spearman")
        a, b = st.columns([1, 2])
        a.metric("Error medio", f"{err:.1f} pp")
        a.metric("Correlación de rangos", f"{rho:.2f}")
        mm = m.sample(min(len(m), 4000), random_state=0)
        f5 = px.scatter(mm, x="participacion_pred", y="participacion_real", opacity=0.35,
                        color_discrete_sequence=[PALETA["primario"]], title="Participación predicha vs. real")
        f5.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=PALETA["gris"]))
        f5.update_xaxes(tickformat=".0%", range=[0, 1])
        f5.update_yaxes(tickformat=".0%", range=[0, 1])
        b.plotly_chart(_fig(f5), width="stretch")

    # ---- Qué midió
    with st.expander("🔎 ¿Qué midió el sistema y cómo?"):
        met = cargar_json("riesgo_metricas.json") or {}
        st.markdown(
            "**Fuente:** conteos censales de participación del INE (2009–2024), por sección, edad y sexo, "
            "más el calendario de elecciones locales concurrentes.\n\n"
            "**Pasos:** 1) limpieza e integración por sección · 2) perfiles de votante (K‑Means) · "
            f"3) modelo de riesgo (`{met.get('modelo_elegido', '—')}`) validado con elecciones pasadas · "
            "4) explicación por factor (SHAP) · 5) agente que elige medidas por utilidad/costo.")
        def _tipo(g):
            if g in FACTORES_BASE:
                return "Nivel base (cuánto riesgo)"
            if g == FACTOR_ESTATAL:
                return "Contexto estatal → recomendación OPLE"
            return "Causa atendible → medida por sección"
        filas = [{"Factor": NOMBRE_FACTOR.get(g, g), "Tipo": _tipo(g), "Qué mide": DESCRIPCION_FACTOR.get(g, ""),
                  "Medida": next((d["nombre"] for d in CATALOGO.values() if d["factor"] == g), "—"),
                  "Variables": ", ".join(vs)} for g, vs in GRUPOS.items()]
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    # ---- Descarga
    salida = v[[c for c in ["EDONOM", "MPIONOM", "SECCION", "tipo", "LN_prev", "riesgo", "nivel",
                            "causa_1", "causa_2", "perfil_nombre", "participacion_real"] if c in v]]
    salida = salida.assign(**{c: salida[c].map(NOMBRE_FACTOR) for c in ("causa_1", "causa_2") if c in salida})
    st.download_button("⬇️ Descargar resultados de la zona (CSV)", salida.to_csv(index=False).encode("utf-8"),
                       f"resultados_{anio}_{zona.replace(' ', '_')}.csv", "text/csv")


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
disponibles = [a for a in ANIOS if os.path.exists(_ruta(f"riesgo_{a}.parquet"))]
if not disponibles:
    st.error("No hay salidas del modelo en data/processed/. Corre el pipeline en Colab "
             "(notebooks/colab_pipeline.ipynb) y copia la carpeta data/processed.")
    st.stop()

st.sidebar.title("🗳️ Riesgo de abstención")
anio = st.sidebar.selectbox(
    "Elección", disponibles,
    format_func=lambda a: f"{a} (pronóstico)" if a > 2024 else f"{a} (backtest, con dato real)")
datos = cargar_riesgo(anio)

edo_sel = st.sidebar.selectbox("Entidad", [0] + sorted(int(e) for e in datos["EDOCVE"].unique()),
                               format_func=lambda e: "Todo el país" if e == 0 else ENTIDADES.get(e, e))
vista = datos if edo_sel == 0 else datos[datos["EDOCVE"] == edo_sel]
mpio_sel = 0
if edo_sel and "MPIONOM" in vista:
    # Un nombre por municipio (el más frecuente con dato); -1 = sin municipio
    m_ = vista.loc[(vista["MPIOCVE"] > 0) & (vista["MPIONOM"] != "Sin dato"), ["MPIOCVE", "MPIONOM"]]
    nombres = m_.groupby("MPIOCVE")["MPIONOM"].agg(lambda s: s.mode().iat[0]).sort_values()
    mpio_sel = st.sidebar.selectbox("Municipio", [0] + [int(m) for m in nombres.index],
                                    format_func=lambda m: "Todos" if m == 0 else
                                    str(nombres.get(m, f"Municipio {m}")))
    if mpio_sel:
        vista = vista[vista["MPIOCVE"] == mpio_sel]

pct = st.sidebar.slider(
    "🤖 Presupuesto del agente (%)", 1, 100, PRESUPUESTO_DEFECTO,
    help="Porcentaje del costo de aplicar todas las acciones posibles en la zona. "
         "Lo usan Resultados, Mapa, Sección y Agente.")

st.sidebar.caption(
    "Riesgo = 1 − participación predicha. La participación predicha suma el nivel "
    "relativo que aprende el modelo y un supuesto de nivel nacional (elección comparable "
    "anterior). El modelo indica **dónde** y **por qué** hay riesgo, no efectos causales.")

# --------------------------------------------------------------------------
# Indicadores
# --------------------------------------------------------------------------
zona = ("Todo el país" if edo_sel == 0 else
        ENTIDADES[edo_sel] + (f" · {nombres.get(mpio_sel, mpio_sel)}" if mpio_sel else ""))
st.markdown(
    f"<div class='banda'><h1>Riesgo de <span>abstención</span> por sección electoral</h1>"
    f"<p>Elección {anio}{' (pronóstico)' if anio > 2024 else ''} · {zona}</p></div>",
    unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Secciones", f"{len(vista):,}")
c2.metric("Lista nominal", f"{vista['LN_prev'].sum():,.0f}")
c3.metric("Riesgo medio (pond. LN)", f"{pond(vista, 'riesgo'):.1%}")
if "participacion_real" in vista:
    c4.metric("Abstención real (pond. LN)", f"{1 - pond(vista, 'participacion_real'):.1%}")
else:
    c4.metric("Abstenc. esperada (personas)", f"{(vista['riesgo'] * vista['LN_prev']).sum():,.0f}")

# Navegación: a diferencia de st.tabs, solo se calcula y envía la vista abierta.
VISTAS = ["📊 Resultados", "🗺️ Mapa", "📍 Sección", "👥 Perfiles", "🤖 Agente de intervención", "⚙️ Modelo"]
pantalla = st.radio("Vista", VISTAS, horizontal=True, key="vista", label_visibility="collapsed")

if pantalla == VISTAS[0]:
    pantalla_resultados(vista, datos, anio, zona, edo_sel, mpio_sel, pct)

# --------------------------------------------------------------------------
# Mapa
# --------------------------------------------------------------------------
if pantalla == VISTAS[1]:
    plan_m, _ = _plan_zona(anio, edo_sel, mpio_sel, pct)
    acc = _acciones_por_seccion(plan_m)
    c_capa, c_nivel = st.columns([3, 2])
    capa = c_capa.radio("Colorear por", ["Riesgo", "🤖 Plan del agente", "Causa principal", "Perfil"],
                        horizontal=True)
    nivel = "Secciones"
    if edo_sel and not mpio_sel:
        nivel = c_nivel.radio("Detalle", ["Municipios (rápido)", "Secciones"], horizontal=True,
                              help="Por sección se dibujan miles de polígonos y tarda más. "
                                   "Elige un municipio en la barra lateral para ver sus secciones al instante.")
    plan_txt = (f"Plan del agente con presupuesto del {pct}%: "
                f"{len(acc):,} secciones con acción de {len(vista):,}.")

    if edo_sel == 0:
        geo, centro, zoom = geo_pais()
        agg = (datos.assign(_w=datos["riesgo"] * datos["LN_prev"])
               .groupby(["EDOCVE", "EDONOM"], as_index=False)
               .agg(_w=("_w", "sum"), LN=("LN_prev", "sum"), secciones=("SECCION", "size")))
        agg["riesgo"] = agg["_w"] / agg["LN"]
        a_e = acc.groupby("EDOCVE").agg(votantes=("votantes", "sum"), acciones=("acciones", "sum"))
        agg = agg.join(a_e, on="EDOCVE").fillna({"votantes": 0, "acciones": 0})
        if capa in ("Causa principal", "Perfil"):
            st.info("A nivel nacional el mapa muestra el riesgo por entidad. "
                    "Elige una entidad para ver causas y perfiles.")
        if geo is None:
            st.warning("Falta la cartografía (data/processed/geo). Genérala con `python -m src.geo`.")
            st.bar_chart(agg.set_index("EDONOM")["riesgo"].sort_values())
        elif capa == "🤖 Plan del agente":
            st.caption(plan_txt)
            st.plotly_chart(mapa(agg, geo, centro, zoom, "EDOCVE", "votantes",
                                 ["EDONOM", "acciones", "votantes", "riesgo"],
                                 "Votantes adicionales que espera el agente por entidad",
                                 barra="Votantes", formato=",.0f"), width="stretch")
        else:
            st.plotly_chart(mapa(agg, geo, centro, zoom, "EDOCVE", "riesgo", ["EDONOM", "secciones", "LN"],
                                 "Riesgo medio por entidad (ponderado por lista nominal)"),
                            width="stretch")
    elif nivel.startswith("Municipios"):
        geo, centro, zoom = geo_municipios(edo_sel)
        v_ = vista[vista["MPIOCVE"] > 0].assign(
            _w=lambda d: d["riesgo"] * d["LN_prev"], factor=lambda d: d["causa_principal"].map(NOMBRE_FACTOR))
        g = v_.groupby("MPIOCVE")
        mun = pd.DataFrame({"_w": g["_w"].sum(), "LN": g["LN_prev"].sum(), "secciones": g.size(),
                            "MPIONOM": g["MPIONOM"].agg(_moda), "factor": g["factor"].agg(_moda)})
        if "perfil_nombre" in v_:
            mun["perfil_nombre"] = g["perfil_nombre"].agg(_moda)
        mun["riesgo"] = mun["_w"] / mun["LN"]
        a_m = acc[acc["EDOCVE"] == edo_sel].groupby("MPIOCVE").agg(votantes=("votantes", "sum"),
                                                                   acciones=("acciones", "sum"))
        mun = mun.join(a_m).fillna({"votantes": 0, "acciones": 0}).reset_index()
        mun["clave"] = edo_sel * 1000 + mun["MPIOCVE"]
        if geo is None:
            st.warning("Falta la cartografía de municipios. Genérala con `python -m src.geo`.")
        elif capa == "🤖 Plan del agente":
            st.caption(plan_txt)
            st.plotly_chart(mapa(mun, geo, centro, zoom, "clave", "votantes",
                                 ["MPIONOM", "acciones", "votantes", "riesgo"],
                                 f"Votantes adicionales que espera el agente · {ENTIDADES[edo_sel]}",
                                 barra="Votantes", formato=",.0f"), width="stretch")
        else:
            color = {"Riesgo": "riesgo", "Causa principal": "factor", "Perfil": "perfil_nombre"}[capa]
            if color not in mun:
                st.warning("No hay perfiles cargados (perfiles_seccion_2024.parquet).")
            else:
                st.plotly_chart(mapa(mun, geo, centro, zoom, "clave", color,
                                     ["MPIONOM", "riesgo", "factor", "secciones"],
                                     f"{capa} por municipio · {ENTIDADES[edo_sel]}"
                                     + (" (la más frecuente)" if color != "riesgo" else ""),
                                     continuo=color == "riesgo"), width="stretch")
    else:
        geo, centro, zoom = geo_secciones(edo_sel, mpio_sel)
        df = vista.copy()
        df["factor"] = df["causa_principal"].map(NOMBRE_FACTOR)
        df = df.merge(acc[CLAVE + ["medida_agente"]], on=CLAVE, how="left")
        df["medida_agente"] = df["medida_agente"].fillna(SIN_ACCION)
        color = {"Riesgo": "riesgo", "🤖 Plan del agente": "medida_agente", "Causa principal": "factor",
                 "Perfil": "perfil_nombre"}[capa]
        if color not in df:
            st.warning("No hay perfiles cargados (perfiles_seccion_2024.parquet).")
        elif geo is None:
            st.warning("Falta la cartografía de secciones. Genérala con `python -m src.geo`.")
        else:
            if color == "medida_agente":
                st.caption(plan_txt + " Cada color es la medida principal que eligió; "
                           "el detalle de por qué está en 🤖 Agente.")
            hover = ["SECCION", "MPIONOM", "riesgo", "factor", "medida_agente"]
            st.plotly_chart(mapa(df, geo, centro, zoom, "clave", color, [h for h in hover if h in df],
                                 f"{capa.replace('🤖 ', '')} por sección · {zona}",
                                 continuo=color == "riesgo"), width="stretch")

    st.subheader("Secciones con mayor riesgo (lista nominal ≥ 50)")
    top = vista[vista["LN_prev"] >= 50].nlargest(50, "riesgo").copy()  # sin secciones diminutas
    top["causa principal"] = top["causa_principal"].map(NOMBRE_FACTOR)
    cols = ["EDONOM", "MPIONOM", "SECCION", "tipo", "LN_prev", "riesgo", "causa principal",
            "causa_1_pp", "perfil_nombre", "participacion_real"]
    st.dataframe(top[[c for c in cols if c in top]], hide_index=True, width="stretch",
                 column_config={"riesgo": st.column_config.ProgressColumn("riesgo", format="%.2f",
                                                                          min_value=0, max_value=1),
                                "causa_1_pp": st.column_config.NumberColumn("aporte (pp)", format="%.1f"),
                                "LN_prev": st.column_config.NumberColumn("lista nominal", format="%d")})

# --------------------------------------------------------------------------
# Sección
# --------------------------------------------------------------------------
if pantalla == VISTAS[2]:
    opciones = vista.sort_values("riesgo", ascending=False)
    if opciones.empty:
        st.info("No hay secciones con los filtros elegidos.")
    else:
        idx = st.selectbox(
            "Sección (las 1,000 de mayor riesgo)", opciones.index[:1000],
            format_func=lambda i: f"{ENTIDADES[opciones.at[i, 'EDOCVE']]} · sección "
                                  f"{opciones.at[i, 'SECCION']} · riesgo {opciones.at[i, 'riesgo']:.1%}")
        fila = opciones.loc[idx]
        a, b, c = st.columns(3)
        a.metric("Riesgo de abstención", f"{fila['riesgo']:.1%}",
                 (f"rango 80%: {fila['riesgo_min']:.0%}–{fila['riesgo_max']:.0%}"
                  if pd.notna(fila.get("riesgo_min", np.nan)) else None), delta_color="off")
        a.metric("Participación predicha", f"{fila['participacion_pred']:.1%}")
        if pd.notna(fila.get("participacion_real", np.nan)):
            b.metric("Participación real", f"{fila['participacion_real']:.1%}",
                     f"{(fila['participacion_real'] - fila['participacion_pred']) * 100:+.1f} pp vs predicción")
        b.metric("Lista nominal (elección previa)", f"{fila['LN_prev']:,.0f}")
        if pd.notna(fila.get("prob_riesgo_alto", np.nan)):
            c.metric("Prob. de estar en el 10% de mayor riesgo", f"{fila['prob_riesgo_alto']:.0%}")
        c.metric("Tipo de sección", fila["tipo"])
        if pd.notna(fila.get("perfil_nombre")):
            c.markdown(f"**Perfil:** {fila['perfil_nombre']}")

        shap = {k[5:]: v * 100 for k, v in fila.items() if k.startswith("shap_") and pd.notna(v)}
        s = (pd.DataFrame({"factor": [NOMBRE_FACTOR.get(k, k) for k in shap], "pp": list(shap.values())})
             .sort_values("pp"))
        base_n = {NOMBRE_FACTOR.get(f, f) for f in FACTORES_BASE}
        s["sentido"] = np.select([s["pp"] <= 0, s["factor"].isin(base_n)],
                                 ["reduce el riesgo", "nivel base (historial/contexto)"], "causa que eleva el riesgo")
        fig = px.bar(s, x="pp", y="factor", color="sentido", orientation="h",
                     color_discrete_map={"causa que eleva el riesgo": PALETA["primario"],
                                         "nivel base (historial/contexto)": PALETA["secundario"],
                                         "reduce el riesgo": PALETA["bajo"]},
                     title="¿Por qué? Aporte de cada factor al riesgo (puntos porcentuales, SHAP)")
        fig.update_layout(height=420, yaxis_title=None, xaxis_title="pp", legend_title=None)
        st.plotly_chart(fig, width="stretch")

        st.subheader("🤖 Qué decide el agente en esta sección")
        plan_s, _ = _plan_zona(anio, edo_sel, mpio_sel, pct)
        traza_agente(fila, plan_s, pct)

# --------------------------------------------------------------------------
# Perfiles
# --------------------------------------------------------------------------
if pantalla == VISTAS[3]:
    desc = cargar_json("perfiles_descripcion_2024.json")
    if desc is None or "perfil_nombre" not in datos:
        st.info("No hay perfiles disponibles.")
    else:
        st.caption(f"Perfiles ciudadanos (K-Means, k={desc['k']}) con la ventana "
                   f"{desc['ventana'][0]}–{desc['ventana'][-1]}.")
        for p in desc["perfiles"]:
            with st.expander(f"**{p['nombre']}** · {p['n_secciones']:,} secciones"):
                st.write(p["narrativa"])
        q = (vista[vista["perfil_nombre"] != "Sin dato"].groupby("perfil_nombre")["riesgo"]
             .quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack())
        fig = go.Figure(go.Box(y=q.index, q1=q[0.25], median=q[0.5], q3=q[0.75], lowerfence=q[0.05],
                               upperfence=q[0.95], orientation="h", marker_color=PALETA["secundario"],
                               fillcolor=PALETA["lavanda_media"], name="riesgo"))
        fig.update_layout(title=f"Riesgo {anio} por perfil (percentiles 5–95)", showlegend=False)
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(yaxis_title=None, height=380)
        st.plotly_chart(fig, width="stretch")
        mezcla = (vista[vista["perfil_nombre"] != "Sin dato"]
                  .assign(factor=lambda d: d["causa_principal"].map(NOMBRE_FACTOR))
                  .groupby(["perfil_nombre", "factor"]).size().rename("secciones").reset_index())
        fig = px.bar(mezcla, x="secciones", y="perfil_nombre", color="factor", orientation="h",
                     title="Causa principal de riesgo dentro de cada perfil")
        fig.update_layout(yaxis_title=None, height=420, legend_title=None)
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
# Agente
# --------------------------------------------------------------------------
if pantalla == VISTAS[4]:
    st.markdown(
        "Agente racional **basado en utilidad**. En cada sección repite el mismo ciclo: **percibe** el riesgo "
        "que predice el modelo, **diagnostica** qué lo causa, **evalúa** las medidas posibles y **decide** "
        "cuáles aplicar con el presupuesto. Efectos y costos son **supuestos ilustrativos** (ver `src/agent.py`).")
    plan, res = _plan_zona(anio, edo_sel, mpio_sel, pct)
    if plan.empty:
        st.info("No hay acciones candidatas para este filtro.")
    else:
        sel = plan[plan["seleccionada"]]
        corte = float(sel["utilidad"].min()) if len(sel) else float("nan")
        n_causa = int((vista["causa_1"] != "ninguna").sum())
        n_inercia = int((vista["causa_principal"] == INERCIA).sum())
        n_nada = len(vista) - n_causa - n_inercia
        n_secc_cand = plan[CLAVE].drop_duplicates().shape[0]
        st.markdown(
            "<div class='retos'>"
            f"<div class='reto r1'><h5>1 · Percibe</h5><p><b>{len(vista):,}</b> secciones: riesgo predicho por "
            "el modelo y aporte de cada factor (SHAP).</p></div>"
            f"<div class='reto r2'><h5>2 · Diagnostica</h5><p><b>{n_causa:,}</b> con una causa atendible · "
            f"<b>{n_inercia:,}</b> solo con baja participación crónica · <b>{n_nada:,}</b> sin nada que atender.</p></div>"
            f"<div class='reto r2'><h5>3 · Evalúa</h5><p><b>{len(plan):,}</b> acciones posibles en "
            f"<b>{n_secc_cand:,}</b> secciones. A cada una le calcula utilidad = votantes extra ÷ costo.</p></div>"
            f"<div class='reto r3'><h5>4 · Decide</h5><p>Ordena por utilidad y elige hasta gastar el <b>{pct}%</b>: "
            f"<b>{res['acciones']:,}</b> acciones en <b>{res['secciones_intervenidas']:,}</b> secciones "
            f"(entran las de utilidad ≥ {corte:.3f}).</p></div></div>",
            unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Acciones", f"{res['acciones']:,}")
        k2.metric("Secciones intervenidas", f"{res['secciones_intervenidas']:,}")
        k3.metric("Ciudadanos alcanzados", f"{res['ciudadanos_alcanzados']:,}")
        k4.metric("Votantes adicionales esperados", f"{res['votantes_adicionales_esperados']:,.0f}")

        a, b = st.columns(2)
        # Curva de decisión: acciones en orden de utilidad contra el costo acumulado
        paso = max(1, len(plan) // 4000)
        cur = plan.iloc[::paso].assign(estado=lambda d: np.where(d["seleccionada"], "Elegida", "No cabe"))
        f_c = px.scatter(cur, x="costo_acum", y="utilidad", color="estado",
                         color_discrete_map={"Elegida": PALETA["primario"], "No cabe": PALETA["lavanda_media"]},
                         title="Cómo decide: acciones ordenadas por utilidad")
        f_c.add_vline(x=res["presupuesto"], line_dash="dash", line_color=PALETA["secundario"],
                      annotation_text=f"presupuesto {pct}%")
        f_c.update_traces(marker_size=4)
        f_c.update_xaxes(title="costo acumulado")
        f_c.update_yaxes(title="utilidad (votantes / costo)")
        a.plotly_chart(_fig(f_c, 400), width="stretch")
        pm = pd.DataFrame(res["por_medida"])
        if not pm.empty:
            f_m = px.bar(pm.sort_values("votantes"), x="votantes", y="nombre", orientation="h",
                         color_discrete_sequence=[PALETA["primario"]], hover_data=["acciones", "costo"],
                         title="Votantes adicionales esperados por medida")
            f_m.update_layout(yaxis_title=None)
            b.plotly_chart(_fig(f_m, 400), width="stretch")

        if "participacion_real" in vista and len(sel):
            r_ = vista.set_index(CLAVE)
            elegidas = r_.index.isin(pd.MultiIndex.from_frame(sel[CLAVE].drop_duplicates()))
            st.caption(
                f"Validación ({anio}): abstención real en secciones elegidas "
                f"{1 - pond(r_[elegidas].reset_index(), 'participacion_real'):.1%} vs. resto "
                f"{1 - pond(r_[~elegidas].reset_index(), 'participacion_real'):.1%}.")

        # ---- Tabla explícita de lo que eligió
        st.markdown("#### Secciones que eligió el agente y por qué")
        ver_todas = st.toggle("Mostrar también las acciones que no cupieron en el presupuesto")
        t = (plan if ver_todas else sel).head(2000)
        t = t.merge(vista[CLAVE + ["MPIONOM"]] if "MPIONOM" in vista else vista[CLAVE], on=CLAVE, how="left")
        tabla = pd.DataFrame({
            "Prioridad": t["prioridad"],
            "Decisión": np.where(t["seleccionada"], "✅ Elegida", "❌ No cabe"),
            "Entidad": t["EDOCVE"].map(ENTIDADES),
            "Municipio": t["MPIONOM"] if "MPIONOM" in t else "—",
            "Sección": t["SECCION"],
            "Riesgo": t["riesgo"],
            "Causa detectada": t["factor"].map(NOMBRE_FACTOR),
            "Aporte de la causa (pp)": t["contribucion_pp"],
            "Medida": t["medida"].map(NOMBRE_MEDIDA),
            "Motivo": np.where(t["factor"] == INERCIA, "Sin causa atendible → diagnóstico",
                               np.where(t["orden_en_seccion"] == 1, "Causa principal",
                                        "2ª causa (rinde la mitad)")),
            "Efecto esperado (pp)": t["ganancia_pp"],
            "Votantes extra": t["votantes"],
            "Costo": t["costo"],
            "Utilidad": t["utilidad"],
        })
        st.dataframe(tabla, hide_index=True, width="stretch",
                     column_config={"Riesgo": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                    "Aporte de la causa (pp)": st.column_config.NumberColumn(format="%.1f"),
                                    "Efecto esperado (pp)": st.column_config.NumberColumn(format="%.2f"),
                                    "Votantes extra": st.column_config.NumberColumn(format="%.1f"),
                                    "Costo": st.column_config.NumberColumn(format="%.1f"),
                                    "Utilidad": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Efecto esperado = efecto máximo de la medida × min(1, aporte ÷ 4 pp), a la mitad si es la "
                   "2ª causa · Votantes extra = lista nominal × efecto ÷ 100 · Utilidad = votantes extra ÷ costo. "
                   "Se muestran hasta 2,000 filas.")
        st.download_button("⬇️ Descargar plan (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                           f"plan_intervencion_{anio}.csv", "text/csv")

        # ---- Razonamiento de una sección
        st.markdown("#### Sigue el razonamiento del agente en una sección")
        orden = plan.drop_duplicates(CLAVE).head(1000)
        sel_claves = set(map(tuple, sel[CLAVE].to_numpy().tolist()))
        opciones_a = list(map(tuple, orden[CLAVE].to_numpy().tolist()))
        clave_s = st.selectbox(
            "Sección (en orden de prioridad)", opciones_a,
            format_func=lambda c: f"{'✅' if c in sel_claves else '❌'} {ENTIDADES[int(c[0])]} · sección {int(c[1])}")
        fila_a = vista[(vista["EDOCVE"] == clave_s[0]) & (vista["SECCION"] == clave_s[1])]
        if len(fila_a):
            traza_agente(fila_a.iloc[0], plan, pct)

# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
if pantalla == VISTAS[5]:
    met = cargar_json("riesgo_metricas.json")
    if met is None:
        st.info("No hay métricas (riesgo_metricas.json).")
    else:
        NOMBRE_MODELO = {"baseline_comparable": "Baseline: repetir la elección comparable",
                         "baseline_anterior": "Baseline: repetir la elección anterior",
                         "ridge": "Ridge", "lasso": "Lasso", "lineal": "Lineal (Ridge)",
                         "gradient_boosting": "Gradient Boosting", "random_forest": "Random Forest",
                         "ensamble": "Ensamble", "ens_lineal_gb": "Ensamble lineal + GB"}
        COLS = {"MAE_rel_pp": "Error del dónde (pp)", "MAE_dentro_estado_pp": "Error dentro del estado (pp)",
                "spearman": "Corr. de rangos", "precision_top10": "Top 10%", "AUC_top10": "AUC top 10%",
                "MAE_pp": "Error total (pp)", "sesgo_pp": "Sesgo (pp)", "error_nacional_pp": "Error del nivel nacional (pp)"}
        elegido = met.get("modelo_elegido")
        ens = met.get("ensamble") or {}
        detalle = (f" ({NOMBRE_MODELO.get(ens.get('lineal'), ens.get('lineal'))} "
                   f"{ens.get('peso_lineal', 0):.0%} + Gradient Boosting {1 - ens.get('peso_lineal', 0):.0%})"
                   if elegido == "ensamble" and ens else "")
        criterio = met.get("criterio") or (
            f"menor {COLS.get(met['metrica_seleccion'], met['metrica_seleccion']).lower()} en validación "
            + " y ".join(str(v) for v in met.get("pliegues_seleccion", [2021]))
            if met.get("metrica_seleccion") else "")
        st.markdown(f"**Modelo elegido:** {NOMBRE_MODELO.get(elegido, elegido)}{detalle} — {criterio}.")
        st.caption("El **error del dónde** mide qué tan bien se predice la participación de cada sección "
                   "respecto al promedio del país: es lo que aprende el modelo. El **error total** suma además el "
                   "error del supuesto del nivel nacional, que es igual para todas las secciones y no depende del "
                   "modelo. Todo se elige con los años de validación; 2024 es una prueba que no se usó para elegir.")

        def tabla(bloque: dict) -> pd.DataFrame:
            t = pd.DataFrame({NOMBRE_MODELO.get(m, m): {COLS[c]: v.get(c) for c in COLS if c in v}
                              for m, v in bloque.items() if isinstance(v, dict)}).T
            return t.dropna(axis=1, how="all")

        bloques = [(k, v) for k, v in met.items()
                   if k.startswith(("validacion", "referencia", "prueba")) and isinstance(v, dict)]
        orden = {"prueba_2024": 0, "validacion_2021": 1, "validacion_promedio": 1}
        for nombre, bloque in sorted(bloques, key=lambda kv: orden.get(kv[0], 2)):
            titulo = {"prueba_2024": "Prueba 2024 (no vista)",
                      "validacion_promedio": "Validación (promedio de los pliegues)"}.get(
                nombre, nombre.replace("validacion_", "Validación ").replace("referencia_", "Referencia "))
            st.markdown(f"**{titulo}**")
            st.dataframe(tabla(bloque), width="stretch")

        nac = met.get("supuesto_nacional")
        if nac:
            st.markdown(f"**Supuesto del nivel nacional:** regla `{nac['regla_elegida']}` — "
                        + (nac.get("criterio") or "elegida con " + ", ".join(map(str, nac.get("anios_para_elegir", []))))
                        + ". Error en pp de cada regla en cada elección:")
            st.dataframe(pd.DataFrame(nac["error_pp_por_eleccion"]).T, width="stretch")

        top = met.get("top10")
        if top:
            st.markdown("**Detectar el 10% de mayor riesgo:** clasificador específico vs. ranking de la regresión")
            filas = {(b, m): v for b, d in top.items() for m, v in d.items()}
            st.dataframe(pd.DataFrame(filas).T.rename(columns=COLS), width="stretch")

        inter = met.get("intervalos")
        if inter:
            st.markdown(f"**Intervalos de predicción** (objetivo: {inter['cobertura_objetivo']:.0%} de las "
                        "secciones dentro de su rango)")
            t = pd.DataFrame({k: v for k, v in inter.items() if isinstance(v, dict)}).T.rename(columns={
                "cobertura_rel": "Cobertura (dónde)", "cobertura_abs": "Cobertura (total)",
                "ancho_medio_pp": "Ancho medio (pp)", "ajuste_conformal_pp": "Ajuste conformal (pp)",
                "calibrado_con": "Calibrado con"})
            st.dataframe(t, width="stretch")

        imp = met.get("importancia_factores_2024")
        if imp:
            s_ = pd.Series(imp).rename(index=NOMBRE_FACTOR).sort_values()
            fig = px.bar(s_, orientation="h", title="Importancia por factor (pp medio de |SHAP|, 2024)",
                         color_discrete_sequence=[PALETA["primario"]])
            fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="pp", height=440)
            st.plotly_chart(fig, width="stretch")
        st.caption("Top 10% = fracción del 10% de secciones con mayor abstención real que el modelo pone en su 10% "
                   "de mayor riesgo. AUC top 10% = qué tan bien separa ese grupo del resto (0.5 = azar, 1 = perfecto).")
