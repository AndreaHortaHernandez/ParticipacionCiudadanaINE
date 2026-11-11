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
import streamlit as st

from src.agent import (CATALOGO, CAUSAS, FACTOR_ESTATAL, FACTORES_BASE, INERCIA,
                       RECOMENDACION_ESTATAL, candidatas, causas, planificar,
                       recomendacion_estatal, recomendar_seccion, resumen)
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
    "territorio_acceso": "Tipo de sección y crecimiento del padrón", "contexto_municipal": "Contexto municipal",
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
                    "participacion_pred", "riesgo", "factor_1", "factor_1_pp", "factor_2",
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


@st.cache_data(show_spinner="Cargando cartografía…")
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


def mapa(df, geo, clave, color, hover, titulo, continuo=True):
    """Coroplético encuadrado a las geometrías mostradas (zoom al estado o municipio)."""
    centro, zoom = _encuadre(geo, ancho_px=1100, alto_px=ALTO_MAPA - 40)
    kw = dict(geojson=geo, locations=clave, featureidkey=f"properties.{clave}", color=color,
              hover_name=hover[0], hover_data=hover[1:], center=centro, zoom=zoom,
              opacity=0.8, title=titulo)
    if continuo:
        kw.update(color_continuous_scale=ESCALA_RIESGO)
    else:
        kw.update(color_discrete_sequence=SECUENCIA)
    fn = getattr(px, "choropleth_map", None)
    if fn is not None:
        fig = fn(df, map_style="carto-positron", **kw)
    else:
        fig = px.choropleth_mapbox(df, mapbox_style="carto-positron", **kw)
    fig.update_traces(marker_line_width=0.4, marker_line_color="#FFFFFF")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=ALTO_MAPA,
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
PRESUPUESTO_RESULTADOS = 10  # % del costo de todas las acciones candidatas


def _fig(fig, alto=360):
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=50, b=10), legend_title=None,
                      font=dict(color=PALETA["texto"], family=FUENTE),
                      title_font=dict(size=16, color=PALETA["primario"]))
    return fig


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


def pantalla_resultados(vista, datos, anio, zona, edo, mpio):
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
    plan, res = _plan_zona(anio, edo, mpio, PRESUPUESTO_RESULTADOS)

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
        texto += (f" Con un presupuesto del {PRESUPUESTO_RESULTADOS}%, el agente propone "
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
    h = px.histogram(v, x="riesgo", nbins=40, title="Distribución del riesgo de abstención",
                     color_discrete_sequence=[PALETA["primario"]])
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
        st.caption(f"Presupuesto: {PRESUPUESTO_RESULTADOS}% del costo de todas las acciones posibles. "
                   "Ajústalo en la pestaña 🤖 Agente. Efectos y costos son supuestos ilustrativos.")

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

tab_res, tab_mapa, tab_sec, tab_perf, tab_agente, tab_modelo = st.tabs(
    ["📊 Resultados", "🗺️ Mapa", "📍 Sección", "👥 Perfiles", "🤖 Agente de intervención", "⚙️ Modelo"])

with tab_res:
    pantalla_resultados(vista, datos, anio, zona, edo_sel, mpio_sel)

# --------------------------------------------------------------------------
# Mapa
# --------------------------------------------------------------------------
with tab_mapa:
    capa = st.radio("Colorear por", ["Riesgo", "Causa principal", "Perfil"], horizontal=True)
    if edo_sel == 0:
        geo = cargar_geo("entidades.geojson")
        agg = (datos.assign(_w=datos["riesgo"] * datos["LN_prev"])
               .groupby(["EDOCVE", "EDONOM"], as_index=False)
               .agg(_w=("_w", "sum"), LN=("LN_prev", "sum"), secciones=("SECCION", "size")))
        agg["riesgo"] = agg["_w"] / agg["LN"]
        if capa != "Riesgo":
            st.info("A nivel nacional el mapa muestra el riesgo por entidad. "
                    "Elige una entidad para ver factores y perfiles por sección.")
        if geo is None:
            st.warning("Falta la cartografía (data/processed/geo). Genérala con `python -m src.geo`.")
            st.bar_chart(agg.set_index("EDONOM")["riesgo"].sort_values())
        else:
            st.plotly_chart(mapa(agg, geo, "EDOCVE", "riesgo", ["EDONOM", "secciones", "LN"],
                                 "Riesgo medio por entidad (ponderado por lista nominal)"),
                            width="stretch")
    else:
        geo = cargar_geo(f"secciones_{edo_sel:02d}.geojson")
        df = vista.copy()
        df["factor"] = df["causa_principal"].map(NOMBRE_FACTOR)
        color = {"Riesgo": "riesgo", "Causa principal": "factor", "Perfil": "perfil_nombre"}[capa]
        if color not in df:
            st.warning("No hay perfiles cargados (perfiles_seccion_2024.parquet).")
        elif geo is None:
            st.warning("Falta la cartografía de secciones. Genérala con `python -m src.geo`.")
        else:
            claves = set(df["clave"])
            geo = {"type": "FeatureCollection",
                   "features": [f for f in geo["features"] if f["properties"]["clave"] in claves]}
            hover = ["SECCION", "MPIONOM", "riesgo", "factor"] if "MPIONOM" in df else ["SECCION", "riesgo", "factor"]
            st.plotly_chart(mapa(df, geo, "clave", color, hover,
                                 f"{capa} por sección · {ENTIDADES[edo_sel]}", continuo=color == "riesgo"),
                            width="stretch")

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
with tab_sec:
    opciones = vista.sort_values("riesgo", ascending=False)
    if opciones.empty:
        st.info("No hay secciones con los filtros elegidos.")
    else:
        idx = st.selectbox(
            "Sección (ordenadas por riesgo)", opciones.index[:5000],
            format_func=lambda i: f"{ENTIDADES[opciones.at[i, 'EDOCVE']]} · sección "
                                  f"{opciones.at[i, 'SECCION']} · riesgo {opciones.at[i, 'riesgo']:.1%}")
        fila = opciones.loc[idx]
        a, b, c = st.columns(3)
        a.metric("Riesgo de abstención", f"{fila['riesgo']:.1%}")
        a.metric("Participación predicha", f"{fila['participacion_pred']:.1%}")
        if pd.notna(fila.get("participacion_real", np.nan)):
            b.metric("Participación real", f"{fila['participacion_real']:.1%}",
                     f"{(fila['participacion_real'] - fila['participacion_pred']) * 100:+.1f} pp vs predicción")
        b.metric("Lista nominal (elección previa)", f"{fila['LN_prev']:,.0f}")
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

        st.subheader("Recomendación del agente")
        rec = recomendar_seccion(fila)
        if not rec:
            st.success("Ningún factor eleva el riesgo de esta sección: no se recomienda intervención.")
        for r_ in rec:
            st.markdown(
                f"- **{r_['nombre']}** — atiende *{NOMBRE_FACTOR.get(r_['factor'], r_['factor']).lower()}* "
                f"(aporta {r_['contribucion_pp']:.1f} pp)"
                + (f"; efecto supuesto ≈ +{r_['ganancia_pp']:.1f} pp" if r_.get("ganancia_pp") else ""))

# --------------------------------------------------------------------------
# Perfiles
# --------------------------------------------------------------------------
with tab_perf:
    desc = cargar_json("perfiles_descripcion_2024.json")
    if desc is None or "perfil_nombre" not in datos:
        st.info("No hay perfiles disponibles.")
    else:
        st.caption(f"Perfiles ciudadanos (K-Means, k={desc['k']}) con la ventana "
                   f"{desc['ventana'][0]}–{desc['ventana'][-1]}.")
        for p in desc["perfiles"]:
            with st.expander(f"**{p['nombre']}** · {p['n_secciones']:,} secciones"):
                st.write(p["narrativa"])
        fig = px.box(vista[vista["perfil_nombre"] != "Sin dato"], x="riesgo", y="perfil_nombre",
                     points=False, title=f"Riesgo {anio} por perfil")
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
with tab_agente:
    st.markdown(
        "Agente **basado en utilidad**: para cada sección propone medidas que atienden sus "
        "factores de riesgo principales, estima votantes adicionales por unidad de costo y "
        "elige acciones en orden de utilidad hasta agotar el presupuesto. "
        "Efectos y costos son **supuestos ilustrativos** (ver `src/agent.py`).")
    cand = cargar_candidatas(anio)
    if edo_sel:
        cand = cand[cand["EDOCVE"] == edo_sel]
        if mpio_sel:
            cand = cand[cand["MPIOCVE"] == mpio_sel]
    if cand.empty:
        st.info("No hay acciones candidatas para este filtro.")
    else:
        pct = st.slider("Presupuesto (% del costo de aplicar todas las acciones candidatas)",
                        1, 100, 10)
        presupuesto = cand["costo"].sum() * pct / 100
        plan = planificar(cand, presupuesto)
        res = resumen(plan, presupuesto)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Acciones", f"{res['acciones']:,}")
        k2.metric("Secciones intervenidas", f"{res['secciones_intervenidas']:,}")
        k3.metric("Ciudadanos alcanzados", f"{res['ciudadanos_alcanzados']:,}")
        k4.metric("Votantes adicionales esperados", f"{res['votantes_adicionales_esperados']:,.0f}")

        pm = pd.DataFrame(res["por_medida"])
        if not pm.empty:
            fig = px.bar(pm.sort_values("votantes"), x="votantes", y="nombre", orientation="h",
                         hover_data=["acciones", "costo"], title="Votantes adicionales esperados por medida")
            fig.update_layout(yaxis_title=None, height=380)
            st.plotly_chart(fig, width="stretch")

        sel = plan[plan["seleccionada"]].copy()
        if "participacion_real" in vista and len(sel):
            r_ = vista.set_index(CLAVE)
            elegidas = r_.index.isin(pd.MultiIndex.from_frame(sel[CLAVE].drop_duplicates()))
            st.caption(
                f"Validación ({anio}): abstención real en secciones elegidas "
                f"{1 - pond(r_[elegidas].reset_index(), 'participacion_real'):.1%} vs. resto "
                f"{1 - pond(r_[~elegidas].reset_index(), 'participacion_real'):.1%}.")

        sel["medida"] = sel["medida"].map(NOMBRE_MEDIDA)
        sel["factor"] = sel["factor"].map(NOMBRE_FACTOR)
        sel["entidad"] = sel["EDOCVE"].map(ENTIDADES)
        cols = ["prioridad", "entidad", "SECCION", "medida", "factor", "riesgo", "LN_prev",
                "ganancia_pp", "votantes", "costo", "utilidad"]
        st.dataframe(sel[cols].head(1000), hide_index=True, width="stretch",
                     column_config={"riesgo": st.column_config.NumberColumn(format="%.3f"),
                                    "ganancia_pp": st.column_config.NumberColumn("ganancia (pp)", format="%.2f"),
                                    "votantes": st.column_config.NumberColumn(format="%.1f"),
                                    "costo": st.column_config.NumberColumn(format="%.1f"),
                                    "utilidad": st.column_config.NumberColumn(format="%.3f")})
        st.download_button("Descargar plan (CSV)", sel[cols].to_csv(index=False).encode("utf-8"),
                           f"plan_intervencion_{anio}.csv", "text/csv")

# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
with tab_modelo:
    met = cargar_json("riesgo_metricas.json")
    if met is None:
        st.info("No hay métricas (riesgo_metricas.json).")
    else:
        st.markdown(f"**Modelo elegido:** `{met.get('modelo_elegido')}` — {met.get('criterio', '')}")
        cols = ["MAE_pp", "MAE_rel_pp", "MAE_dentro_estado_pp", "RMSE_pp", "sesgo_pp",
                "spearman", "precision_top10"]
        # Compatible con ambos formatos de salida de src.risk_model
        bloques = [(k, v) for k, v in met.items()
                   if k.startswith(("validacion", "referencia", "prueba")) and isinstance(v, dict)]
        for nombre, bloque in bloques:
            t = pd.DataFrame({m: {c: v.get(c) for c in cols} for m, v in bloque.items()
                              if isinstance(v, dict)}).T
            st.markdown(f"**{nombre.replace('_', ' ').capitalize()}**")
            st.dataframe(t, width="stretch")
        imp = met.get("importancia_factores_2024")
        if imp:
            s = pd.Series(imp).rename(index=NOMBRE_FACTOR).sort_values()
            fig = px.bar(s, orientation="h", title="Importancia por factor (pp medio de |SHAP|, 2024)")
            fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title="pp", height=420)
            st.plotly_chart(fig, width="stretch")
        st.caption("MAE en puntos porcentuales; spearman = correlación de rangos entre riesgo "
                   "predicho y real; precision_top10 = fracción del 10% de secciones con mayor riesgo real "
                   "que captura el 10% de mayor riesgo predicho.")
