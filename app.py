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

from src.agent import CATALOGO, candidatas, planificar, recomendar_seccion, resumen
from src.ingest import CLAVE, PROC_DIR
from src.risk_model import DESCRIPCION_FACTOR

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
    "historial_bajo": "Historial bajo", "desmovilizacion_intermedias": "Desmovilización en intermedias",
    "caida_reciente": "Caída reciente", "volatilidad": "Volatilidad", "jovenes": "Jóvenes",
    "envejecimiento": "Estructura de edad", "genero": "Brecha de género",
    "territorio_acceso": "Territorio / acceso", "contexto_municipal": "Contexto municipal",
    "contexto_estatal": "Contexto estatal", "concurrencia_local": "Sin concurrencia local",
    "calidad_dato": "Calidad del dato", "ninguno": "Ninguno",
}
NOMBRE_MEDIDA = {m: d["nombre"] for m, d in CATALOGO.items()}

st.set_page_config(page_title="Riesgo de abstención", page_icon="🗳️", layout="wide")


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
    r["clave"] = r["EDOCVE"] * 100000 + r["SECCION"]
    # Sin pd.NA ni tipos "nullable": Plotly/Streamlit no los serializan (TypeError NAType)
    for c in r.columns:
        if c.startswith("factor_") and not c.endswith("_pp") or c == "factor_dominante":
            r[c] = r[c].astype(object).where(r[c].notna(), "ninguno").astype(str)
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


def mapa(df, geo, clave, color, hover, titulo, continuo=True):
    centro = {"lat": 23.6, "lon": -102.5}
    zoom = 3.8
    if clave == "clave":  # mapa de secciones: centrar en las secciones mostradas
        xs = [c for f in geo["features"][:300] for c in _coords(f["geometry"])]
        if xs:
            lon, lat = np.array(xs).T
            centro, zoom = {"lat": float(lat.mean()), "lon": float(lon.mean())}, 6.5
    kw = dict(geojson=geo, locations=clave, featureidkey=f"properties.{clave}", color=color,
              hover_name=hover[0], hover_data=hover[1:], center=centro, zoom=zoom,
              opacity=0.75, title=titulo)
    if continuo:
        kw.update(color_continuous_scale="OrRd")
    fn = getattr(px, "choropleth_map", None)
    if fn is not None:
        fig = fn(df, map_style="carto-positron", **kw)
    else:
        fig = px.choropleth_mapbox(df, mapbox_style="carto-positron", **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=620)
    return fig


def _coords(g):
    t, c = g["type"], g["coordinates"]
    if t == "Polygon":
        return c[0][::10]
    if t == "MultiPolygon":
        return [p for poly in c for p in poly[0][::10]]
    return []


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
st.title(f"Riesgo de abstención por sección electoral · {anio}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Secciones", f"{len(vista):,}")
c2.metric("Lista nominal", f"{vista['LN_prev'].sum():,.0f}")
c3.metric("Riesgo medio (pond. LN)", f"{pond(vista, 'riesgo'):.1%}")
if "participacion_real" in vista:
    c4.metric("Abstención real (pond. LN)", f"{1 - pond(vista, 'participacion_real'):.1%}")
else:
    c4.metric("Abstenc. esperada (personas)", f"{(vista['riesgo'] * vista['LN_prev']).sum():,.0f}")

tab_mapa, tab_sec, tab_perf, tab_agente, tab_modelo = st.tabs(
    ["Mapa", "Sección", "Perfiles", "Agente de intervención", "Modelo"])

# --------------------------------------------------------------------------
# Mapa
# --------------------------------------------------------------------------
with tab_mapa:
    capa = st.radio("Colorear por", ["Riesgo", "Factor dominante", "Perfil"], horizontal=True)
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
        df["factor"] = df["factor_dominante"].map(NOMBRE_FACTOR)
        color = {"Riesgo": "riesgo", "Factor dominante": "factor", "Perfil": "perfil_nombre"}[capa]
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
    top["factor principal"] = top["factor_1"].map(NOMBRE_FACTOR)
    cols = ["EDONOM", "MPIONOM", "SECCION", "tipo", "LN_prev", "riesgo", "factor principal",
            "factor_1_pp", "perfil_nombre", "participacion_real"]
    st.dataframe(top[[c for c in cols if c in top]], hide_index=True, width="stretch",
                 column_config={"riesgo": st.column_config.ProgressColumn("riesgo", format="%.2f",
                                                                          min_value=0, max_value=1),
                                "factor_1_pp": st.column_config.NumberColumn("aporte (pp)", format="%.1f"),
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
        s["sentido"] = np.where(s["pp"] > 0, "aumenta el riesgo", "reduce el riesgo")
        fig = px.bar(s, x="pp", y="factor", color="sentido", orientation="h",
                     color_discrete_map={"aumenta el riesgo": "#c0392b", "reduce el riesgo": "#2e86c1"},
                     title="¿Por qué? Aporte de cada factor al riesgo (puntos porcentuales, SHAP)")
        fig.update_layout(height=420, yaxis_title=None, xaxis_title="pp", legend_title=None)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Recomendación del agente")
        rec = recomendar_seccion(fila)
        if not rec:
            st.success("Ningún factor eleva el riesgo de esta sección: no se recomienda intervención.")
        for r_ in rec:
            st.markdown(
                f"- **{r_['nombre']}** — atiende *{DESCRIPCION_FACTOR.get(r_['factor'], r_['factor'])}* "
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
                  .assign(factor=lambda d: d["factor_dominante"].map(NOMBRE_FACTOR))
                  .groupby(["perfil_nombre", "factor"]).size().rename("secciones").reset_index())
        fig = px.bar(mezcla, x="secciones", y="perfil_nombre", color="factor", orientation="h",
                     title="Factor dominante de riesgo dentro de cada perfil")
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
