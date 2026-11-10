"""Ingesta y limpieza de los Conteos Censales de Participación Ciudadana (INE).

Construye un panel  sección × elección  a partir de los CSV crudos
(sección × sexo × edad) de data/raw/<año>/.

Decisiones (ver README):
- Unidad territorial: sección electoral, clave (EDOCVE, SECCION).
- Participación = SV / (SV + NV). Los registros NS (cuadernillo sin marcas o no
  disponible) no se cuentan como votantes ni como abstencionistas.
- Si NS/LN > NS_MAX en una sección-elección, su participación se marca como
  faltante (NaN) solo para esa elección; la sección no se descarta.
- Estructura sociodemográfica disponible en la fuente: edad, sexo y tipo de
  sección (urbana / mixta / rural). No se añaden variables externas.

Uso:  python -m src.ingest
Salida: data/processed/panel_seccion.parquet y data/processed/ingest_report.json
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")
GEO_SECCION = os.path.join(ROOT, "data", "geo", "SECCION", "SECCIONES_26", "SECCION_26.shp")

ELECCIONES = [2009, 2012, 2015, 2018, 2021, 2024]
TIPO_ELECCION = {2009: "intermedia", 2012: "presidencial", 2015: "intermedia",
                 2018: "presidencial", 2021: "intermedia", 2024: "presidencial"}
NS_MAX = 0.5          # umbral de registros no especificados para considerar válida la participación
EDAD_MAX = 100        # edades > EDAD_MAX se consideran error de padrón para la estructura etaria
GRUPOS_EDAD = {"18_24": (18, 24), "25_34": (25, 34), "35_49": (35, 49),
               "50_64": (50, 64), "65p": (65, EDAD_MAX)}
CONTEOS = ["LN", "SV", "NV", "NS"]
CLAVE = ["EDOCVE", "SECCION"]


def _leer_csv(path: str) -> pd.DataFrame:
    cols = ["AELEC", "EDOCVE", "MPIOCVE", "MPIONOM", "SECCION", "TIPOSEC",
            "DEF", "DEL", "SEXO", "EDAD"] + CONTEOS
    for enc in ("utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, usecols=cols, encoding=enc,
                             dtype={"MPIONOM": "string", "TIPOSEC": "string",
                                    "DEF": "string", "DEL": "string"})
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"No se pudo decodificar {path}")
    # DEF / DEL vienen como " " en algunos archivos 2009/2012
    for c in ("DEF", "DEL"):
        df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce").astype("Int16")
    df["TIPOSEC"] = df["TIPOSEC"].str.strip()
    return df


def _agregar_archivo(path: str) -> pd.DataFrame:
    """Agrega un CSV entidad-año a nivel sección con desagregados por edad y sexo."""
    df = _leer_csv(path)
    keys = ["AELEC"] + CLAVE

    base = df.groupby(keys, as_index=False).agg(
        MPIOCVE=("MPIOCVE", "first"), MPIONOM=("MPIONOM", "first"),
        TIPOSEC=("TIPOSEC", "first"), DEF=("DEF", "first"), DEL=("DEL", "first"),
        **{c: (c, "sum") for c in CONTEOS})

    # Grupos: edad y sexo. SEXO=2 (no binario, solo 2024, ~100 personas) entra solo en totales.
    grupos = {f"e{g}": df["EDAD"].between(lo, hi) for g, (lo, hi) in GRUPOS_EDAD.items()}
    grupos["hom"] = df["SEXO"] == 0
    grupos["muj"] = df["SEXO"] == 1
    for nombre, mask in grupos.items():
        sub = df.loc[mask].groupby(keys)[["LN", "SV", "NV"]].sum()
        sub.columns = [f"{c}_{nombre}" for c in sub.columns]
        base = base.merge(sub, left_on=keys, right_index=True, how="left")

    # Edad media de la lista nominal (con tope EDAD_MAX)
    e = df.assign(EDADc=df["EDAD"].clip(upper=EDAD_MAX))
    e["w"] = e["EDADc"] * e["LN"]
    em = e.groupby(keys)[["w", "LN"]].sum()
    base = base.merge((em["w"] / em["LN"]).rename("edad_media"),
                      left_on=keys, right_index=True, how="left")
    return base


def construir_panel(raw_dir: str = RAW_DIR, n_jobs: int = 8) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*", "*", "*.csv")))
    if not files:
        raise FileNotFoundError(f"No hay CSV en {raw_dir}")
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        partes = list(ex.map(_agregar_archivo, files))
    p = pd.concat(partes, ignore_index=True)

    num_cols = [c for c in p.columns if c.startswith(("LN", "SV", "NV", "NS"))]
    p[num_cols] = p[num_cols].fillna(0).astype("int64")
    p = p.rename(columns={"AELEC": "eleccion"})
    p["tipo_eleccion"] = p["eleccion"].map(TIPO_ELECCION)

    # --- Variable objetivo y validez ---
    p["ns_share"] = p["NS"] / p["LN"]
    emitidos = p["SV"] + p["NV"]
    p["valido"] = (p["ns_share"] <= NS_MAX) & (emitidos > 0)
    p["participacion"] = np.where(p["valido"], p["SV"] / emitidos.where(emitidos > 0), np.nan)
    p["abstencion"] = 1 - p["participacion"]

    # --- Estructura sociodemográfica (% de la lista nominal) ---
    for g in GRUPOS_EDAD:
        p[f"pct_e{g}"] = p[f"LN_e{g}"] / p["LN"]
        em = p[f"SV_e{g}"] + p[f"NV_e{g}"]
        p[f"part_e{g}"] = np.where(p["valido"] & (em > 0), p[f"SV_e{g}"] / em.where(em > 0), np.nan)
    p["pct_mujeres"] = p["LN_muj"] / p["LN"]
    for s in ("hom", "muj"):
        em = p[f"SV_{s}"] + p[f"NV_{s}"]
        p[f"part_{s}"] = np.where(p["valido"] & (em > 0), p[f"SV_{s}"] / em.where(em > 0), np.nan)
    p["brecha_joven"] = p["part_e18_24"] - p["participacion"]
    p["brecha_genero"] = p["part_muj"] - p["part_hom"]

    # --- Cobertura temporal de la sección ---
    n_elec = p.groupby(CLAVE)["eleccion"].transform("nunique")
    p["en_todas"] = n_elec == len(ELECCIONES)
    p["n_validas"] = p.groupby(CLAVE)["valido"].transform("sum")

    # Nombre de municipio / tipo de sección más reciente para mostrar en la app
    ult = p.sort_values("eleccion").groupby(CLAVE).tail(1).set_index(CLAVE)
    p = p.merge(ult[["MPIOCVE", "MPIONOM", "TIPOSEC"]].add_suffix("_actual"),
                left_on=CLAVE, right_index=True, how="left")

    p["EDOCVE"] = p["EDOCVE"].astype("int16")
    p["SECCION"] = p["SECCION"].astype("int32")
    p["MPIOCVE"] = p["MPIOCVE"].astype("int16")
    return p.sort_values(CLAVE + ["eleccion"]).reset_index(drop=True)


def cobertura_geo(panel: pd.DataFrame, shp: str = GEO_SECCION) -> dict:
    """Cuántas secciones del panel tienen polígono en la cartografía INE 2025."""
    if not os.path.exists(shp):
        return {"disponible": False}
    import pyogrio
    g = pyogrio.read_dataframe(shp, columns=["entidad", "municipio", "seccion"],
                               read_geometry=False)
    gk = set(zip(g.entidad, g.seccion))
    out = {"disponible": True, "poligonos": int(len(g)),
           "claves_duplicadas_shp": int(g.duplicated(["entidad", "seccion"]).sum())}
    for y in ELECCIONES:
        k = panel.loc[panel.eleccion == y, CLAVE].drop_duplicates()
        m = np.array([(a, b) in gk for a, b in zip(k.EDOCVE, k.SECCION)])
        out[f"match_{y}"] = round(float(m.mean()), 4)
    k = panel.loc[panel.en_todas, CLAVE].drop_duplicates()
    out["match_en_todas"] = round(float(np.mean([(a, b) in gk for a, b in zip(k.EDOCVE, k.SECCION)])), 4)
    # ¿Coincide el municipio del shp con el del dato 2024?
    p24 = panel.loc[panel.eleccion == 2024, ["EDOCVE", "SECCION", "MPIOCVE"]]
    mm = p24.merge(g, left_on=["EDOCVE", "SECCION"], right_on=["entidad", "seccion"])
    out["municipio_coincide_2024"] = round(float((mm.MPIOCVE == mm.municipio).mean()), 4)
    return out


def reporte(p: pd.DataFrame) -> dict:
    r = {"filas": int(len(p)), "secciones": int(p[CLAVE].drop_duplicates().shape[0]),
         "secciones_en_todas": int(p.loc[p.en_todas, CLAVE].drop_duplicates().shape[0]),
         "ns_max": NS_MAX, "por_eleccion": {}}
    for y, d in p.groupby("eleccion"):
        r["por_eleccion"][int(y)] = {
            "secciones": int(len(d)), "validas": int(d.valido.sum()),
            "pct_validas": round(float(d.valido.mean()), 4),
            "participacion_nacional": round(float(d.SV.sum() / (d.SV.sum() + d.NV.sum())), 4),
            "participacion_mediana_seccion": round(float(d.participacion.median()), 4)}
    s = p[p.en_todas].groupby(CLAVE)["n_validas"].first()
    r["en_todas_por_n_validas"] = {int(k): int(v) for k, v in s.value_counts().sort_index().items()}
    return r


def cargar_panel() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(PROC_DIR, "panel_seccion.parquet"))


def main() -> None:
    os.makedirs(PROC_DIR, exist_ok=True)
    panel = construir_panel()
    panel.to_parquet(os.path.join(PROC_DIR, "panel_seccion.parquet"), index=False)
    rep = reporte(panel)
    rep["cartografia"] = cobertura_geo(panel)
    with open(os.path.join(PROC_DIR, "ingest_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print("\nColumnas:", list(panel.columns))


if __name__ == "__main__":
    main()
