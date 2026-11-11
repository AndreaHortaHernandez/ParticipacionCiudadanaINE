"""Agente de intervención: recomienda y prioriza medidas contra la abstención.

Agente racional basado en utilidad
----------------------------------
- Percepción: salida del modelo de riesgo (`riesgo_<año>.parquet`): riesgo de
  abstención por sección, padrón (LN_prev) y contribución de cada factor al
  riesgo (SHAP agregado, en puntos porcentuales).
- Acciones: catálogo de medidas (`CATALOGO`). Cada medida atiende uno o más
  factores de riesgo y tiene un costo y un efecto máximo esperado.
- Utilidad: votantes adicionales esperados por unidad de costo.
- Restricción: presupuesto total (en unidades de costo relativas).
- Política: para cada sección se proponen medidas para sus factores principales
  (factor_1..factor_3 con contribución > 0). Una segunda medida en la misma
  sección rinde la mitad (rendimientos decrecientes). Las acciones se ordenan
  por utilidad y se eligen mientras alcance el presupuesto (greedy de mochila
  fraccional sin fraccionar: prefijo de mayor razón votantes/costo).

Supuestos (explícitos y configurables)
--------------------------------------
Efectos y costos son ILUSTRATIVOS: órdenes de magnitud inspirados en la
literatura de experimentos de movilización del voto (el contacto cara a cara
rinde más que los recordatorios masivos, y cuesta más). No son estimaciones
causales con estos datos; el modelo solo identifica DÓNDE y POR QUÉ hay riesgo.

    ganancia_pp = efecto_max_pp · min(1, contribución_factor_pp / ESCALA_PP)
    votantes    = LN_prev · ganancia_pp / 100   (acotado por los abstencionistas esperados)
    costo       = costo_fijo + costo_por_ciudadano · LN_prev

Uso:  python -m src.agent --anio 2027 --presupuesto-pct 10
Salida: data/processed/intervenciones_<año>.parquet y agente_resumen_<año>.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.ingest import CLAVE, PROC_DIR

ESCALA_PP = 5.0          # contribución (pp) a partir de la cual la medida rinde su efecto máximo
RENDIMIENTO_SEGUNDA = 0.5
MAX_MEDIDAS_SECCION = 2

# medida -> factores que atiende, efecto máximo (pp de participación), costos relativos
CATALOGO: dict[str, dict] = {
    "brigadas_casa_por_casa": {
        "nombre": "Brigadas de promoción del voto casa por casa",
        "factores": ["historial_bajo", "contexto_municipal"],
        "efecto_max_pp": 3.0, "costo_fijo": 20.0, "costo_por_ciudadano": 0.20,
    },
    "campania_intermedias": {
        "nombre": "Campaña informativa sobre la importancia de las elecciones intermedias",
        "factores": ["desmovilizacion_intermedias"],
        "efecto_max_pp": 1.5, "costo_fijo": 5.0, "costo_por_ciudadano": 0.03,
    },
    "dialogo_vecinal": {
        "nombre": "Diagnóstico participativo y reuniones vecinales",
        "factores": ["caida_reciente"],
        "efecto_max_pp": 2.0, "costo_fijo": 15.0, "costo_por_ciudadano": 0.05,
    },
    "recordatorios": {
        "nombre": "Recordatorios cercanos a la jornada (SMS, redes, perifoneo)",
        "factores": ["volatilidad", "historial_bajo"],
        "efecto_max_pp": 0.8, "costo_fijo": 2.0, "costo_por_ciudadano": 0.01,
    },
    "programa_jovenes": {
        "nombre": "Programa para jóvenes en escuelas, universidades y redes sociales",
        "factores": ["jovenes"],
        "efecto_max_pp": 2.0, "costo_fijo": 10.0, "costo_por_ciudadano": 0.04,
    },
    "accesibilidad_mayores": {
        "nombre": "Accesibilidad de casillas y acompañamiento a personas mayores",
        "factores": ["envejecimiento"],
        "efecto_max_pp": 1.5, "costo_fijo": 10.0, "costo_por_ciudadano": 0.05,
    },
    "enfoque_genero": {
        "nombre": "Campaña con enfoque de género y apoyo en tareas de cuidado",
        "factores": ["genero"],
        "efecto_max_pp": 1.0, "costo_fijo": 5.0, "costo_por_ciudadano": 0.03,
    },
    "acceso_territorial": {
        "nombre": "Información de ubicación de casillas y apoyo de traslado (rural/mixta)",
        "factores": ["territorio_acceso"],
        "efecto_max_pp": 2.0, "costo_fijo": 15.0, "costo_por_ciudadano": 0.08,
    },
    "coordinacion_ople": {
        "nombre": "Campaña coordinada con el OPLE estatal (sin elección local concurrente)",
        "factores": ["concurrencia_local", "contexto_estatal"],
        "efecto_max_pp": 1.0, "costo_fijo": 3.0, "costo_por_ciudadano": 0.02,
    },
}
# Factores sin medida de movilización: se marcan para revisión, no consumen presupuesto.
REVISION = {"calidad_dato": "Revisar calidad del dato (registros no especificados en la elección previa)"}

MEDIDAS_DE_FACTOR: dict[str, list[str]] = {}
for _m, _d in CATALOGO.items():
    for _f in _d["factores"]:
        MEDIDAS_DE_FACTOR.setdefault(_f, []).append(_m)


def cargar_riesgo(anio: int) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(PROC_DIR, f"riesgo_{anio}.parquet"))


def candidatas(r: pd.DataFrame) -> pd.DataFrame:
    """Acciones posibles (sección × medida) con su ganancia, costo y utilidad."""
    filas = []
    ln = r["LN_prev"].fillna(0).to_numpy(float)
    abst_esp = ln * r["riesgo"].fillna(0).to_numpy(float)
    for i in (1, 2, 3):
        fac = r[f"factor_{i}"].to_numpy()
        pp = r[f"factor_{i}_pp"].fillna(0).to_numpy(float)
        for f, medidas in MEDIDAS_DE_FACTOR.items():
            m = (fac == f) & (pp > 0) & (ln > 0)
            if not m.any():
                continue
            for med in medidas:
                d = CATALOGO[med]
                g = d["efecto_max_pp"] * np.minimum(1.0, pp[m] / ESCALA_PP)
                filas.append(pd.DataFrame({
                    "idx": np.flatnonzero(m), "medida": med, "factor": f, "rango_factor": i,
                    "contribucion_pp": pp[m], "ganancia_pp": g,
                    "votantes": np.minimum(ln[m] * g / 100, abst_esp[m]),
                    "costo": d["costo_fijo"] + d["costo_por_ciudadano"] * ln[m],
                }))
    if not filas:
        return pd.DataFrame()
    c = pd.concat(filas, ignore_index=True)
    # Una medida que atiende varios factores de la misma sección cuenta una vez (su mejor caso)
    c = c.sort_values("votantes", ascending=False).drop_duplicates(["idx", "medida"])
    c["utilidad"] = c["votantes"] / c["costo"]
    # Rendimientos decrecientes dentro de la sección
    c = c.sort_values(["idx", "utilidad"], ascending=[True, False])
    c["orden_en_seccion"] = c.groupby("idx").cumcount() + 1
    c = c[c["orden_en_seccion"] <= MAX_MEDIDAS_SECCION].copy()
    seg = c["orden_en_seccion"] == 2
    c.loc[seg, "votantes"] *= RENDIMIENTO_SEGUNDA
    c["utilidad"] = c["votantes"] / c["costo"]
    for col in CLAVE + ["MPIOCVE", "riesgo", "LN_prev"]:
        if col in r.columns:
            c[col] = r[col].to_numpy()[c["idx"].to_numpy()]
    return c.drop(columns="idx").reset_index(drop=True)


def planificar(cand: pd.DataFrame, presupuesto: float) -> pd.DataFrame:
    """Elige acciones por mayor utilidad hasta agotar el presupuesto."""
    if cand.empty:
        return cand.assign(seleccionada=pd.Series(dtype=bool))
    c = cand.sort_values("utilidad", ascending=False).reset_index(drop=True)
    c["costo_acum"] = c["costo"].cumsum()
    c["seleccionada"] = c["costo_acum"] <= presupuesto
    c["prioridad"] = np.arange(1, len(c) + 1)
    return c


def resumen(plan: pd.DataFrame, presupuesto: float) -> dict:
    s = plan[plan["seleccionada"]]
    por_medida = (s.groupby("medida").agg(acciones=("costo", "size"), votantes=("votantes", "sum"),
                                          costo=("costo", "sum"))
                  .sort_values("votantes", ascending=False))
    por_medida["nombre"] = [CATALOGO[m]["nombre"] for m in por_medida.index]
    return {
        "presupuesto": round(float(presupuesto), 2),
        "costo_total_candidatas": round(float(plan["costo"].sum()), 2),
        "costo_usado": round(float(s["costo"].sum()), 2),
        "acciones": int(len(s)),
        "secciones_intervenidas": int(s[CLAVE].drop_duplicates().shape[0]) if len(s) else 0,
        "ciudadanos_alcanzados": int(s.drop_duplicates(CLAVE)["LN_prev"].sum()) if len(s) else 0,
        "votantes_adicionales_esperados": round(float(s["votantes"].sum()), 0),
        "por_medida": por_medida.round(2).reset_index().to_dict("records"),
    }


def recomendar_seccion(fila: pd.Series) -> list[dict]:
    """Recomendación explicada para una sección (fila de riesgo_<año>)."""
    out = []
    for i in (1, 2, 3):
        f, pp = fila.get(f"factor_{i}"), fila.get(f"factor_{i}_pp")
        if f is None or f == "ninguno" or pd.isna(pp) or pp <= 0:
            continue
        if f in REVISION:
            out.append({"factor": f, "contribucion_pp": float(pp), "medida": None,
                        "nombre": REVISION[f]})
        for med in MEDIDAS_DE_FACTOR.get(f, []):
            out.append({"factor": f, "contribucion_pp": float(pp), "medida": med,
                        "nombre": CATALOGO[med]["nombre"],
                        "ganancia_pp": CATALOGO[med]["efecto_max_pp"] * min(1.0, pp / ESCALA_PP)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anio", type=int, default=2027)
    ap.add_argument("--presupuesto-pct", type=float, default=10.0,
                    help="presupuesto como %% del costo de aplicar todas las acciones candidatas")
    a = ap.parse_args()

    cand = candidatas(cargar_riesgo(a.anio))
    presupuesto = cand["costo"].sum() * a.presupuesto_pct / 100
    plan = planificar(cand, presupuesto)
    res = resumen(plan, presupuesto)
    res.update({"anio": a.anio, "presupuesto_pct": a.presupuesto_pct,
                "supuestos": {"escala_pp": ESCALA_PP, "rendimiento_segunda": RENDIMIENTO_SEGUNDA,
                              "catalogo": CATALOGO}})
    plan.to_parquet(os.path.join(PROC_DIR, f"intervenciones_{a.anio}.parquet"), index=False)
    with open(os.path.join(PROC_DIR, f"agente_resumen_{a.anio}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "supuestos"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
