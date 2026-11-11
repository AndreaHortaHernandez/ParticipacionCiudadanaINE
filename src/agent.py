"""Agente de intervención: diagnostica la causa del riesgo y prioriza medidas.

Agente racional basado en utilidad
----------------------------------
- Percepción: salida del modelo de riesgo (`riesgo_<año>.parquet`): riesgo de
  abstención por sección, padrón (LN_prev) y aporte de cada factor al riesgo
  (SHAP agregado por factor).
- Diagnóstico: el aporte se separa en
    * NIVEL BASE (inercia): historial de la sección y contexto municipal/estatal.
      Dice CUÁNTO riesgo hay, pero no POR QUÉ ni qué hacer ("vota poco porque
      ha votado poco").
    * CAUSAS ATENDIBLES: desmovilización en intermedias, caída reciente,
      volatilidad, jóvenes, estructura de edad, género y acceso territorial.
      Cada una tiene una medida específica por sección.
    * CONTEXTO ESTATAL: no tener elección local concurrente. Es igual en toda la
      entidad, así que se atiende con una sola recomendación estatal (OPLE).
- Acciones: una medida por causa (`CATALOGO`). Si en una sección solo pesa el
  nivel base, se propone un diagnóstico participativo para descubrir la causa.
- Utilidad: votantes adicionales esperados por unidad de costo.
- Política: a cada sección se le asigna la medida de su causa principal y, con
  rendimiento a la mitad, la de su segunda causa. Las acciones se eligen por
  utilidad hasta agotar el presupuesto.

Supuestos (explícitos y configurables)
--------------------------------------
Efectos y costos son ILUSTRATIVOS. Se fijan con una costo-efectividad parecida
entre medidas a propósito: así la prioridad la decide el DIAGNÓSTICO (qué tanto
pesa la causa en cada sección y cuántos electores hay), no qué medida es más
barata. No son estimaciones causales con estos datos.

    ganancia_pp = efecto_max_pp · min(1, aporte_causa_pp / ESCALA_PP)
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

ESCALA_PP = 4.0          # aporte (pp) a partir del cual la medida rinde su efecto máximo
RENDIMIENTO_SEGUNDA = 0.5
N_CAUSAS = 2

FACTORES_BASE = ["historial_bajo", "contexto_municipal", "contexto_estatal", "calidad_dato"]
# La concurrencia es igual para todas las secciones de una entidad: se atiende con
# UNA recomendación estatal (ver `recomendacion_estatal`), no sección por sección.
FACTOR_ESTATAL = "concurrencia_local"
CAUSAS = ["desmovilizacion_intermedias", "caida_reciente", "volatilidad", "jovenes",
          "envejecimiento", "genero", "territorio_acceso"]
INERCIA = "inercia"      # pseudo-causa: solo pesa el nivel base

# causa -> medida (1 a 1), efecto máximo (pp de participación) y costos relativos
CATALOGO: dict[str, dict] = {
    "campania_intermedias": {
        "nombre": "Campaña sobre la importancia de las elecciones intermedias",
        "factor": "desmovilizacion_intermedias",
        "efecto_max_pp": 1.5, "costo_fijo": 5.0, "costo_por_ciudadano": 0.05,
    },
    "brigadas_casa_por_casa": {
        "nombre": "Brigadas casa por casa para recuperar votantes",
        "factor": "caida_reciente",
        "efecto_max_pp": 3.0, "costo_fijo": 15.0, "costo_por_ciudadano": 0.10,
    },
    "recordatorios": {
        "nombre": "Recordatorios cercanos a la jornada (SMS, redes, perifoneo)",
        "factor": "volatilidad",
        "efecto_max_pp": 0.8, "costo_fijo": 2.0, "costo_por_ciudadano": 0.03,
    },
    "programa_jovenes": {
        "nombre": "Programa para jóvenes en escuelas, universidades y redes",
        "factor": "jovenes",
        "efecto_max_pp": 2.0, "costo_fijo": 8.0, "costo_por_ciudadano": 0.07,
    },
    "accesibilidad_mayores": {
        "nombre": "Accesibilidad de casillas y acompañamiento a personas mayores",
        "factor": "envejecimiento",
        "efecto_max_pp": 1.5, "costo_fijo": 8.0, "costo_por_ciudadano": 0.05,
    },
    "enfoque_genero": {
        "nombre": "Campaña con enfoque de género y apoyo en tareas de cuidado",
        "factor": "genero",
        "efecto_max_pp": 1.0, "costo_fijo": 5.0, "costo_por_ciudadano": 0.035,
    },
    "acceso_territorial": {
        "nombre": "Ubicación de casillas, traslado y orientación a nuevos residentes",
        "factor": "territorio_acceso",
        "efecto_max_pp": 2.0, "costo_fijo": 12.0, "costo_por_ciudadano": 0.07,
    },
    "diagnostico_participativo": {
        "nombre": "Diagnóstico participativo y reuniones vecinales (causa por identificar)",
        "factor": INERCIA,
        "efecto_max_pp": 2.0, "costo_fijo": 12.0, "costo_por_ciudadano": 0.07,
    },
}
MEDIDA_DE_FACTOR = {d["factor"]: m for m, d in CATALOGO.items()}


def cargar_riesgo(anio: int) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(PROC_DIR, f"riesgo_{anio}.parquet"))


def causas(r: pd.DataFrame, top: int = N_CAUSAS) -> pd.DataFrame:
    """Causas atendibles principales por sección (aporte SHAP en pp) y nivel base."""
    cols = [f"shap_{c}" for c in CAUSAS if f"shap_{c}" in r.columns]
    v = r[cols].astype(float).fillna(0).to_numpy() * 100
    nombres = np.array([c[5:] for c in cols])
    orden = np.argsort(-v, axis=1)
    out = pd.DataFrame(index=r.index)
    for i in range(top):
        pp = np.take_along_axis(v, orden[:, i:i + 1], 1)[:, 0]
        out[f"causa_{i + 1}"] = np.where(pp > 0, nombres[orden[:, i]], "ninguna")
        out[f"causa_{i + 1}_pp"] = np.where(pp > 0, pp, 0.0)
    base = [f"shap_{c}" for c in FACTORES_BASE if f"shap_{c}" in r.columns]
    out["inercia_pp"] = r[base].astype(float).fillna(0).sum(axis=1).clip(lower=0).to_numpy() * 100
    # Causa principal para mostrar: si no hay causa atendible, es la inercia
    out["causa_principal"] = np.where(out["causa_1"] == "ninguna",
                                      np.where(out["inercia_pp"] > 0, INERCIA, "ninguna"),
                                      out["causa_1"])
    return out


RECOMENDACION_ESTATAL = ("Coordinar con el OPLE una campaña estatal de promoción del voto: "
                         "con menos elecciones locales el mismo día que en la elección anterior, "
                         "la participación tiende a caer.")


def recomendacion_estatal(r: pd.DataFrame, min_pp: float = 0.5) -> pd.DataFrame:
    """Entidades donde la falta de concurrencia eleva el riesgo (aporte medio en pp)."""
    col = f"shap_{FACTOR_ESTATAL}"
    if col not in r.columns:
        return pd.DataFrame(columns=["EDOCVE", "aporte_pp", "LN"])
    w = r["LN_prev"].fillna(0)
    g = (r.assign(_x=r[col].astype(float).fillna(0) * 100 * w, _w=w)
         .groupby("EDOCVE")[["_x", "_w"]].sum())
    g["aporte_pp"] = g["_x"] / g["_w"].where(g["_w"] > 0)
    g = g[g["aporte_pp"] >= min_pp].rename(columns={"_w": "LN"})
    return g[["aporte_pp", "LN"]].sort_values("aporte_pp", ascending=False).reset_index()


def candidatas(r: pd.DataFrame) -> pd.DataFrame:
    """Acciones posibles (sección × medida de su causa) con ganancia, costo y utilidad."""
    c = causas(r)
    ln = r["LN_prev"].fillna(0).to_numpy(float)
    abst_esp = ln * r["riesgo"].fillna(0).to_numpy(float)
    filas = []
    for i in range(1, N_CAUSAS + 1):
        fac = c[f"causa_{i}"].to_numpy()
        pp = c[f"causa_{i}_pp"].to_numpy()
        if i == 1:  # sin causa atendible: diagnóstico participativo sobre la inercia
            sin = (fac == "ninguna") & (c["inercia_pp"].to_numpy() > 0)
            fac = np.where(sin, INERCIA, fac)
            pp = np.where(sin, c["inercia_pp"].to_numpy(), pp)
        for f, med in MEDIDA_DE_FACTOR.items():
            m = (fac == f) & (pp > 0) & (ln > 0)
            if not m.any():
                continue
            d = CATALOGO[med]
            g = d["efecto_max_pp"] * np.minimum(1.0, pp[m] / ESCALA_PP)
            if i > 1:
                g = g * RENDIMIENTO_SEGUNDA
            filas.append(pd.DataFrame({
                "idx": np.flatnonzero(m), "medida": med, "factor": f, "orden_en_seccion": i,
                "contribucion_pp": pp[m], "ganancia_pp": g,
                "votantes": np.minimum(ln[m] * g / 100, abst_esp[m]),
                "costo": d["costo_fijo"] + d["costo_por_ciudadano"] * ln[m],
            }))
    if not filas:
        return pd.DataFrame()
    out = pd.concat(filas, ignore_index=True)
    out["utilidad"] = out["votantes"] / out["costo"]
    for col in CLAVE + ["MPIOCVE", "riesgo", "LN_prev"]:
        if col in r.columns:
            out[col] = r[col].to_numpy()[out["idx"].to_numpy()]
    return out.drop(columns="idx").reset_index(drop=True)


def planificar(cand: pd.DataFrame, presupuesto: float) -> pd.DataFrame:
    """Elige acciones por mayor utilidad hasta agotar el presupuesto. La segunda
    medida de una sección solo entra si la primera ya entró."""
    if cand.empty:
        return cand.assign(seleccionada=pd.Series(dtype=bool))
    c = cand.sort_values("utilidad", ascending=False).reset_index(drop=True)
    c["costo_acum"] = c["costo"].cumsum()
    c["seleccionada"] = c["costo_acum"] <= presupuesto
    primeras = set(map(tuple, c.loc[c["seleccionada"] & (c["orden_en_seccion"] == 1), CLAVE].to_numpy()))
    seg = c["seleccionada"] & (c["orden_en_seccion"] > 1)
    if seg.any():
        ok = [tuple(k) in primeras for k in c.loc[seg, CLAVE].to_numpy()]
        c.loc[seg, "seleccionada"] = ok
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
    c = causas(fila.to_frame().T).iloc[0]
    out = []
    for i in range(1, N_CAUSAS + 1):
        f, pp = c[f"causa_{i}"], float(c[f"causa_{i}_pp"])
        if f == "ninguna" or pp <= 0:
            continue
        med = MEDIDA_DE_FACTOR[f]
        g = CATALOGO[med]["efecto_max_pp"] * min(1.0, pp / ESCALA_PP) * (1 if i == 1 else RENDIMIENTO_SEGUNDA)
        out.append({"factor": f, "contribucion_pp": pp, "medida": med,
                    "nombre": CATALOGO[med]["nombre"], "ganancia_pp": g})
    if not out and c["inercia_pp"] > 0:
        med = MEDIDA_DE_FACTOR[INERCIA]
        out.append({"factor": INERCIA, "contribucion_pp": float(c["inercia_pp"]), "medida": med,
                    "nombre": CATALOGO[med]["nombre"],
                    "ganancia_pp": CATALOGO[med]["efecto_max_pp"] * min(1.0, c["inercia_pp"] / ESCALA_PP)})
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
                              "factores_base": FACTORES_BASE, "catalogo": CATALOGO}})
    plan.to_parquet(os.path.join(PROC_DIR, f"intervenciones_{a.anio}.parquet"), index=False)
    with open(os.path.join(PROC_DIR, f"agente_resumen_{a.anio}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "supuestos"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
