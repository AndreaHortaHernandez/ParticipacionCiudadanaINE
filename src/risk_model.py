"""Modelo de riesgo de abstención por sección + explicabilidad (SHAP).

Formulación
-----------
participación_t(s) = nacional_t + rel_t(s)

- rel_t(s): participación de la sección relativa al nivel nacional de la
  elección t. Es lo que el modelo aprende: DÓNDE se participa más o menos.
- nacional_t: nivel nacional de la elección objetivo. No es predecible con
  estos datos (depende de la contienda), así que se usa un supuesto simple y
  explícito: el nivel nacional de la elección comparable anterior (mismo tipo,
  t-6 años). Es configurable (`nacional_supuesto`).
- riesgo de abstención = 1 − participación predicha.

Validación temporal estricta (sin split aleatorio). Todas las variables de una
fila objetivo t se calculan con elecciones < t:
- selección (hiperparámetros, pesos del ensamble y modelo): entrenar con 2015+2018 →
  validar en 2021 (intermedia, como 2027). 2015 → 2018 se reporta solo como referencia
  (sin presidenciales en el entrenamiento)
- prueba: entrenar con 2015, 2018 y 2021 → probar en 2024 (solo se reporta)
- variables: elecciones < t + calendario de elecciones locales concurrentes en t (conocido ex-ante)
- pronóstico: entrenar con 2015-2024 → predecir 2027 (sin dato real)

Uso:  python -m src.risk_model            (pesado: correr en Colab)
      python -m src.risk_model --rapido --salida <carpeta>   (prueba de humo, 3 entidades)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from scipy.spatial import cKDTree
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest import CLAVE, ELECCIONES, GEO_SECCION, PROC_DIR, TIPO_ELECCION, cargar_panel
from src.profiles import indicadores, participacion_nacional

SEED = 42
LN_MIN_EVAL = 50
OBJETIVOS_ENTRENAMIENTO = [2015, 2018, 2021, 2024]
ANIO_PRONOSTICO = 2027

# Variables agrupadas en factores interpretables (usados para explicar el riesgo
# y para elegir la medida de intervención en el agente).
GRUPOS = {
    "historial_bajo": ["rel_lag1", "rel_comp", "rel_comp2", "nivel_rel", "n_validas",
                       "rel_lag1_x_pres", "rel_comp_x_pres"],
    "desmovilizacion_intermedias": ["caida_intermedia", "obj_presidencial"],
    "caida_reciente": ["cambio_reciente", "tendencia_rel"],
    "volatilidad": ["volatilidad"],
    "jovenes": ["rel_jov_lag1", "brecha_joven", "pct_jovenes", "pct_e25_34"],
    "envejecimiento": ["edad_media", "pct_e65p", "brecha_65_lag1"],
    "genero": ["brecha_genero_lag1", "pct_mujeres"],
    "territorio_acceso": ["rural", "mixta", "log_LN", "crec_LN"],
    "contexto_municipal": ["mpio_rel_lag1", "mpio_rel_comp"],
    "contexto_vecinal": ["vec_rel_lag1", "vec_rel_comp", "vec_cambio_reciente"],
    "contexto_estatal": ["edo_rel_lag1", "edo_rel_comp"],
    "concurrencia_local": ["conc_gob", "conc_ayun", "conc_dloc",
                           "d_conc_gob_lag1", "d_conc_gob_comp",
                           "d_conc_ayun_lag1", "d_conc_ayun_comp"],
    "calidad_dato": ["ns_lag1"],
}
# Variables espaciales (secciones vecinas): requieren la cartografía de secciones
CENTROIDES = os.path.join(PROC_DIR, "centroides_seccion.parquet")
ESPACIAL = os.path.exists(CENTROIDES) or os.path.exists(GEO_SECCION)
K_VECINOS = 8
FEATURES = [f for g, fs in GRUPOS.items() for f in fs if ESPACIAL or g != "contexto_vecinal"]
FACTOR_DE = {f: g for g, fs in GRUPOS.items() for f in fs}

DESCRIPCION_FACTOR = {
    "historial_bajo": "historial de participación por debajo del promedio",
    "desmovilizacion_intermedias": "desmovilización en elecciones intermedias",
    "caida_reciente": "caída o tendencia reciente a la baja",
    "volatilidad": "participación inestable entre elecciones",
    "jovenes": "baja participación / alto peso de jóvenes",
    "envejecimiento": "estructura de edad",
    "genero": "brecha de género",
    "territorio_acceso": "tipo de sección (rural/mixta), tamaño o crecimiento del padrón",
    "contexto_municipal": "baja participación en el municipio",
    "contexto_vecinal": "baja participación o caída en las secciones vecinas",
    "contexto_estatal": "baja participación en la entidad",
    "concurrencia_local": "sin elecciones locales concurrentes (o menos que en la elección previa)",
    "calidad_dato": "registros sin dato en la elección previa",
}


# --------------------------------------------------------------------------
# Calendario de elecciones locales concurrentes (fuente externa, ex-ante)
# --------------------------------------------------------------------------
CALENDARIO = os.path.join(os.path.dirname(PROC_DIR), "external",
                          "fechasEleccionesMexicoDesde1994.csv")


def calendario_concurrencia(path: str = CALENDARIO) -> pd.DataFrame:
    """Indicadores 0/1 por (entidad, año): elección local el MISMO día que la
    federal de diputados. Fuente: E. Magar, 'Consecutive Reelection Institutions
    and Electoral Calendars Since 1994 in Mexico' (github.com/emagar/calendarioReelecion).
    El calendario se conoce antes de la elección, por lo que no hay fuga."""
    c = pd.read_csv(path)
    filas = []
    anios = [int(col[1:]) for col in c.columns if col.startswith("y") and col[1:].isdigit()]
    for y in anios:
        fed = c.loc[(c.edon == 0) & (c.elec == "dip"), f"y{y}"].iloc[0]
        if fed == "--" or pd.isna(fed):
            continue
        s = c[c.edon > 0]
        for edo, g in s.groupby("edon"):
            fila = {"EDOCVE": int(edo), "anio": y}
            for e in ("gob", "ayun", "dloc"):
                v = g.loc[g.elec == e, f"y{y}"]
                fila[f"conc_{e}"] = float(len(v) > 0 and v.iloc[0] == fed)
            filas.append(fila)
    return pd.DataFrame(filas).set_index(["EDOCVE", "anio"])


# --------------------------------------------------------------------------
# Construcción de variables (solo pasado)
# --------------------------------------------------------------------------
def _anterior(t: int, k: int = 1) -> int | None:
    prev = [y for y in ELECCIONES if y < t]
    return prev[-k] if len(prev) >= k else None


def centroides() -> pd.DataFrame | None:
    """Centroide de cada sección (metros, proyección de la cartografía del INE)."""
    if os.path.exists(CENTROIDES):
        return pd.read_parquet(CENTROIDES).set_index(CLAVE)
    if not os.path.exists(GEO_SECCION):
        return None
    import geopandas as gpd
    g = gpd.read_file(GEO_SECCION, columns=["entidad", "seccion"])
    c = g.geometry.centroid
    d = pd.DataFrame({"EDOCVE": g["entidad"].astype(int), "SECCION": g["seccion"].astype(int),
                      "x": c.x, "y": c.y}).groupby(CLAVE)[["x", "y"]].mean()
    d.reset_index().to_parquet(CENTROIDES, index=False)
    return d


def vecinos(X: pd.DataFrame, cols: list[str], k: int = K_VECINOS) -> pd.DataFrame:
    """Promedio de `cols` en las k secciones más cercanas (sin contar la propia)."""
    out = pd.DataFrame(np.nan, index=X.index, columns=[f"vec_{c}" for c in cols])
    cen = centroides()
    if cen is None:
        return out
    xy = cen.reindex(X.index)
    ok = xy["x"].notna().to_numpy()
    if ok.sum() <= k:
        return out
    _, idx = cKDTree(xy.loc[ok, ["x", "y"]].to_numpy()).query(xy.loc[ok, ["x", "y"]].to_numpy(), k=k + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # vecinos sin dato
        for c in cols:
            v = X.loc[ok, c].to_numpy(float)
            out.loc[ok, f"vec_{c}"] = np.nanmean(v[idx[:, 1:]], axis=1)
    return out


def construir_features(panel: pd.DataFrame, t: int, nac: pd.Series | None = None) -> pd.DataFrame:
    """Una fila por sección con variables calculadas con elecciones < t y, si
    existe, la participación observada en t."""
    nac = participacion_nacional(panel) if nac is None else nac
    y1, yc, yc2 = _anterior(t, 1), t - 6, t - 12
    ind = indicadores(panel, t, min_elecciones=2)
    X = ind[["nivel_rel", "caida_intermedia", "tendencia_rel", "volatilidad",
             "cambio_reciente", "brecha_joven", "pct_jovenes", "edad_media",
             "n_validas"]].copy()

    p = panel.set_index(CLAVE)
    last = panel[panel.eleccion == y1].set_index(CLAVE)

    def rel(year: int, col: str = "participacion") -> pd.Series:
        if year not in nac.index:
            return pd.Series(np.nan, index=X.index)
        s = p.loc[p.eleccion == year, col] - nac[year]
        return s.reindex(X.index)

    X["rel_lag1"] = rel(y1)
    X["rel_comp"] = rel(yc)
    X["rel_comp2"] = rel(yc2)
    X["rel_jov_lag1"] = rel(y1, "part_e18_24")
    X["brecha_65_lag1"] = (last["part_e65p"] - last["participacion"]).reindex(X.index)
    X["brecha_genero_lag1"] = last["brecha_genero"].reindex(X.index)
    X["pct_e25_34"] = last["pct_e25_34"].reindex(X.index)
    X["pct_e65p"] = last["pct_e65p"].reindex(X.index)
    X["pct_mujeres"] = last["pct_mujeres"].reindex(X.index)
    X["ns_lag1"] = last["ns_share"].reindex(X.index)
    tip = last["TIPOSEC"].reindex(X.index)
    X["rural"] = (tip == "R").astype(float)
    X["mixta"] = (tip == "M").astype(float)
    ln1 = last["LN"].reindex(X.index)
    X["log_LN"] = np.log1p(ln1)
    ln_c = p.loc[p.eleccion == yc, "LN"].reindex(X.index) if yc in nac.index else np.nan
    X["crec_LN"] = ln1 / ln_c - 1

    # Contexto municipal / estatal (secciones válidas), relativo al nacional
    k = p.loc[p.eleccion == y1, ["MPIOCVE"]].reindex(X.index).reset_index()

    def area_rel(year: int, nivel: list[str]) -> np.ndarray:
        if year not in nac.index:
            return np.full(len(X), np.nan)
        d = panel[(panel.eleccion == year) & panel.valido]
        m = d.groupby(nivel)[["SV", "NV"]].sum()
        m = (m.SV / (m.SV + m.NV) - nac[year]).rename("v")
        return k.merge(m, left_on=nivel, right_index=True, how="left")["v"].to_numpy()

    X["mpio_rel_lag1"] = area_rel(y1, ["EDOCVE", "MPIOCVE"])
    X["mpio_rel_comp"] = area_rel(yc, ["EDOCVE", "MPIOCVE"])
    X["edo_rel_lag1"] = area_rel(y1, ["EDOCVE"])
    X["edo_rel_comp"] = area_rel(yc, ["EDOCVE"])
    X["obj_presidencial"] = float(TIPO_ELECCION[t] == "presidencial")
    X["rel_lag1_x_pres"] = X["rel_lag1"] * X["obj_presidencial"]
    X["rel_comp_x_pres"] = X["rel_comp"] * X["obj_presidencial"]

    # Concurrencia local en t y su cambio frente a la elección anterior / comparable
    cal = calendario_concurrencia()
    edo = X.index.get_level_values("EDOCVE")

    def conc(year: int, e: str) -> np.ndarray:
        try:
            return cal.xs(year, level="anio")[f"conc_{e}"].reindex(edo).to_numpy()
        except KeyError:
            return np.full(len(X), np.nan)

    for e in ("gob", "ayun", "dloc"):
        X[f"conc_{e}"] = conc(t, e)
    for e in ("gob", "ayun"):
        X[f"d_conc_{e}_lag1"] = X[f"conc_{e}"] - conc(y1, e)
        X[f"d_conc_{e}_comp"] = X[f"conc_{e}"] - conc(yc, e)

    # Contexto vecinal: solo con elecciones < t, como el resto de variables
    if ESPACIAL:
        X = X.join(vecinos(X, ["rel_lag1", "rel_comp", "cambio_reciente"]))

    # Metadatos (no son variables del modelo)
    X["objetivo"] = t
    X["LN_prev"] = ln1
    X["MPIOCVE"] = last["MPIOCVE"].reindex(X.index)
    X["nacional_supuesto"] = nac.get(yc, np.nan)
    if t in nac.index:
        obs = p[p.eleccion == t]
        X["participacion_real"] = obs["participacion"].reindex(X.index)
        X["LN_obj"] = obs["LN"].reindex(X.index)
        X["nacional_real"] = nac[t]
        X["y_rel"] = X["participacion_real"] - nac[t]
    # Solo secciones con al menos una elección previa válida
    return X[X["rel_lag1"].notna() | X["rel_comp"].notna()]


def dataset(panel: pd.DataFrame, objetivos: list[int]) -> pd.DataFrame:
    return pd.concat([construir_features(panel, t) for t in objetivos])


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def _prep_lineal():
    return ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                                  keep_empty_features=True),
                                                    StandardScaler()), FEATURES)])


def modelo_ridge(alpha: float = 1.0):
    return make_pipeline(_prep_lineal(), Ridge(alpha=alpha))


def modelo_lasso(alpha: float = 1e-4):
    return make_pipeline(_prep_lineal(), Lasso(alpha=alpha, max_iter=5000, random_state=SEED))


def modelo_rf(**kw):
    base = dict(n_estimators=200, min_samples_leaf=20, max_features=0.5,
                n_jobs=-1, random_state=SEED)
    base.update(kw)
    return RandomForestRegressor(**base)


def modelo_hgb(**kw):
    base = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                min_samples_leaf=100, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
                random_state=SEED)
    base.update(kw)
    return HistGradientBoostingRegressor(**base)


LINEALES = ("ridge", "lasso")
GRID = {
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    # La variable objetivo está en fracción (0-1): alphas chicos
    "lasso": [{"alpha": a} for a in (1e-5, 1e-4, 1e-3)],
    # Random Forest se deja fuera de la búsqueda: en validación 2021 fue peor que el
    # lineal (MAE 7.02-7.25 vs 6.83) y agota la RAM de Colab. Para incluirlo:
    # "random_forest": [{"min_samples_leaf": m, "max_features": f}
    #                   for m in (10, 50) for f in (0.33, 0.6)],
    "gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": l, "min_samples_leaf": m}
                          for lr in (0.05, 0.1) for l in (31, 63) for m in (50, 200)],
}
# Mejores hiperparámetros de la búsqueda completa (validación 2021, corrida del 2026-11-17).
# `--fijo` los usa directamente para no repetir la búsqueda (~10 min menos en Colab).
GRID_FIJO = {"ridge": [{"alpha": 0.1}], "lasso": [{"alpha": 1e-5}],
             "gradient_boosting": [{"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 200}]}
GRID_RAPIDO = {"ridge": [{"alpha": 10.0}], "lasso": [{"alpha": 1e-4}],
               "gradient_boosting": [{"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 50}]}
FABRICA = {"ridge": modelo_ridge, "lasso": modelo_lasso, "random_forest": modelo_rf,
           "gradient_boosting": modelo_hgb}


def _eval_mask(d: pd.DataFrame) -> pd.Series:
    return d["y_rel"].notna() & (d["LN_obj"] >= LN_MIN_EVAL)


def _entrenables(tr: pd.DataFrame) -> pd.DataFrame:
    return tr[tr["y_rel"].notna()]


def _ajustar(nombre: str, params: dict, tr: pd.DataFrame):
    m = FABRICA[nombre](**params)
    tr = _entrenables(tr)
    if nombre == "random_forest":
        # RF en sklearn admite NaN, pero imputamos el indicador de faltantes de forma explícita
        m.fit(tr[FEATURES].fillna(-9), tr["y_rel"])
    else:
        m.fit(tr[FEATURES], tr["y_rel"])
    return m


def predecir_rel(nombre: str, m, X: pd.DataFrame) -> np.ndarray:
    Z = X[FEATURES].fillna(-9) if nombre == "random_forest" else X[FEATURES]
    return m.predict(Z)


def baseline_rel(X: pd.DataFrame, cual: str = "comparable") -> np.ndarray:
    """Baseline: participación relativa de la elección comparable anterior
    (misma clase; si falta, la elección inmediata anterior)."""
    if cual == "comparable":
        return X["rel_comp"].fillna(X["rel_lag1"]).to_numpy()
    return X["rel_lag1"].fillna(X["rel_comp"]).to_numpy()


def metricas(d: pd.DataFrame, pred_rel: np.ndarray) -> dict:
    """Métricas en puntos porcentuales. Las del "dónde" (relativas al nivel nacional:
    MAE_rel, MAE dentro del estado, Spearman, top 10%, AUC) dependen solo del modelo.
    Las absolutas (MAE_pp, RMSE_pp, sesgo) incluyen además el error del supuesto
    nacional, que es igual para todas las secciones (`error_nacional_pp`)."""
    m = _eval_mask(d).to_numpy()
    y = d["participacion_real"].to_numpy()[m]
    yhat = (pred_rel + d["nacional_supuesto"].to_numpy())[m]
    yr, yrh = d["y_rel"].to_numpy()[m], pred_rel[m]
    w = d["LN_obj"].to_numpy()[m]
    e, er = yhat - y, yrh - yr
    edo = d.index.get_level_values("EDOCVE").to_numpy()[m]
    er_dentro = er - pd.Series(er).groupby(edo).transform("mean").to_numpy()
    return {
        "MAE_rel_pp": round(100 * np.mean(np.abs(er)), 3),
        "MAE_dentro_estado_pp": round(100 * np.mean(np.abs(er_dentro)), 3),
        "spearman": round(float(pd.Series(yrh).corr(pd.Series(yr), method="spearman")), 4),
        # ¿Cuántas del 10% de mayor riesgo real captura el 10% de mayor riesgo predicho?
        "precision_top10": round(_precision_top(yr, yrh, 0.10), 4),
        "AUC_top10": round(_auc_top(yr, -yrh, 0.10), 4),
        "MAE_pp": round(100 * np.mean(np.abs(e)), 3),
        "RMSE_pp": round(100 * np.sqrt(np.mean(e ** 2)), 3),
        "sesgo_pp": round(100 * np.mean(e), 3),
        "error_nacional_pp": round(100 * float(d["nacional_supuesto"].iloc[0] - d["nacional_real"].iloc[0]), 3),
        "RMSE_rel_pp": round(100 * np.sqrt(np.mean(er ** 2)), 3),
        "MAE_rel_pond_LN_pp": round(100 * np.average(np.abs(er), weights=w), 3),
        "n": int(m.sum()),
    }


def _precision_top(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    k = int(len(y) * q)
    real = set(np.argsort(y)[:k])          # menor participación = mayor riesgo
    pred = set(np.argsort(yhat)[:k])
    return len(real & pred) / k


def _auc_top(y: np.ndarray, puntaje: np.ndarray, q: float) -> float:
    """AUC para separar el q% de menor participación real con un puntaje de riesgo."""
    real = y <= np.quantile(y, q)
    return float(roc_auc_score(real, puntaje))


# --------------------------------------------------------------------------
# Candidatos: modelos base y ensamble con pesos elegidos en validación
# --------------------------------------------------------------------------
# Un candidato ajustado es {"miembros": {base: estimador}, "pesos": {base: w}}.
PESOS_ENSAMBLE = np.round(np.arange(0.0, 1.01, 0.1), 2)   # peso del modelo lineal


def ajustar_candidato(pesos: dict, params: dict, tr: pd.DataFrame) -> dict:
    return {"miembros": {b: _ajustar(b, params[b], tr) for b in pesos if pesos[b] > 0},
            "pesos": {b: w for b, w in pesos.items() if w > 0}}


def predecir_candidato(obj: dict, X: pd.DataFrame) -> np.ndarray:
    return sum(w * predecir_rel(b, obj["miembros"][b], X) for b, w in obj["pesos"].items())


# --------------------------------------------------------------------------
# Supuesto del nivel nacional (no predecible con estos datos: reglas simples)
# --------------------------------------------------------------------------
def nivel_nacional(nac: pd.Series, t: int, regla: str) -> float:
    c, c2 = nac.get(t - 6, np.nan), nac.get(t - 12, np.nan)
    if regla == "comparable":
        return c
    if regla == "promedio_comparables":
        return (c + c2) / 2
    if regla == "tendencia":
        return c + (c - c2)
    if regla == "tendencia_amortiguada":
        return c + 0.5 * (c - c2)
    raise ValueError(regla)


REGLAS_NACIONAL = ["comparable", "promedio_comparables", "tendencia", "tendencia_amortiguada"]


REGLA_NACIONAL = "comparable"


def evaluar_reglas_nacional(nac: pd.Series, anios_validacion: list[int]) -> tuple[str, dict]:
    """Error (pp) de cada regla en cada elección donde se puede calcular. La regla que se
    USA es fija (`REGLA_NACIONAL`): las alternativas solo existen desde 2021, así que habría
    un único año para elegir, y en 2024 ninguna mejora claramente a la comparable."""
    err = {r: {int(t): round(100 * abs(nivel_nacional(nac, t, r) - nac[t]), 2)
               for t in nac.index if not np.isnan(nivel_nacional(nac, t, r))} for r in REGLAS_NACIONAL}
    comunes = [t for t in anios_validacion if all(t in e for e in err.values())]
    mejor_val = min(REGLAS_NACIONAL, key=lambda r: np.mean([err[r][t] for t in comunes]))
    return REGLA_NACIONAL, {"error_pp_por_eleccion": err, "regla_elegida": REGLA_NACIONAL,
                            "criterio": "fija a priori (un solo año para comparar reglas)",
                            "mejor_en_validacion": {"regla": mejor_val, "anios": comunes}}


def fijar_supuesto(d: pd.DataFrame, nac: pd.Series, regla: str) -> pd.DataFrame:
    """Aplica la regla; si no se puede calcular (faltan elecciones), usa la comparable."""
    t = int(d["objetivo"].iloc[0])
    v = nivel_nacional(nac, t, regla)
    d = d.copy()
    d["nacional_supuesto"] = v if not np.isnan(v) else nivel_nacional(nac, t, "comparable")
    return d


# --------------------------------------------------------------------------
# Riesgo alto (clasificador) e intervalos de predicción
# --------------------------------------------------------------------------
def _etiqueta_alto(d: pd.DataFrame, q: float = 0.10) -> pd.Series:
    """1 si la sección está en el q% de menor participación relativa de SU elección."""
    return d.groupby("objetivo")["y_rel"].transform(lambda s: (s <= s.quantile(q)).astype(float))


def ajustar_clasificador(tr: pd.DataFrame, params: dict):
    tr = _entrenables(tr)
    m = HistGradientBoostingClassifier(
        learning_rate=params.get("learning_rate", 0.05), max_leaf_nodes=params.get("max_leaf_nodes", 31),
        min_samples_leaf=params.get("min_samples_leaf", 100), l2_regularization=1.0, max_iter=600,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)
    return m.fit(tr[FEATURES], _etiqueta_alto(tr))


def metricas_top(d: pd.DataFrame, puntaje: np.ndarray) -> dict:
    m = _eval_mask(d).to_numpy()
    yr, p = d["y_rel"].to_numpy()[m], puntaje[m]
    return {"precision_top10": round(_precision_top(yr, -p, 0.10), 4),
            "AUC_top10": round(_auc_top(yr, p, 0.10), 4)}


COBERTURA = 0.80   # intervalos de 80%


def ajustar_cuantiles(tr: pd.DataFrame, params: dict) -> dict:
    tr = _entrenables(tr)
    a = (1 - COBERTURA) / 2
    return {q: modelo_hgb(loss="quantile", quantile=q, **params).fit(tr[FEATURES], tr["y_rel"])
            for q in (a, 1 - a)}


def _bandas(mods: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = (mods[q].predict(X[FEATURES]) for q in sorted(mods))
    return np.minimum(lo, hi), np.maximum(lo, hi)


def calibrar_intervalo(mods: dict, cal: pd.DataFrame) -> float:
    """Conformal (CQR): cuánto ampliar las bandas para cubrir el 80% en un año no visto."""
    m = _eval_mask(cal).to_numpy()
    lo, hi = _bandas(mods, cal)
    y = cal["y_rel"].to_numpy()
    s = np.maximum(lo - y, y - hi)[m]
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * COBERTURA) / n)))


def intervalo(mods: dict, X: pd.DataFrame, ajuste: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _bandas(mods, X)
    return lo - ajuste, hi + ajuste


def metricas_intervalo(d: pd.DataFrame, lo: np.ndarray, hi: np.ndarray) -> dict:
    m = _eval_mask(d).to_numpy()
    yr = d["y_rel"].to_numpy()[m]
    y = d["participacion_real"].to_numpy()[m]
    nac = d["nacional_supuesto"].to_numpy()[m]
    return {"cobertura_rel": round(float(np.mean((yr >= lo[m]) & (yr <= hi[m]))), 4),
            "cobertura_abs": round(float(np.mean((y >= lo[m] + nac) & (y <= hi[m] + nac))), 4),
            "ancho_medio_pp": round(100 * float(np.mean(hi[m] - lo[m])), 3)}


# --------------------------------------------------------------------------
# Explicabilidad
# --------------------------------------------------------------------------
def _shap_riesgo(nombre: str, m, X: pd.DataFrame) -> pd.DataFrame:
    import shap
    if nombre in LINEALES:
        Xf = X[FEATURES]
        fondo = m[0].transform(Xf.sample(min(2000, len(Xf)), random_state=SEED))
        # Se ignoran las columnas indicadoras de faltantes del imputador
        sv = shap.LinearExplainer(m[-1], fondo).shap_values(m[0].transform(Xf))[:, :len(FEATURES)]
    else:
        Z = X[FEATURES].fillna(-9) if nombre == "random_forest" else X[FEATURES]
        sv = shap.TreeExplainer(m).shap_values(Z, check_additivity=False)
    return -pd.DataFrame(sv, index=X.index, columns=FEATURES)


def explicar(obj: dict, X: pd.DataFrame, top: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP por sección en contribución al RIESGO de abstención (= -contribución
    a participación), agregado por factor. Para un ensamble ponderado, el SHAP del
    ensamble es la suma ponderada de los SHAP de sus miembros (aditividad)."""
    riesgo = sum(w * _shap_riesgo(b, obj["miembros"][b], X) for b, w in obj["pesos"].items())
    por_factor = riesgo.T.groupby(FACTOR_DE).sum().T
    out = por_factor.add_prefix("shap_")
    orden = np.argsort(-por_factor.to_numpy(), axis=1)
    cols = por_factor.columns.to_numpy()
    for i in range(top):
        out[f"factor_{i + 1}"] = cols[orden[:, i]]
        out[f"factor_{i + 1}_pp"] = 100 * np.take_along_axis(
            por_factor.to_numpy(), orden[:, i:i + 1], 1)[:, 0]
        out.loc[out[f"factor_{i + 1}_pp"] <= 0, f"factor_{i + 1}"] = "ninguno"
    out["factor_dominante"] = out["factor_1"]
    return out, riesgo


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
# Selección SOLO con el pliegue 2021 (train 2015+2018): es el único cuyo entrenamiento
# incluye una presidencial, y es intermedia como 2027. El pliegue 2018 (train solo 2015,
# sin presidenciales) castiga a los modelos no lineales: se reporta como referencia.
# 2024 es la prueba: no se usa para elegir nada.
PLIEGUES = {2018: [2015], 2021: [2015, 2018]}   # se calculan y reportan ambos
SELECCION = [2021]                              # con este se elige
METRICA_SELECCION = "MAE_rel_pp"   # error del "dónde": no depende del supuesto nacional


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true",
                    help="prueba de humo: 3 entidades y una sola configuración por modelo")
    ap.add_argument("--salida", default=PROC_DIR, help="carpeta de salida")
    ap.add_argument("--fijo", action="store_true",
                    help="usar los hiperparámetros ya elegidos (GRID_FIJO) en vez de buscarlos")
    a = ap.parse_args()
    grid = GRID_RAPIDO if a.rapido else GRID_FIJO if a.fijo else GRID
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    panel = cargar_panel()
    if a.rapido:
        panel = panel[panel["EDOCVE"].isin([1, 3, 6])]
    nac = participacion_nacional(cargar_panel() if a.rapido else panel)
    regla, info_nac = evaluar_reglas_nacional(nac, list(PLIEGUES))
    log(f"Supuesto nacional: '{regla}' (error por regla: {info_nac['error_pp_por_eleccion']})")
    datos = {t: fijar_supuesto(construir_features(panel, t, nac), nac, regla)
             for t in OBJETIVOS_ENTRENAMIENTO + [ANIO_PRONOSTICO]}
    log(f"Filas por objetivo: { {t: len(d) for t, d in datos.items()} } · espacial={ESPACIAL}")
    del panel  # ya no se usa: libera RAM para SHAP
    gc.collect()

    def entrenar(trs):
        return pd.concat([datos[t] for t in trs])

    def promedio(mets: list[dict]) -> dict:
        return {k: round(float(np.mean([m[k] for m in mets])), 4) for k in mets[0]}

    res = {"metrica_seleccion": METRICA_SELECCION, "pliegues_seleccion": SELECCION,
           "supuesto_nacional": info_nac, "variables_espaciales": ESPACIAL}

    # 1) Hiperparámetros de cada modelo base (se elige con SELECCION; se reportan ambos pliegues)
    mejores, preds_val = {}, {}
    for nombre, opciones in grid.items():
        best = None
        for params in opciones:
            ps = {v: predecir_rel(nombre, _ajustar(nombre, params, entrenar(trs)), datos[v])
                  for v, trs in PLIEGUES.items()}
            met = promedio([metricas(datos[v], ps[v]) for v in SELECCION])
            log(f"  [valid] {nombre} {params} -> MAE_rel {met['MAE_rel_pp']} top10 {met['precision_top10']}")
            if best is None or met[METRICA_SELECCION] < best[1][METRICA_SELECCION]:
                best = (params, met, ps)
        mejores[nombre], preds_val[nombre] = best[0], best[2]

    # 2) Ensamble: mejor lineal + Gradient Boosting, peso elegido en validación
    lin = min(LINEALES, key=lambda b: promedio([metricas(datos[v], preds_val[b][v])
                                                 for v in SELECCION])[METRICA_SELECCION])
    curva = {}
    for w in PESOS_ENSAMBLE:
        curva[float(w)] = promedio([metricas(datos[v], w * preds_val[lin][v]
                                             + (1 - w) * preds_val["gradient_boosting"][v])
                                    for v in SELECCION])[METRICA_SELECCION]
    w_lin = min(curva, key=curva.get)
    log(f"  [valid] ensamble {lin}+GB: peso {lin} = {w_lin} (curva MAE_rel {curva})")
    candidatos = {b: {b: 1.0} for b in grid}
    candidatos["ensamble"] = {lin: w_lin, "gradient_boosting": round(1 - w_lin, 2)}
    res["ensamble"] = {"lineal": lin, "peso_lineal": w_lin, "curva_validacion": curva}
    res["hiperparametros"] = mejores

    # 3) Validación por pliegue y promedio; baselines
    for v, trs in PLIEGUES.items():
        bloque = res.setdefault(f"{'validacion' if v in SELECCION else 'referencia'}_{v}", {})
        for b in ("comparable", "anterior"):
            bloque[f"baseline_{b}"] = metricas(datos[v], baseline_rel(datos[v], b))
        for c, pesos in candidatos.items():
            pr = sum(w * preds_val[b][v] for b, w in pesos.items() if w > 0)
            bloque[c] = metricas(datos[v], pr)
    elegido = min(candidatos, key=lambda c: promedio([res[f"validacion_{v}"][c]
                                                     for v in SELECCION])[METRICA_SELECCION])
    res["modelo_elegido"] = elegido
    log(f"Modelo elegido: {elegido} {candidatos[elegido]}")

    # 4) Prueba 2024 (solo se reporta): train 2015+2018+2021
    te, tr = datos[2024], entrenar([2015, 2018, 2021])
    res["prueba_2024"] = {f"baseline_{b}": metricas(te, baseline_rel(te, b)) for b in ("comparable", "anterior")}
    objs = {}
    for c, pesos in candidatos.items():
        if c == "ensamble":
            objs[c] = {"miembros": {b: objs[b]["miembros"][b] for b in pesos if pesos[b] > 0},
                       "pesos": {b: w for b, w in pesos.items() if w > 0}}
        else:
            objs[c] = ajustar_candidato(pesos, mejores, tr)
        res["prueba_2024"][c] = metricas(te, predecir_candidato(objs[c], te))
    for clave in [k for k in res if k.startswith(("validacion_", "referencia_", "prueba_"))]:
        print(f"\n[{clave}]\n", pd.DataFrame(res[clave]).T.drop(columns="n", errors="ignore").to_string())

    # 5) Clasificador de riesgo alto (10%) frente al ranking de la regresión
    gb = mejores["gradient_boosting"]
    res["top10"] = {}
    for v, trs in PLIEGUES.items():
        clf = ajustar_clasificador(entrenar(trs), gb)
        pr = sum(w * preds_val[b][v] for b, w in candidatos[elegido].items() if w > 0)
        res["top10"][f"{'validacion' if v in SELECCION else 'referencia'}_{v}"] = {
            "clasificador": metricas_top(datos[v], clf.predict_proba(datos[v][FEATURES])[:, 1]),
            "regresion_elegida": metricas_top(datos[v], -pr),
            "baseline_comparable": metricas_top(datos[v], -baseline_rel(datos[v]))}
    clf24 = ajustar_clasificador(tr, gb)
    res["top10"]["prueba_2024"] = {
        "clasificador": metricas_top(te, clf24.predict_proba(te[FEATURES])[:, 1]),
        "regresion_elegida": metricas_top(te, -predecir_candidato(objs[elegido], te)),
        "baseline_comparable": metricas_top(te, -baseline_rel(te))}
    log(f"Top 10%: {res['top10']}")

    # 6) Intervalos de 80% (Gradient Boosting por cuantiles + calibración conformal)
    #    Se calibra con un año no visto del MISMO tipo de elección (la dispersión entre secciones
    #    es distinta en presidenciales e intermedias): 2024 con 2018 y 2027 con 2021. Para el
    #    backtest 2021 no hay intermedia previa con pliegue: se calibra con 2018 (solo referencia).
    def cuantiles_y_ajuste(trs_fit, cal_v, trs_cal):
        ajuste = calibrar_intervalo(ajustar_cuantiles(entrenar(trs_cal), gb), datos[cal_v])
        return ajustar_cuantiles(entrenar(trs_fit), gb), ajuste

    res["intervalos"] = {"cobertura_objetivo": COBERTURA}
    bandas = {}
    for t, trs_fit, (cal_v, trs_cal) in ((2021, [2015, 2018], (2018, [2015])),
                                         (2024, [2015, 2018, 2021], (2018, [2015])),
                                         (2027, OBJETIVOS_ENTRENAMIENTO, (2021, [2015, 2018]))):
        mods, ajuste = cuantiles_y_ajuste(trs_fit, cal_v, trs_cal)
        bandas[t] = intervalo(mods, datos[t], ajuste)
        if t <= 2024:
            res["intervalos"][f"{'prueba' if t == 2024 else 'backtest'}_{t}"] = {
                **metricas_intervalo(datos[t], *bandas[t]), "ajuste_conformal_pp": round(100 * ajuste, 3),
                "calibrado_con": cal_v}
    log(f"Intervalos: {res['intervalos']}")

    # 7) Salidas para la app: backtest 2021, prueba 2024, pronóstico 2027
    os.makedirs(a.salida, exist_ok=True)

    def guardar_metricas():
        with open(os.path.join(a.salida, "riesgo_metricas.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    guardar_metricas()  # por si la sesión se corta durante SHAP
    log("Métricas guardadas; calculando SHAP y salidas…")
    salida = {}
    exp24, sv24 = explicar(objs[elegido], te)
    res["importancia_factores_2024"] = (
        sv24.abs().T.groupby(FACTOR_DE).sum().T.mean().sort_values(ascending=False) * 100
    ).round(3).to_dict()
    res["importancia_variables_2024"] = (
        sv24.abs().mean().sort_values(ascending=False) * 100).round(3).to_dict()
    salida[2024] = (te, predecir_candidato(objs[elegido], te), exp24, clf24)

    tr21 = entrenar([2015, 2018])
    o21 = ajustar_candidato(candidatos[elegido], mejores, tr21)
    salida[2021] = (datos[2021], predecir_candidato(o21, datos[2021]), explicar(o21, datos[2021])[0],
                    ajustar_clasificador(tr21, gb))

    tr27 = entrenar(OBJETIVOS_ENTRENAMIENTO)
    o27 = ajustar_candidato(candidatos[elegido], mejores, tr27)
    salida[2027] = (datos[2027], predecir_candidato(o27, datos[2027]), explicar(o27, datos[2027])[0],
                    ajustar_clasificador(tr27, gb))

    for t, (X, pr, ex, clf) in salida.items():
        r = X.copy()
        r["pred_rel"] = pr
        r["participacion_pred"] = np.clip(pr + r["nacional_supuesto"], 0, 1)
        r["riesgo"] = 1 - r["participacion_pred"]
        lo, hi = bandas[t]
        r["riesgo_min"] = np.clip(1 - (hi + r["nacional_supuesto"]), 0, 1)
        r["riesgo_max"] = np.clip(1 - (lo + r["nacional_supuesto"]), 0, 1)
        r["prob_riesgo_alto"] = clf.predict_proba(X[FEATURES])[:, 1]
        r = r.join(ex)
        r.reset_index().to_parquet(os.path.join(a.salida, f"riesgo_{t}.parquet"), index=False)
        log(f"riesgo_{t}.parquet escrito")
        del r
        gc.collect()
    joblib.dump({"nombre": elegido, "pesos": candidatos[elegido], "modelo": o27, "params": mejores,
                 "supuesto_nacional": regla, "features": FEATURES, "grupos": GRUPOS},
                os.path.join(a.salida, "modelo_riesgo.joblib"))
    guardar_metricas()
    log(f"Importancia por factor (pp medio |SHAP|): {res['importancia_factores_2024']}")
    log("Listo")


if __name__ == "__main__":
    main()
