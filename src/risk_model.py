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
- selección (hiperparámetros y modelo): entrenar con objetivos 2015 y 2018 → validar en 2021
- referencia: entrenar con 2015 → validar 2018 (no se usa para elegir: sin presidenciales en train)
- prueba: entrenar con 2015, 2018 y 2021 → probar en 2024 (solo se reporta)
- variables: elecciones < t + calendario de elecciones locales concurrentes en t (conocido ex-ante)
- pronóstico: entrenar con 2015-2024 → predecir 2027 (sin dato real)

Uso:  python -m src.risk_model
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest import CLAVE, ELECCIONES, PROC_DIR, TIPO_ELECCION, cargar_panel
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
    "contexto_estatal": ["edo_rel_lag1", "edo_rel_comp"],
    "concurrencia_local": ["conc_gob", "conc_ayun", "conc_dloc",
                           "d_conc_gob_lag1", "d_conc_gob_comp",
                           "d_conc_ayun_lag1", "d_conc_ayun_comp"],
    "calidad_dato": ["ns_lag1"],
}
FEATURES = [f for fs in GRUPOS.values() for f in fs]
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


def construir_features(panel: pd.DataFrame, t: int) -> pd.DataFrame:
    """Una fila por sección con variables calculadas con elecciones < t y, si
    existe, la participación observada en t."""
    nac = participacion_nacional(panel)
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
def modelo_lineal(alpha: float = 1.0):
    return make_pipeline(
        ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median",
                                                               add_indicator=True),
                                                 StandardScaler()), FEATURES)]),
        Ridge(alpha=alpha))


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


GRID = {
    "lineal": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    # Random Forest se deja fuera de la búsqueda: en validación 2021 fue peor que el
    # lineal (MAE 7.02-7.25 vs 6.83) y agota la RAM de Colab. Para incluirlo:
    # "random_forest": [{"min_samples_leaf": m, "max_features": f}
    #                   for m in (10, 50) for f in (0.33, 0.6)],
    "gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": l, "min_samples_leaf": m}
                          for lr in (0.05, 0.1) for l in (31, 63) for m in (50, 200)],
}
FABRICA = {"lineal": modelo_lineal, "random_forest": modelo_rf, "gradient_boosting": modelo_hgb}


def _eval_mask(d: pd.DataFrame) -> pd.Series:
    return d["y_rel"].notna() & (d["LN_obj"] >= LN_MIN_EVAL)


def _ajustar(nombre: str, params: dict, tr: pd.DataFrame):
    m = FABRICA[nombre](**params)
    tr = tr[tr["y_rel"].notna()]
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
    """MAE/RMSE en puntos porcentuales sobre participación absoluta (con el
    supuesto nacional) y sobre participación relativa (calidad del 'dónde')."""
    m = _eval_mask(d).to_numpy()
    y = d["participacion_real"].to_numpy()[m]
    yhat = (pred_rel + d["nacional_supuesto"].to_numpy())[m]
    yr, yrh = d["y_rel"].to_numpy()[m], pred_rel[m]
    w = d["LN_obj"].to_numpy()[m]
    e, er = yhat - y, yrh - yr
    edo = d.index.get_level_values("EDOCVE").to_numpy()[m]
    er_dentro = er - pd.Series(er).groupby(edo).transform("mean").to_numpy()
    return {
        "MAE_dentro_estado_pp": round(100 * np.mean(np.abs(er_dentro)), 3),
        "n": int(m.sum()),
        "MAE_pp": round(100 * np.mean(np.abs(e)), 3),
        "RMSE_pp": round(100 * np.sqrt(np.mean(e ** 2)), 3),
        "sesgo_pp": round(100 * np.mean(e), 3),
        "MAE_rel_pp": round(100 * np.mean(np.abs(er)), 3),
        "RMSE_rel_pp": round(100 * np.sqrt(np.mean(er ** 2)), 3),
        "MAE_rel_pond_LN_pp": round(100 * np.average(np.abs(er), weights=w), 3),
        "spearman": round(float(pd.Series(yrh).corr(pd.Series(yr), method="spearman")), 4),
        # ¿Cuántas del 10% de mayor riesgo real captura el 10% de mayor riesgo predicho?
        "precision_top10": round(_precision_top(yr, yrh, 0.10), 4),
    }


def _precision_top(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    k = int(len(y) * q)
    real = set(np.argsort(y)[:k])          # menor participación = mayor riesgo
    pred = set(np.argsort(yhat)[:k])
    return len(real & pred) / k


# --------------------------------------------------------------------------
# Candidatos: modelos base y ensambles (definidos de antemano)
# --------------------------------------------------------------------------
ENSAMBLES = {"ens_lineal_gb": ("lineal", "gradient_boosting")}
# Con Random Forest en GRID se puede añadir: "ens_3": ("lineal", "random_forest", "gradient_boosting")
CANDIDATOS = list(GRID) + list(ENSAMBLES)


def miembros(nombre: str) -> tuple[str, ...]:
    return ENSAMBLES.get(nombre, (nombre,))


def ajustar_candidato(nombre: str, params: dict, tr: pd.DataFrame) -> dict:
    """Devuelve {modelo_base: estimador ajustado}. Un modelo base es un ensamble de 1."""
    return {b: _ajustar(b, params[b], tr) for b in miembros(nombre)}


def predecir_candidato(obj: dict, X: pd.DataFrame) -> np.ndarray:
    return np.mean([predecir_rel(b, m, X) for b, m in obj.items()], axis=0)


# --------------------------------------------------------------------------
# Explicabilidad
# --------------------------------------------------------------------------
def _shap_riesgo(nombre: str, m, X: pd.DataFrame) -> pd.DataFrame:
    import shap
    if nombre == "lineal":
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
    a participación), agregado por factor. Para un ensamble de promedio simple, el
    SHAP del ensamble es el promedio de los SHAP de sus miembros (aditividad)."""
    riesgo = sum(_shap_riesgo(b, m, X) for b, m in obj.items()) / len(obj)
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
# Validación: el único pliegue válido para elegir es 2021 (train 2015+2018), porque
# el pliegue 2018 (train solo 2015) no contiene ninguna elección presidencial en
# entrenamiento y no puede aprender el tipo de elección. 2018 se reporta como referencia.
PLIEGUE_SELECCION = (2021, [2015, 2018])
PLIEGUE_REFERENCIA = (2018, [2015])
METRICA_SELECCION = "MAE_pp"   # MAE de participación (con supuesto nacional)


def main() -> None:
    t0 = time.time()
    panel = cargar_panel()
    datos = {t: construir_features(panel, t) for t in OBJETIVOS_ENTRENAMIENTO + [ANIO_PRONOSTICO]}
    print("Filas por objetivo:", {t: len(d) for t, d in datos.items()},
          f"({time.time() - t0:.0f}s)")
    res = {"validacion_2021": {}, "referencia_2018": {}, "prueba_2024": {},
           "metrica_seleccion": METRICA_SELECCION}

    def evaluar(pliegue, nombre, params):
        v, trs = pliegue
        obj = ajustar_candidato(nombre, params, pd.concat([datos[t] for t in trs]))
        return metricas(datos[v], predecir_candidato(obj, datos[v]))

    # 1) Hiperparámetros de cada modelo base en el pliegue 2021
    mejores = {}
    for nombre, grid in GRID.items():
        best = None
        for params in grid:
            met = evaluar(PLIEGUE_SELECCION, nombre, {nombre: params})
            print(f"  [valid 2021] {nombre} {params} -> MAE {met['MAE_pp']} MAE_rel {met['MAE_rel_pp']}")
            if best is None or met[METRICA_SELECCION] < best[1][METRICA_SELECCION]:
                best = (params, met)
        mejores[nombre] = best[0]

    # 2) Comparación de candidatos (bases + ensambles) y baselines en validación
    for b in ("comparable", "anterior"):
        for clave, (v, _) in (("validacion_2021", PLIEGUE_SELECCION),
                              ("referencia_2018", PLIEGUE_REFERENCIA)):
            res[clave][f"baseline_{b}"] = metricas(datos[v], baseline_rel(datos[v], b))
    for c in CANDIDATOS:
        res["validacion_2021"][c] = evaluar(PLIEGUE_SELECCION, c, mejores)
        res["referencia_2018"][c] = evaluar(PLIEGUE_REFERENCIA, c, mejores)
    elegido = min(CANDIDATOS, key=lambda c: res["validacion_2021"][c][METRICA_SELECCION])
    res["modelo_elegido"] = elegido
    res["hiperparametros"] = mejores

    # 3) Prueba: train 2015+2018+2021 -> test 2024 (solo se reporta, no se usa para elegir)
    te = datos[2024]
    tr = pd.concat([datos[t] for t in (2015, 2018, 2021)])
    for b in ("comparable", "anterior"):
        res["prueba_2024"][f"baseline_{b}"] = metricas(te, baseline_rel(te, b))
    objs = {}
    for c in CANDIDATOS:
        objs[c] = ajustar_candidato(c, mejores, tr)
        res["prueba_2024"][c] = metricas(te, predecir_candidato(objs[c], te))

    for clave in ("validacion_2021", "referencia_2018", "prueba_2024"):
        print(f"\n[{clave}]\n", pd.DataFrame(res[clave]).T.to_string())
    print("\nModelo elegido (validacion 2021, MAE):", elegido)

    # 4) Salidas para la app: backtest 2021, prueba 2024, pronóstico 2027
    os.makedirs(PROC_DIR, exist_ok=True)
    salida = {}
    exp24, sv24 = explicar(objs[elegido], te)
    res["importancia_factores_2024"] = (
        sv24.abs().T.groupby(FACTOR_DE).sum().T.mean().sort_values(ascending=False) * 100
    ).round(3).to_dict()
    res["importancia_variables_2024"] = (
        sv24.abs().mean().sort_values(ascending=False) * 100).round(3).to_dict()
    salida[2024] = (te, predecir_candidato(objs[elegido], te), exp24)

    o21 = ajustar_candidato(elegido, mejores, pd.concat([datos[2015], datos[2018]]))
    salida[2021] = (datos[2021], predecir_candidato(o21, datos[2021]), explicar(o21, datos[2021])[0])

    o27 = ajustar_candidato(elegido, mejores, pd.concat([datos[t] for t in OBJETIVOS_ENTRENAMIENTO]))
    salida[2027] = (datos[2027], predecir_candidato(o27, datos[2027]), explicar(o27, datos[2027])[0])

    for t, (X, pr, ex) in salida.items():
        r = X.copy()
        r["pred_rel"] = pr
        r["participacion_pred"] = np.clip(pr + r["nacional_supuesto"], 0, 1)
        r["riesgo"] = 1 - r["participacion_pred"]
        r = r.join(ex)
        r.reset_index().to_parquet(os.path.join(PROC_DIR, f"riesgo_{t}.parquet"), index=False)
    joblib.dump({"nombre": elegido, "modelo": o27, "params": mejores,
                 "features": FEATURES, "grupos": GRUPOS},
                os.path.join(PROC_DIR, "modelo_riesgo.joblib"))
    with open(os.path.join(PROC_DIR, "riesgo_metricas.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print("\nImportancia por factor (pp medio |SHAP|):", res["importancia_factores_2024"])
    print(f"Listo en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
