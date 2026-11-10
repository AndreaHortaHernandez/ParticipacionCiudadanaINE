"""Perfiles ciudadanos de participación (reto 1) mediante clustering de secciones.

Cada sección se describe con indicadores de su historial de participación
calculados SOLO con elecciones anteriores a la elección objetivo (sin fuga de
información del futuro) y con su estructura etaria más reciente en esa ventana.

Indicadores (todas las participaciones son SV/(SV+NV)):
- nivel_rel        media de (participación sección − participación nacional de esa elección)
- caida_intermedia media en presidenciales − media en intermedias (puntos de proporción)
- tendencia_rel    pendiente de nivel_rel por elección (cambio por cada elección)
- volatilidad      desviación estándar del residuo relativo tras quitar el efecto tipo de elección
- cambio_reciente  nivel_rel de la última elección − media de nivel_rel de las anteriores
- brecha_joven     participación 18-24 − participación total (media en la ventana)
- pct_jovenes      % de la lista nominal de 18-24 en la última elección de la ventana
- edad_media       edad media de la lista nominal en la última elección de la ventana

Se comparan K-Means, GMM y jerárquico (Ward); k se elige con silhouette
(con un mínimo de k para que los perfiles sean accionables, ver K_MIN).

Uso:  python -m src.profiles [--objetivo 2024]
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.ingest import CLAVE, ELECCIONES, PROC_DIR, TIPO_ELECCION, cargar_panel

SEED = 42
K_RANGO = range(3, 9)
K_MIN = 4                 # con k<4 los perfiles no distinguen tipos de riesgo accionables
MIN_VALIDAS = 3           # elecciones válidas mínimas en la ventana para asignar perfil
LN_MIN = 50               # secciones muy pequeñas: tasas inestables
N_SIL = 15000             # muestra para silhouette / Ward (costo cuadrático)

FEATURES = ["nivel_rel", "caida_intermedia", "tendencia_rel", "volatilidad",
            "cambio_reciente", "brecha_joven", "pct_jovenes", "edad_media"]

ETIQUETAS = {
    "nivel_rel": ("participación por encima del promedio nacional",
                  "participación por debajo del promedio nacional"),
    "caida_intermedia": ("fuerte caída en elecciones intermedias",
                         "participación estable entre presidenciales e intermedias"),
    "tendencia_rel": ("tendencia al alza", "tendencia a la baja"),
    "volatilidad": ("participación muy variable entre elecciones",
                    "participación muy estable"),
    "cambio_reciente": ("repunte en la última elección", "caída brusca en la última elección"),
    "brecha_joven": ("jóvenes participan casi como el promedio",
                     "jóvenes participan mucho menos que el resto"),
    "pct_jovenes": ("alta proporción de jóvenes 18-24", "baja proporción de jóvenes 18-24"),
    "edad_media": ("población envejecida", "población joven"),
}


# --------------------------------------------------------------------------
# Indicadores
# --------------------------------------------------------------------------
def participacion_nacional(panel: pd.DataFrame) -> pd.Series:
    v = panel[panel.valido]
    g = v.groupby("eleccion")[["SV", "NV"]].sum()
    return g.SV / (g.SV + g.NV)


def indicadores(panel: pd.DataFrame, objetivo: int) -> pd.DataFrame:
    """Indicadores por sección usando solo elecciones < objetivo."""
    ventana = [y for y in ELECCIONES if y < objetivo]
    if len(ventana) < 3:
        raise ValueError("Se requieren al menos 3 elecciones previas")
    p = panel[panel.eleccion.isin(ventana)].copy()
    nac = participacion_nacional(p)
    p["rel"] = p["participacion"] - p["eleccion"].map(nac)
    p["pres"] = p["eleccion"].map(TIPO_ELECCION).eq("presidencial")

    idx = pd.MultiIndex.from_frame(p[CLAVE].drop_duplicates().sort_values(CLAVE))

    def _piv(col: str) -> pd.DataFrame:
        return p.pivot(index=CLAVE, columns="eleccion", values=col).reindex(index=idx, columns=ventana)

    rel, part, bj = _piv("rel"), _piv("participacion"), _piv("brecha_joven")

    R = rel.to_numpy()
    ok = ~np.isnan(R)
    n_val = ok.sum(1)
    out = pd.DataFrame(index=rel.index)
    out["n_validas"] = n_val
    out["nivel_rel"] = np.nanmean(R, axis=1)

    pres_cols = [i for i, y in enumerate(ventana) if TIPO_ELECCION[y] == "presidencial"]
    int_cols = [i for i, y in enumerate(ventana) if TIPO_ELECCION[y] == "intermedia"]
    P = part.to_numpy()
    with np.errstate(all="ignore"):
        out["caida_intermedia"] = np.nanmean(P[:, pres_cols], 1) - np.nanmean(P[:, int_cols], 1)

        # Pendiente de rel vs índice de elección (mínimos cuadrados con faltantes)
        x = np.arange(len(ventana), dtype=float)[None, :].repeat(len(R), 0)
        x[~ok] = np.nan
        xm, ym = np.nanmean(x, 1, keepdims=True), np.nanmean(R, 1, keepdims=True)
        out["tendencia_rel"] = np.nansum((x - xm) * (R - ym), 1) / np.nansum((x - xm) ** 2, 1)

        # Volatilidad: residuo tras quitar media por tipo de elección de la propia sección
        res = R.copy()
        for cols in (pres_cols, int_cols):
            res[:, cols] = R[:, cols] - np.nanmean(R[:, cols], 1, keepdims=True)
        out["volatilidad"] = np.nanstd(res, 1)

        # Cambio reciente: última elección válida vs media de las anteriores
        last_idx = np.where(ok, np.arange(len(ventana))[None, :], -1).max(1)
        last = R[np.arange(len(R)), np.clip(last_idx, 0, None)]
        prev = R.copy()
        prev[np.arange(len(R)), np.clip(last_idx, 0, None)] = np.nan
        out["cambio_reciente"] = last - np.nanmean(prev, 1)
        out["brecha_joven"] = np.nanmean(bj.to_numpy(), 1)

    # Estructura más reciente en la ventana
    ult = p.sort_values("eleccion").groupby(CLAVE).tail(1).set_index(CLAVE)
    out = out.join(ult[["pct_e18_24", "edad_media", "LN", "TIPOSEC", "MPIOCVE", "MPIONOM"]])
    out = out.rename(columns={"pct_e18_24": "pct_jovenes", "LN": "LN_ultima"})
    out["participacion_media"] = np.nanmean(P, 1)
    out["elegible"] = (out.n_validas >= MIN_VALIDAS) & (out.LN_ultima >= LN_MIN) \
        & out[FEATURES].notna().all(1)
    out.attrs["ventana"] = ventana
    return out


# --------------------------------------------------------------------------
# Selección de modelo y k
# --------------------------------------------------------------------------
def evaluar_k(Xs: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.choice(len(Xs), size=min(N_SIL, len(Xs)), replace=False)
    Xm = Xs[idx]
    filas = []
    for k in K_RANGO:
        km = KMeans(k, n_init=10, random_state=SEED).fit(Xs)
        gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(Xs)
        wa = AgglomerativeClustering(k, linkage="ward").fit_predict(Xm)
        lk, lg = km.labels_, gm.predict(Xs)
        filas.append({
            "k": k,
            "sil_kmeans": silhouette_score(Xm, lk[idx], random_state=SEED),
            "sil_gmm": silhouette_score(Xm, lg[idx], random_state=SEED),
            "sil_ward": silhouette_score(Xm, wa, random_state=SEED),
            "db_kmeans": davies_bouldin_score(Xm, lk[idx]),
            "ch_kmeans": calinski_harabasz_score(Xm, lk[idx]),
            "bic_gmm": gm.bic(Xs),
            "ari_kmeans_gmm": adjusted_rand_score(lk, lg),
            "ari_kmeans_ward": adjusted_rand_score(lk[idx], wa),
            # estabilidad: misma k con otra semilla
            "ari_kmeans_semilla": adjusted_rand_score(
                lk, KMeans(k, n_init=10, random_state=SEED + 1).fit_predict(Xs)),
        })
    return pd.DataFrame(filas).set_index("k")


def elegir(met: pd.DataFrame) -> tuple[str, int]:
    """Algoritmo y k con mayor silhouette, respetando K_MIN."""
    cand = met.loc[met.index >= K_MIN, ["sil_kmeans", "sil_gmm", "sil_ward"]]
    alg, k = None, None
    best = -np.inf
    for col in cand:
        kk = cand[col].idxmax()
        if cand.loc[kk, col] > best:
            best, alg, k = cand.loc[kk, col], col.replace("sil_", ""), int(kk)
    return alg, k


# --------------------------------------------------------------------------
# Descripción narrativa
# --------------------------------------------------------------------------
def describir(ind: pd.DataFrame, labels: np.ndarray, nac_media: float) -> tuple[list[dict], dict]:
    d = ind.assign(perfil=labels)
    z = (d.groupby("perfil")[FEATURES].mean() - d[FEATURES].mean()) / d[FEATURES].std()
    medias = d.groupby("perfil")[FEATURES + ["participacion_media"]].mean()
    tipo = pd.crosstab(d.perfil, d.TIPOSEC, normalize="index")
    n = d.perfil.value_counts()

    nombres = {}
    perfiles = []
    for c in z.index:
        zc = z.loc[c]
        m = medias.loc[c]
        nombre = _nombrar(zc)
        base = nombre
        i = 2
        while nombre in nombres.values():
            nombre = f"{base} ({i})"
            i += 1
        nombres[int(c)] = nombre
        rasgos = [ETIQUETAS[f][0 if zc[f] > 0 else 1]
                  for f in zc.abs().sort_values(ascending=False).index if abs(zc[f]) >= 0.5][:4]
        tipo_txt = ", ".join(f"{ {'U': 'urbana', 'M': 'mixta', 'R': 'rural'}.get(t, t)} {v:.0%}"
                             for t, v in tipo.loc[c].sort_values(ascending=False).items() if v >= .1)
        narrativa = (
            f"{n[c]:,} secciones ({n[c] / len(d):.0%}). Participación media histórica "
            f"{m.participacion_media:.1%} (nacional {nac_media:.1%}; {m.nivel_rel * 100:+.1f} pp). "
            f"En intermedias cae {m.caida_intermedia * 100:.1f} pp respecto a presidenciales; "
            f"los jóvenes de 18-24 participan {abs(m.brecha_joven) * 100:.1f} pp "
            f"{'menos' if m.brecha_joven < 0 else 'más'} que el total. "
            f"Composición: {tipo_txt}."
        )
        if rasgos:
            narrativa += " Rasgos distintivos: " + "; ".join(rasgos) + "."
        perfiles.append({"perfil": int(c), "nombre": nombre, "n_secciones": int(n[c]),
                         "rasgos": rasgos, "narrativa": narrativa,
                         "medias": {k: round(float(v), 4) for k, v in m.items()},
                         "z": {k: round(float(v), 3) for k, v in zc.items()},
                         "tipo_seccion": {k: round(float(v), 3) for k, v in tipo.loc[c].items()}})
    return perfiles, nombres


def _nombrar(z: pd.Series) -> str:
    """Nombre corto a partir de los rasgos dominantes del centroide (z-scores)."""
    if z["cambio_reciente"] <= -1.0:
        return "Caída brusca reciente"
    if z["cambio_reciente"] >= 1.0:
        return "Repunte reciente con alta volatilidad" if z["volatilidad"] >= 0.8 else "Repunte reciente"
    if z["nivel_rel"] <= -0.6 and z["caida_intermedia"] >= 0.6:
        return "Baja participación y desmovilización en intermedias"
    if z["nivel_rel"] <= -0.6 and z["volatilidad"] < 0.5:
        return "Baja participación persistente"
    if z["nivel_rel"] <= -0.6:
        return "Baja participación volátil"
    if z["caida_intermedia"] >= 0.6 and (z["brecha_joven"] <= -0.3 or z["pct_jovenes"] >= 0.3):
        return "Jóvenes que se desmovilizan en intermedias"
    if z["caida_intermedia"] >= 0.6:
        return "Movilización solo en presidenciales"
    if z["volatilidad"] >= 0.8:
        return "Participación errática"
    if z["tendencia_rel"] <= -0.6:
        return "Desgaste gradual"
    if z["nivel_rel"] >= 0.6:
        return "Participación alta consolidada"
    if z["pct_jovenes"] >= 0.6 or z["edad_media"] <= -0.6:
        return "Secciones jóvenes en participación media"
    if z["edad_media"] >= 0.6:
        return "Población envejecida con participación estable"
    return "Participación promedio estable"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def ajustar_perfiles(panel: pd.DataFrame, objetivo: int, k: int | None = None,
                     algoritmo: str | None = None, verbose: bool = True) -> dict:
    ind = indicadores(panel, objetivo)
    el = ind[ind.elegible]
    scaler = StandardScaler().fit(el[FEATURES])
    Xs = scaler.transform(el[FEATURES])
    rng = np.random.default_rng(SEED)

    met = evaluar_k(Xs, rng)
    alg_sel, k_sel = elegir(met)
    k = k or k_sel
    algoritmo = algoritmo or alg_sel
    if verbose:
        print(f"Ventana {ind.attrs['ventana']} | secciones elegibles {len(el):,} de {len(ind):,}")
        print(met.round(3).to_string())
        print(f"Selección: {algoritmo}, k={k}")

    if algoritmo == "kmeans":
        modelo = KMeans(k, n_init=20, random_state=SEED).fit(Xs)
        labels = modelo.labels_
    elif algoritmo == "gmm":
        modelo = GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEED).fit(Xs)
        labels = modelo.predict(Xs)
    else:  # ward: ajustar en muestra y asignar el resto al centroide más cercano
        idx = rng.choice(len(Xs), size=min(N_SIL, len(Xs)), replace=False)
        wl = AgglomerativeClustering(k, linkage="ward").fit_predict(Xs[idx])
        cent = np.vstack([Xs[idx][wl == c].mean(0) for c in range(k)])
        modelo = KMeans(k, init=cent, n_init=1, max_iter=1).fit(Xs)
        labels = modelo.predict(Xs)

    # Reordenar perfiles por participación media (0 = más baja) para estabilidad de ids
    orden = pd.Series(el.participacion_media.to_numpy()).groupby(labels).mean().sort_values().index
    remap = {old: new for new, old in enumerate(orden)}
    labels = np.array([remap[l] for l in labels])

    nac = participacion_nacional(panel[panel.eleccion < objetivo]).mean()
    perfiles, nombres = describir(el, labels, nac)
    asign = ind.copy()
    asign["perfil"] = pd.Series(labels, index=el.index).reindex(ind.index).astype("Int8")
    asign["perfil_nombre"] = asign["perfil"].map(nombres).fillna("Sin historial suficiente")
    return {"objetivo": objetivo, "ventana": ind.attrs["ventana"], "algoritmo": algoritmo,
            "k": k, "metricas": met, "scaler": scaler, "modelo": modelo, "remap": remap,
            "asignacion": asign, "perfiles": perfiles, "nombres": nombres}


def guardar(res: dict) -> None:
    os.makedirs(PROC_DIR, exist_ok=True)
    y = res["objetivo"]
    res["asignacion"].reset_index().to_parquet(
        os.path.join(PROC_DIR, f"perfiles_seccion_{y}.parquet"), index=False)
    res["metricas"].to_csv(os.path.join(PROC_DIR, f"perfiles_metricas_{y}.csv"))
    with open(os.path.join(PROC_DIR, f"perfiles_descripcion_{y}.json"), "w", encoding="utf-8") as f:
        json.dump({"objetivo": y, "ventana": res["ventana"], "algoritmo": res["algoritmo"],
                   "k": res["k"], "features": FEATURES, "perfiles": res["perfiles"]},
                  f, ensure_ascii=False, indent=2)
    joblib.dump({k: res[k] for k in ("scaler", "modelo", "remap", "nombres", "algoritmo")},
                os.path.join(PROC_DIR, f"perfiles_modelo_{y}.joblib"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objetivo", type=int, default=2024)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--algoritmo", choices=["kmeans", "gmm", "ward"], default=None)
    a = ap.parse_args()
    res = ajustar_perfiles(cargar_panel(), a.objetivo, a.k, a.algoritmo)
    guardar(res)
    for p in res["perfiles"]:
        print(f"\n[{p['perfil']}] {p['nombre']}\n  {p['narrativa']}")


if __name__ == "__main__":
    main()
