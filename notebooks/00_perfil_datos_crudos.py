"""Perfilado de los Conteos Censales de Participación Ciudadana (INE) 2009-2024.

Lee todos los CSV crudos de data/raw/<año>/, reporta estructura y calidad, y
guarda un agregado sección × elección en data/interim/ para la fase de ingesta.

Uso:  python notebooks/00_perfil_datos_crudos.py
"""
from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "interim")
YEARS = [2009, 2012, 2015, 2018, 2021, 2024]
NUM = ["LN", "SV", "NV", "NS"]


def read_one(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"TIPOSEC": "string",
                               "FELECCION": "string", "EDONOM": "string",
                               "MPIONOM": "string"}, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(path)


def profile_file(path: str) -> dict:
    df = read_one(path)
    r = {"file": os.path.relpath(path, RAW), "rows": len(df),
         "columns": list(df.columns),
         "nulls": {c: int(v) for c, v in df.isna().sum().items() if v},
         "dtypes": {c: str(t) for c, t in df.dtypes.items()}}
    # Valores no numéricos en columnas que deberían ser enteras
    r["non_numeric"] = {}
    for c in ["AELEC", "EDOCVE", "MPIOCVE", "SECCION", "DEF", "DEL", "SEXO", "EDAD"] + NUM:
        conv = pd.to_numeric(df[c], errors="coerce")
        bad = df.loc[conv.isna() & df[c].notna(), c]
        if len(bad):
            r["non_numeric"][c] = {"n": int(len(bad)),
                                   "ejemplos": bad.astype(str).value_counts().head(5).to_dict()}
        df[c] = conv
    r["aelec"] = sorted(df["AELEC"].dropna().unique().tolist())
    r["fecha"] = sorted(df["FELECCION"].dropna().unique().tolist())
    r["edocve"] = sorted(df["EDOCVE"].dropna().unique().tolist())
    r["tiposec"] = df["TIPOSEC"].value_counts(dropna=False).to_dict()
    r["sexo"] = {str(k): int(v) for k, v in df["SEXO"].value_counts(dropna=False).items()}
    r["edad_min"], r["edad_max"] = float(df["EDAD"].min()), float(df["EDAD"].max())
    nums = df[NUM].apply(pd.to_numeric, errors="coerce")
    r["neg"] = int((nums < 0).sum().sum())
    r["ln_mismatch_rows"] = int((nums["LN"] != nums[["SV", "NV", "NS"]].sum(axis=1)).sum())
    key = ["EDOCVE", "SECCION", "SEXO", "EDAD"]
    r["dup_keys"] = int(df.duplicated(key).sum())
    # Secciones asignadas a más de un municipio / distrito
    g = df.groupby("SECCION")
    r["secc_multi_mpio"] = int((g["MPIOCVE"].nunique() > 1).sum())
    r["secc_multi_def"] = int((g["DEF"].nunique(dropna=True) > 1).sum())
    r["secc_multi_tiposec"] = int((g["TIPOSEC"].nunique(dropna=True) > 1).sum())

    # Agregado sección: totales + estructura por edad/sexo
    df["joven"] = df["EDAD"].between(18, 29)
    df["mayor"] = df["EDAD"] >= 60
    df["mujer"] = df["SEXO"] == 1
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby(["AELEC", "EDOCVE", "SECCION"], as_index=False).agg(
        MPIOCVE=("MPIOCVE", "first"), MPIONOM=("MPIONOM", "first"),
        TIPOSEC=("TIPOSEC", "first"), DEF=("DEF", "first"), DEL=("DEL", "first"),
        LN=("LN", "sum"), SV=("SV", "sum"), NV=("NV", "sum"), NS=("NS", "sum"))
    for flag in ("joven", "mayor", "mujer"):
        sub = df[df[flag]].groupby(["AELEC", "EDOCVE", "SECCION"])[["LN", "SV"]].sum()
        sub.columns = [f"LN_{flag}", f"SV_{flag}"]
        agg = agg.merge(sub, left_on=["AELEC", "EDOCVE", "SECCION"],
                        right_index=True, how="left")
    r["_agg"] = agg
    return r


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*", "*", "*.csv")))
    print(f"{len(files)} archivos CSV")
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(profile_file, files))

    aggs = [r.pop("_agg") for r in res]
    panel = pd.concat(aggs, ignore_index=True)
    panel.to_parquet(os.path.join(OUT, "seccion_eleccion_raw.parquet"), index=False)
    with open(os.path.join(OUT, "perfil_archivos.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)

    # ---- Resumen por año ----
    fr = pd.DataFrame(res)
    fr["year"] = fr["file"].str[:4].astype(int)
    print("\n=== Filas, nulos y consistencia por año ===")
    for y, g in fr.groupby("year"):
        nulls = {}
        for d in g["nulls"]:
            for k, v in d.items():
                nulls[k] = nulls.get(k, 0) + v
        cols = {tuple(c) for c in g["columns"]}
        print(f"{y}: archivos={len(g)} filas={g.rows.sum():,} cols_distintas={len(cols)} "
              f"nulos={nulls} neg={g.neg.sum()} LN!=SV+NV+NS={g.ln_mismatch_rows.sum()} "
              f"dup={g.dup_keys.sum()} secc_multi_mpio={g.secc_multi_mpio.sum()} "
              f"secc_multi_def={g.secc_multi_def.sum()} "
              f"secc_multi_tiposec={g.secc_multi_tiposec.sum()}")
        print(f"     fechas={sorted({x for l in g.fecha for x in l})} "
              f"edad=[{g.edad_min.min()},{g.edad_max.max()}] "
              f"sexo={sorted({k for d in g.sexo for k in d})} "
              f"tiposec={sorted({str(k) for d in g.tiposec for k in d})}")
        for f_, nn in zip(g.file, g.non_numeric):
            if nn:
                print(f"     no numéricos en {os.path.basename(f_)}: {nn}")
        bad = g[g.edocve.map(len) != 1]
        if len(bad):
            print("     archivos con >1 entidad:", bad.file.tolist())

    print("\n=== Totales por elección (sección) ===")
    t = panel.groupby("AELEC")[NUM].sum()
    t["part_SV/LN"] = t.SV / t.LN
    t["part_SV/(SV+NV)"] = t.SV / (t.SV + t.NV)
    t["NS/LN"] = t.NS / t.LN
    t["secciones"] = panel.groupby("AELEC").size()
    t["municipios"] = panel.groupby("AELEC").apply(
        lambda d: d[["EDOCVE", "MPIOCVE"]].drop_duplicates().shape[0])
    t["DEF"] = panel.groupby("AELEC").apply(
        lambda d: d[["EDOCVE", "DEF"]].dropna().drop_duplicates().shape[0])
    t["DEL_nulos_secc"] = panel.groupby("AELEC")["DEL"].apply(lambda s: s.isna().sum())
    print(t.to_string())

    print("\n=== Calidad a nivel sección ===")
    panel["ns_share"] = panel.NS / panel.LN
    q = panel.groupby("AELEC").agg(
        LN_mediana=("LN", "median"), LN_lt50=("LN", lambda s: (s < 50).sum()),
        LN_eq0=("LN", lambda s: (s == 0).sum()),
        ns_gt20=("ns_share", lambda s: (s > .2).sum()),
        ns_eq1=("ns_share", lambda s: (s >= .999).sum()))
    print(q.to_string())

    print("\n=== Continuidad de secciones (EDOCVE, SECCION) entre elecciones ===")
    keys = panel.groupby(["EDOCVE", "SECCION"])["AELEC"].nunique()
    print(keys.value_counts().sort_index().rename("n_secciones_por_#elecciones").to_string())
    pres = panel.pivot_table(index=["EDOCVE", "SECCION"], columns="AELEC",
                             values="LN", aggfunc="size", fill_value=0).astype(bool)
    print("presentes por año:", pres.sum().to_dict())
    print("en las 6:", int(pres.all(axis=1).sum()), "de", len(pres))
    # Usabilidad para el modelo: secciones con datos válidos (NS<20%) en las 6
    ok = panel[(panel.ns_share <= .2) & (panel.LN >= 50)]
    okk = ok.groupby(["EDOCVE", "SECCION"])["AELEC"].nunique()
    print("en las 6 con LN>=50 y NS<=20%:", int((okk == 6).sum()))

    print("\n=== Continuidad municipios / distritos ===")
    m = panel.groupby(["EDOCVE", "MPIOCVE"])["AELEC"].nunique()
    print("municipios en las 6:", int((m == 6).sum()), "de", len(m))
    sec_mpio = panel.groupby(["EDOCVE", "SECCION"])["MPIOCVE"].nunique()
    print("secciones que cambian de municipio entre años:", int((sec_mpio > 1).sum()))
    sec_def = panel.dropna(subset=["DEF"]).groupby(["EDOCVE", "SECCION"])["DEF"].nunique()
    print("secciones que cambian de DEF entre años:", int((sec_def > 1).sum()))


if __name__ == "__main__":
    main()
