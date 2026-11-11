"""Cartografía ligera para la app.

Reproyecta a WGS84 (EPSG:4326) y simplifica la cartografía del INE para poder
dibujarla en el navegador:
- entidades.geojson            (32 polígonos, nacional)
- municipios_<EDO>.geojson     (por entidad)
- secciones_<EDO>.geojson      (por entidad; ~70 mil secciones en total)

Uso:  python -m src.geo            (pesado: correr en Colab)
Salida: data/processed/geo/
"""
from __future__ import annotations

import os

import geopandas as gpd

from src.ingest import PROC_DIR, ROOT

GEO_DIR = os.path.join(ROOT, "data", "geo")
OUT_DIR = os.path.join(PROC_DIR, "geo")
FUENTES = {
    "ENTIDAD": os.path.join(GEO_DIR, "ENTIDAD", "ENTIDAD_26", "ENTIDAD_26.shp"),
    "MUNICIPIO": os.path.join(GEO_DIR, "MUNICIPIO", "MUNICIPIO_26", "MUNICIPIO_26.shp"),
    "SECCION": os.path.join(GEO_DIR, "SECCION", "SECCIONES_26", "SECCION_26.shp"),
}
# Tolerancia de simplificación en metros (la cartografía viene en Lambert, metros)
TOLERANCIA = {"ENTIDAD": 2000, "MUNICIPIO": 300, "SECCION": 40}
DECIMALES = 5  # ~1 m en coordenadas geográficas


def _preparar(capa: str, columnas: dict[str, str]) -> gpd.GeoDataFrame:
    g = gpd.read_file(FUENTES[capa], columns=list(columnas))
    g = g.rename(columns=columnas)
    g["geometry"] = g.geometry.simplify(TOLERANCIA[capa], preserve_topology=True)
    g = g[~g.geometry.is_empty & g.geometry.notna()]
    return g.to_crs(4326)


def _escribir(g: gpd.GeoDataFrame, nombre: str) -> None:
    g.to_file(os.path.join(OUT_DIR, nombre), driver="GeoJSON",
              COORDINATE_PRECISION=DECIMALES)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    ent = _preparar("ENTIDAD", {"entidad": "EDOCVE", "nombre": "EDONOM"})
    print("entidades:", len(ent))

    mun = _preparar("MUNICIPIO", {"entidad": "EDOCVE", "municipio": "MPIOCVE", "nombre": "MPIONOM"})
    sec = _preparar("SECCION", {"entidad": "EDOCVE", "municipio": "MPIOCVE",
                                "seccion": "SECCION", "tipo": "TIPOSEC"})
    for g in (mun, sec):
        for c in ("EDOCVE", "MPIOCVE", "SECCION"):
            if c in g.columns:
                g[c] = g[c].astype(int)
    # Una sección/municipio puede venir en varios polígonos: se unen por clave
    sec = sec.dissolve(by=["EDOCVE", "SECCION"], as_index=False, aggfunc="first")
    mun = mun.dissolve(by=["EDOCVE", "MPIOCVE"], as_index=False, aggfunc="first")
    # Claves únicas para enlazar con los datos en el mapa
    sec["clave"] = sec["EDOCVE"] * 100000 + sec["SECCION"]
    mun["clave"] = mun["EDOCVE"] * 1000 + mun["MPIOCVE"]
    ent["EDOCVE"] = ent["EDOCVE"].astype(int)
    _escribir(ent, "entidades.geojson")

    for edo in sorted(ent["EDOCVE"]):
        _escribir(mun[mun["EDOCVE"] == edo], f"municipios_{edo:02d}.geojson")
        s = sec[sec["EDOCVE"] == edo]
        _escribir(s, f"secciones_{edo:02d}.geojson")
        print(f"entidad {edo:02d}: {len(s)} secciones")


if __name__ == "__main__":
    main()
