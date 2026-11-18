"""Punto de entrada para Streamlit Community Cloud.

Usa sus propias dependencias (deploy/streamlit/requirements.txt, sin SHAP ni
geopandas) y ejecuta la app de la raíz del repo. La primera vez la app baja
los resultados del Release (ver `asegurar_resultados` en app.py).
"""
import os
import runpy
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)
runpy.run_path(os.path.join(RAIZ, "app.py"), run_name="__main__")
