# Riesgo de abstención por sección electoral

Sistema de inteligencia artificial que, con los **Conteos Censales de Participación Ciudadana del INE (2009–2024)**, identifica **dónde** y **por qué** es más probable que la ciudadanía deje de votar, y propone **qué hacer** en cada sección electoral del país.

Responde a los tres retos del Hackathon INE 2026:

| Reto | Qué hace el sistema |
|---|---|
| 1 · Perfiles ciudadanos de participación | Agrupa ~70 mil secciones en **5 perfiles** según cómo han participado de 2009 a 2024. |
| 2 · Predicción y escenarios de riesgo | Predice la **abstención de cada sección para 2027**, explica sus causas y la valida con elecciones pasadas. |
| 3 · Comunicación y visualización | **App interactiva** con mapa, conclusión automática, gráficas y un **agente** que arma la estrategia de intervención. |

---

## Cómo funciona

```
Datos INE (sección × edad × sexo, 6 elecciones)
        │  src/ingest.py
        ▼
Panel sección × elección (participación, estructura de edad y sexo, tipo de sección)
        │  src/profiles.py                         │  src/risk_model.py
        ▼                                          ▼
5 perfiles de votante (K-Means)        Riesgo de abstención por sección + SHAP
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼  src/agent.py
        Agente: diagnostica la causa de cada sección y prioriza medidas con presupuesto
                           ▼  app.py
        App Streamlit: resultados, mapa, sección, perfiles, agente, modelo
```

### 1. Ingesta (`src/ingest.py`)
- Une los CSV del INE a un panel **sección × elección** (2009, 2012, 2015, 2018, 2021, 2024).
- Participación = votaron / (votaron + no votaron). Los registros "no especificados" no cuentan como votantes ni como abstencionistas. Si superan el 50% de la lista nominal, esa sección-elección se marca como faltante.
- Variables por sección: participación por grupo de edad y sexo, estructura etaria, tipo de sección (urbana/mixta/rural).

### 2. Perfiles ciudadanos (`src/profiles.py`)
K-Means (k = 5) sobre indicadores históricos: nivel de participación relativo al país, caída en intermedias, tendencia, volatilidad, cambio reciente, brecha de jóvenes y edad media.

| Perfil | Secciones | Participación media |
|---|---|---|
| Baja participación y desmovilización en intermedias | 29% | 48.2% |
| Población envejecida con participación estable | 25% | 55.9% |
| Participación alta consolidada | 17% | 64.8% |
| Caída brusca reciente | 14% | 50.6% |
| Repunte reciente con alta volatilidad | 14% | 57.6% |

### 3. Modelo de riesgo (`src/risk_model.py`)
- Predice la participación de cada sección **relativa al nivel nacional**. El riesgo es 1 − participación predicha.
- Solo usa información anterior a la elección objetivo, más el calendario de elecciones locales concurrentes, que se conoce de antemano.
- **Validación temporal estricta** (sin mezclar años): selecciona el modelo con 2021, lo prueba con 2024 sin haberlo visto y pronostica 2027.
- Compara una regresión lineal regularizada, Gradient Boosting y un ensamble, contra baselines simples.
- **SHAP** descompone el riesgo de cada sección en factores interpretables.

**Desempeño en la corrida de referencia** (prueba con 2024, elección no vista):

| Modelo | Error medio (pp) | Correlación de rangos |
|---|---|---|
| Lineal (elegido en validación 2021) | 5.15 | 0.82 |
| Gradient Boosting | 4.35 | 0.85 |
| Baseline: repetir la elección comparable anterior | 4.95 | 0.82 |

> El modelo se elige con la validación de 2021, no con la prueba. En 2024 Gradient Boosting rindió mejor: es una mejora pendiente a evaluar.

### 4. Agente de intervención (`src/agent.py`)
Agente racional **basado en utilidad**:
1. **Diagnóstico.** Separa el **nivel base** (historial y contexto: dicen *cuánto* riesgo hay) de las **causas atendibles** (desmovilización en intermedias, caída reciente, jóvenes, estructura de edad, género, tipo de sección y crecimiento del padrón, volatilidad).
2. **Acción.** Cada causa tiene una medida específica (p. ej., caída reciente → brigadas casa por casa; jóvenes → programa en escuelas y redes). Tener menos elecciones locales concurrentes es igual en toda la entidad, así que se atiende con **una recomendación estatal** al OPLE.
3. **Prioridad.** Ordena las acciones por votantes adicionales esperados por unidad de costo y las elige hasta agotar el presupuesto.

> **Supuesto importante:** los efectos y costos de las medidas son **ilustrativos** y están documentados en el código. El modelo identifica dónde y por qué hay riesgo; no estima efectos causales de las medidas.

### 5. App (`app.py`)
- **📊 Resultados:** eliges la zona y presionas *Ver resultados*. Muestra una conclusión automática, indicadores, nivel de riesgo, causas, perfiles, la estrategia del agente, qué midió el sistema y la descarga en CSV.
- **🗺️ Mapa:** riesgo, causa principal o perfil por sección, con zoom automático a la entidad o municipio elegido.
- **📍 Sección:** por qué tiene riesgo una sección específica y qué recomienda el agente.
- **👥 Perfiles**, **🤖 Agente** (presupuesto ajustable) y **⚙️ Modelo** (métricas).

---

## Cómo correrlo

### Opción A · App rápida en Colab (~10 min, sin reentrenar)
1. Abre [`notebooks/colab_app_rapida.ipynb`](notebooks/colab_app_rapida.ipynb) en Google Colab.
2. *Entorno de ejecución → Ejecutar todas*.
3. Cuando lo pida, sube `resultados_para_colab.zip` (resultados ya calculados; pídelo al equipo, no está en el repo por su tamaño).
4. Abre la liga `https://….trycloudflare.com` que se imprime al final. Funciona mientras la sesión de Colab siga abierta.

### Opción B · Pipeline completo en Colab (~1 h)
[`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb): descarga los datos del INE (~3.5 GB) y la cartografía, y corre todo desde cero: ingesta → perfiles → modelo → cartografía → agente → app.

### Opción C · Local (Python 3.11)
```bash
pip install -r requirements.txt
# con data/processed/ ya generado (opciones A/B):
python -m src.geo          # cartografía ligera (requiere data/geo/)
streamlit run app.py
```
Cada etapa también se puede correr por separado:
```bash
python -m src.ingest
python -m src.profiles --objetivo 2024
python -m src.risk_model
python -m src.agent --anio 2027 --presupuesto-pct 10
```

---

## Estructura

```
app.py                     App Streamlit
.streamlit/config.toml     Tema de colores
src/
  ingest.py                Ingesta y limpieza → panel sección × elección
  profiles.py              Perfiles ciudadanos (K-Means)
  risk_model.py            Modelo de riesgo, validación temporal y SHAP
  agent.py                 Agente de intervención
  geo.py                   Cartografía simplificada para la app
notebooks/
  colab_pipeline.ipynb     Pipeline completo
  colab_app_rapida.ipynb   App con resultados ya calculados
  00_perfil_datos_crudos.py
data/                      (no versionado) raw/, geo/, interim/, processed/
data/external/             Calendario electoral (versionado)
docs/                      Manual de los Conteos Censales
```

## Fuentes de datos
- **INE · Conteos Censales de Participación Ciudadana 2009–2024** (datos abiertos): participación por sección, edad y sexo.
- **INE · Cartografía electoral** (entidad, municipio, sección).
- **Calendario de elecciones locales**: E. Magar, *Consecutive Reelection Institutions and Electoral Calendars Since 1994 in Mexico*, Harvard Dataverse, doi:10.7910/DVN/X2IDWS.

## Limitaciones
- El nivel nacional de 2027 se supone igual al de la última elección comparable (2021); el modelo aprende sobre todo **dónde** se participa más o menos.
- Solo usa variables disponibles en la fuente (edad, sexo, tipo de sección, historial, calendario). No incluye variables socioeconómicas externas.
- Efectos y costos de las medidas del agente son supuestos ilustrativos.

## Equipo
[Integrantes del equipo]
