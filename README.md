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
- Predice la participación de cada sección **relativa al nivel nacional** (el "dónde"). El riesgo es 1 − (participación relativa predicha + supuesto del nivel nacional).
- Solo usa información anterior a la elección objetivo, más el calendario de elecciones locales concurrentes, que se conoce de antemano. Incluye el **contexto vecinal**: la participación pasada de las 8 secciones más cercanas.
- **Validación temporal estricta** (sin mezclar años): todo se elige con dos pliegues de origen móvil (2015 → 2018 y 2015 + 2018 → 2021, promediados), se prueba una sola vez con 2024 y se pronostica 2027.
- Compara **Ridge**, **Lasso**, **Gradient Boosting** y un **ensamble** del mejor lineal con Gradient Boosting, con el peso elegido en validación, contra baselines simples. Se elige por el error del "dónde", que no depende del supuesto nacional.
- **Nivel nacional:** se comparan reglas simples (repetir la comparable, promedio de comparables, tendencia, tendencia amortiguada) y se elige con los años de validación.
- **Riesgo alto:** un clasificador estima la probabilidad de que cada sección quede en el 10% de mayor abstención, y se compara con el ranking de la regresión.
- **Intervalos de 80%:** Gradient Boosting por cuantiles con calibración conformal, usando un año no visto del mismo tipo de elección.
- **SHAP** descompone el riesgo de cada sección en factores interpretables.

**Métricas** (completas en [`data/processed/riesgo_metricas.json`](data/processed/riesgo_metricas.json) y en la vista ⚙️ Modelo de la app). Todas en puntos porcentuales (pp) salvo las de ranking:
- **Error del dónde:** MAE de la participación *relativa al nivel nacional*. Es lo que aprende el modelo.
- **Error total:** MAE de la participación. Suma además el error del supuesto nacional, igual para todas las secciones: −4.8 pp en 2021 y +2.7 pp en 2024.
- **Corr. de rangos:** Spearman entre riesgo predicho y real.
- **Top 10%:** fracción del 10% de secciones con más abstención real que el modelo pone en su 10% de mayor riesgo. **AUC top 10%:** qué tan bien separa ese grupo del resto (0.5 = azar).

**Validación 2021** (con esta se elige: es intermedia, como 2027, y su entrenamiento incluye una presidencial):

| Modelo | Error del dónde | Error total | Corr. de rangos | Top 10% | AUC top 10% |
|---|---|---|---|---|---|
| **Ensamble: 90% Lasso + 10% Gradient Boosting (elegido)** | **6.53** | 6.98 | **0.75** | 0.55 | 0.89 |
| Lasso | 6.54 | 7.09 | 0.74 | 0.55 | 0.89 |
| Ridge | 6.60 | 6.85 | 0.74 | 0.56 | 0.89 |
| Gradient Boosting | 7.29 | 6.95 | 0.72 | 0.48 | 0.86 |
| Baseline: repetir la elección comparable (2015) | 8.73 | 9.75 | 0.60 | 0.45 | 0.84 |
| Baseline: repetir la elección anterior (2018) | 6.31 | 8.13 | 0.76 | 0.62 | 0.92 |

**Prueba 2024** (presidencial, no se usó para elegir nada):

| Modelo | Error del dónde | Error total | Corr. de rangos | Top 10% | AUC top 10% |
|---|---|---|---|---|---|
| **Ensamble (elegido)** | 5.72 | 4.99 | 0.82 | 0.57 | 0.91 |
| Lasso | 5.93 | 5.18 | 0.82 | 0.56 | 0.91 |
| Ridge | 5.94 | 5.18 | 0.82 | 0.56 | 0.91 |
| Gradient Boosting | **4.51** | **4.32** | **0.85** | 0.61 | 0.93 |
| Clasificador de riesgo alto | — | — | — | 0.59 | **0.93** |
| Baseline: repetir la elección comparable (2018) | 4.62 | 4.95 | 0.82 | **0.62** | 0.93 |
| Baseline: repetir la elección anterior (2021) | 6.61 | 7.13 | 0.73 | 0.61 | 0.91 |

**Lectura honesta:**
- **El historial de cada sección explica casi todo.** Ningún modelo le gana de forma consistente a repetir una elección pasada: en 2021 el mejor baseline es la elección anterior y en 2024 la comparable. Los modelos combinan ambas y quedan cerca del mejor de los dos sin saber de antemano cuál será.
- **Por qué no se eligió Gradient Boosting.** En 2024 fue el mejor modelo, pero en 2021, la elección del mismo tipo que 2027, fue el peor. Elegirlo viendo 2024 invalidaría la prueba. Con una sola elección de validación por tipo, esta incertidumbre no se puede resolver con estos datos.
- **Qué se probó y no mejoró.** Promediar el pliegue 2018 para elegir castigó a Gradient Boosting, porque ese pliegue entrena solo con una intermedia; ahora queda como referencia. Elegir la regla del nivel nacional con un solo año escogía "tendencia", que en 2024 fue peor; se fijó la comparable.
- **Lo que sí aportó:**
  - El **clasificador de riesgo alto** tiene el mejor AUC de 2024 (0.935).
  - Los **intervalos** marcan dónde el pronóstico es más incierto.
  - El **contexto vecinal** aporta en promedio 1.4 pp al riesgo según SHAP (8.º de 13 factores). Su efecto en las métricas no se aisló: no hubo una corrida igual sin esas variables.
- **Intervalos de 80% (conservadores):** en 2024 cubrieron el 98% de las secciones, con un ancho medio de 25 pp. Se calibraron con el pliegue 2018, que es débil. Para 2027 se calibran con 2021 y salen más angostos: 22 pp de ancho medio (de 18 a 27 pp entre el 10% y el 90% de las secciones).

### 4. Agente de intervención (`src/agent.py`)
Agente racional **basado en utilidad**:
1. **Diagnóstico.** Separa el **nivel base** (historial y contexto: dicen *cuánto* riesgo hay) de las **causas atendibles** (desmovilización en intermedias, caída reciente, jóvenes, estructura de edad, género, tipo de sección y crecimiento del padrón, volatilidad).
2. **Acción.** Cada causa tiene una medida específica (p. ej., caída reciente → brigadas casa por casa; jóvenes → programa en escuelas y redes). Tener menos elecciones locales concurrentes es igual en toda la entidad, así que se atiende con **una recomendación estatal** al OPLE.
3. **Prioridad.** Ordena las acciones por votantes adicionales esperados por unidad de costo y las elige hasta agotar el presupuesto.

Con un presupuesto del 10% del costo de todas las acciones posibles, para 2027 el agente elige **14,472 acciones** en **14,397 secciones** (20.0 millones de electores) y espera **≈219 mil votantes adicionales**. Las medidas más usadas son la campaña sobre elecciones intermedias (9,854 acciones) y las brigadas casa por casa (2,923).

> **Supuesto importante:** los efectos y costos de las medidas son **ilustrativos** y están documentados en el código. El modelo identifica dónde y por qué hay riesgo; no estima efectos causales de las medidas.

### 5. App (`app.py`)
- **📊 Resultados:** eliges la zona y presionas *Ver resultados*. Muestra una conclusión automática, indicadores, nivel de riesgo, causas, perfiles, la estrategia del agente, qué midió el sistema y la descarga en CSV.
- **🗺️ Mapa:** riesgo, plan del agente, causa principal o perfil. Por entidad a nivel nacional, por municipio al elegir una entidad (rápido) y por sección al elegir un municipio o al pedir el detalle.
- **📍 Sección:** por qué tiene riesgo una sección y el razonamiento del agente en 4 pasos: percibe, diagnostica, evalúa y decide.
- **🤖 Agente:** el ciclo del agente con sus números, la curva de decisión por utilidad, la tabla de secciones elegidas con su causa, medida, efecto, costo y utilidad, y el razonamiento de cualquier sección.
- **👥 Perfiles** y **⚙️ Modelo** (métricas).
- El **presupuesto del agente** se ajusta en la barra lateral y aplica a todas las vistas.

---

## Cómo correrlo

### Opción A · App rápida en Colab (sin reentrenar)
1. Abre el notebook en Colab: <https://colab.research.google.com/github/AndreaHortaHernandez/ParticipacionCiudadanaINE/blob/main/notebooks/colab_app_rapida.ipynb>
2. *Entorno de ejecución → Ejecutar todas*. Descarga solo los resultados y la cartografía del [Release](https://github.com/AndreaHortaHernandez/ParticipacionCiudadanaINE/releases) (~3 min si el Release trae `geo.zip`; ~10 min si tiene que procesar los mapas).
3. Abre la liga `https://….trycloudflare.com` que se imprime al final. Funciona mientras la sesión de Colab siga abierta. Si el navegador dice que la página no existe, espera un minuto: el DNS tarda en conocer la liga nueva.

### Reentrenar solo el modelo en Colab (~1 h, sin bajar los datos del INE)
[`notebooks/colab_reentrenar_modelo.ipynb`](notebooks/colab_reentrenar_modelo.ipynb): toma el panel procesado y los shapefiles del Release, reentrena el modelo, regenera mapas y agente, muestra las métricas y descarga un `resultados_para_colab.zip` nuevo para subirlo al Release.

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
