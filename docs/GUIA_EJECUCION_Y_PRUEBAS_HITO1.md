# Guía de ejecución y pruebas del Hito 1

## 1. Introducción breve

Este repositorio corresponde al Hito 1 de un proyecto de IA sobre DocExplore. En esta etapa se trabajó con una versión simplificada del problema: clasificación binaria de queries visuales ya recortadas, usando las clases `marqeur` y `simple_sep`.

El objetivo del Hito 1 es dejar una base experimental funcional y reproducible para:

- inspeccionar el dataset,
- correr un baseline con píxeles + Regresión Logística,
- correr CLIP + Regresión Logística,
- generar métricas, matrices de confusión y análisis de threshold,
- y validar todo con un notebook reproducible.

Todavía **no** se resuelve el problema completo del proyecto. En este repositorio no se implementa:

- retrieval sobre páginas completas,
- ranking de páginas,
- localización dentro de la página,
- bounding boxes,
- sliding windows,
- ni métricas de detección/localización.

Los experimentos que sí se pueden correr en este hito son:

- inspección del dataset,
- baseline de píxeles,
- CLIP + Regresión Logística,
- ejecución completa del pipeline,
- y ejecución automática del notebook.

## 2. Requisitos previos

Antes de ejecutar cualquier cosa, verifica estas condiciones:

- necesitas **Python 3**,
- conviene usar un **entorno virtual**,
- el dataset debe estar dentro de `data/raw/`,
- y debes ejecutar los comandos desde la **raíz del repositorio**.

En otras palabras, antes de correr scripts deberías estar ubicado en una carpeta que contenga al menos:

- `data/`
- `src/`
- `outputs/`
- `notebooks/`
- `docs/`
- `requirements.txt`

## 3. Estructura mínima relevante del repo

La estructura mínima que conviene entender es esta:

```text
.
├── data/raw/
├── data/processed/
├── src/
├── outputs/
├── notebooks/
├── docs/
├── requirements.txt
└── README.md
```

Qué hace cada parte:

- `data/raw/`: dataset original del proyecto.
- `data/processed/`: artefactos derivados, especialmente el caché de embeddings CLIP.
- `src/`: scripts principales del Hito 1.
- `outputs/`: métricas, figuras y modelos generados por los experimentos.
- `notebooks/`: notebook del hito y su versión ejecutada.
- `docs/`: contexto, informes y documentación de apoyo.
- `requirements.txt`: dependencias de Python del proyecto.
- `README.md`: resumen general del repo y comandos principales.

## 4. Preparación del entorno

Comandos recomendados para Mac/Linux terminal:

### 4.1 Crear entorno virtual

```bash
python3 -m venv .venv
```

### 4.2 Activar entorno virtual

```bash
source .venv/bin/activate
```

### 4.3 Actualizar pip

```bash
python -m pip install --upgrade pip
```

### 4.4 Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

Si en tu sistema `python` no existe, usa `python3` en los comandos de ejecución y `python3 -m pip` cuando corresponda.

## 5. Prueba 1: inspección del dataset

### Comando

```bash
python src/dataset_inspection.py
```

### Qué hace

Este script:

- recorre `data/raw/DocExplore_queries_web/`,
- detecta clases por nombre de carpeta,
- cuenta imágenes válidas por clase,
- ignora archivos no imagen o corruptos,
- cuenta páginas válidas en `data/raw/DocExplore_images/`,
- y genera un resumen con métricas y figura.

### Qué debería imprimir

Si todo está bien, deberías ver en consola algo consistente con esto:

- **35 clases detectadas**
- **1500 páginas completas válidas**
- top de clases encabezado por:
  - `marqeur: 409`
  - `simple_sep: 160`
  - `S: 147`
  - `losange: 117`
  - `croix: 92`
  - `double_sep: 87`

También debería sugerir automáticamente las clases con más imágenes, que en este repo son `marqeur` y `simple_sep`.

### Archivos que debería generar o actualizar

- `outputs/metrics/class_counts.csv`
- `outputs/figures/class_distribution.png`

### Qué debería revisar el ayudante

- que el CSV exista y tenga 35 filas de clases válidas,
- que la figura exista y muestre a `marqeur` y `simple_sep` entre las clases dominantes,
- y que el conteo de páginas válidas sea 1500.

### Señales de que algo salió mal

- error al abrir imágenes,
- cantidad de clases muy distinta de 35,
- cantidad de páginas muy distinta de 1500,
- no se genera `class_counts.csv`,
- o no se genera `class_distribution.png`.

## 6. Prueba 2: baseline con píxeles

### Comando

```bash
python src/run_hito1.py --experiment baseline
```

### Qué corre este experimento

Este experimento ejecuta solo `baseline_pixels`. Usa:

- imágenes RGB,
- redimensionamiento a `64x64`,
- flatten a vector de píxeles,
- `StandardScaler`,
- y `LogisticRegression`.

Trabaja con las clases:

- clase negativa: `marqeur`
- clase positiva: `simple_sep`

Y usa split estratificado con:

- `train_size = 455`
- `test_size = 114`

### Qué métricas debería mostrar

Con los resultados actuales del repo, debería terminar con métricas equivalentes a:

- accuracy = `1.0`
- precision = `1.0`
- recall = `1.0`
- f1 = `1.0`

### Archivos esperados

- `outputs/metrics/baseline_pixels_metrics.json`
- `outputs/metrics/baseline_pixels_classification_report.json`
- `outputs/metrics/baseline_pixels_thresholds.csv`
- `outputs/metrics/baseline_pixels_predictions.csv`
- `outputs/metrics/baseline_pixels_examples.csv`
- `outputs/metrics/baseline_pixels_summary.md`
- `outputs/figures/baseline_pixels_confusion_matrix.png`
- `outputs/figures/baseline_pixels_threshold_curves.png`
- `outputs/figures/baseline_pixels_examples.png`
- `outputs/models/baseline_pixels_model.joblib`

### Cómo interpretar los outputs principales

#### Matriz de confusión

Archivo:

- `outputs/figures/baseline_pixels_confusion_matrix.png`

Qué debería mostrar:

- clasificación perfecta,
- sin falsos positivos,
- sin falsos negativos,
- en números: `[[82, 0], [0, 32]]` con el orden `[marqeur, simple_sep]`.

Si todo salió bien, la diagonal debería estar completamente marcada y fuera de la diagonal no deberían aparecer errores.

#### Curvas de threshold

Archivo:

- `outputs/figures/baseline_pixels_threshold_curves.png`

Qué debería mostrar:

- un rango amplio de thresholds con desempeño perfecto,
- en este repo el baseline queda perfecto entre `0.10` y `0.95`.

Interpretación simple:

- el modelo no solo clasifica bien a threshold `0.5`,
- también es estable frente a cambios amplios del umbral.

### Qué revisar para evaluar rápido si funciona bien

- el JSON de métricas existe,
- las cuatro métricas están en `1.0`,
- la matriz de confusión no tiene errores,
- el CSV de thresholds existe,
- y las figuras fueron generadas sin fallar.

## 7. Prueba 3: CLIP + Regresión Logística

### Comando normal

```bash
python src/run_hito1.py --experiment clip
```

### Comando forzando recomputo de embeddings

```bash
python src/run_hito1.py --experiment clip --force-recompute-embeddings
```

### Cuándo usar cada uno

Usa el primer comando cuando:

- ya existe caché de embeddings,
- solo quieres comprobar que el experimento corre,
- o quieres reutilizar lo ya procesado.

Usa `--force-recompute-embeddings` cuando:

- sospechas que el caché está corrupto,
- cambiaste la lógica de extracción de embeddings,
- o quieres verificar el pipeline completo desde cero.

### Qué hace este experimento

Este experimento:

- carga imágenes de `marqeur` y `simple_sep`,
- usa `openai/clip-vit-base-patch32`,
- extrae embeddings de dimensión 512,
- normaliza L2 esos embeddings,
- y entrena `LogisticRegression`.

### Consideraciones prácticas

- la **primera vez** puede descargar el modelo CLIP,
- por eso puede tardar más que el baseline,
- y requiere conectividad si el modelo todavía no está cacheado localmente.

### Qué hace el caché de embeddings

El caché guarda embeddings y metadatos en:

- `data/processed/clip_embeddings/`

En el estado actual del repo ya existen subcarpetas de train y test con:

- `embeddings.npy`
- `metadata.csv`
- `info.json`

Eso evita recomputar embeddings cada vez.

### Archivos esperados

- `outputs/metrics/clip_logreg_metrics.json`
- `outputs/metrics/clip_logreg_classification_report.json`
- `outputs/metrics/clip_logreg_thresholds.csv`
- `outputs/metrics/clip_logreg_predictions.csv`
- `outputs/metrics/clip_logreg_examples.csv`
- `outputs/metrics/clip_logreg_summary.md`
- `outputs/figures/clip_logreg_confusion_matrix.png`
- `outputs/figures/clip_logreg_threshold_curves.png`
- `outputs/figures/clip_logreg_examples.png`
- `outputs/models/clip_logreg_model.joblib`
- `data/processed/clip_embeddings/`

### Qué debería mostrar si todo funciona bien

Con los resultados actuales del repo, debería terminar con:

- accuracy = `1.0`
- precision = `1.0`
- recall = `1.0`
- f1 = `1.0`

Y una matriz de confusión numéricamente igual a:

- `[[82, 0], [0, 32]]`

### Cómo interpretar los outputs principales

#### Matriz de confusión

Archivo:

- `outputs/figures/clip_logreg_confusion_matrix.png`

Interpretación simple:

- si todo salió bien, no debería haber errores fuera de la diagonal.

#### Threshold analysis

Archivo:

- `outputs/figures/clip_logreg_threshold_curves.png`

Qué muestra en este repo:

- rendimiento perfecto entre `0.40` y `0.55`,
- caída de recall al subir más el threshold,
- y aumento de falsos positivos si el threshold baja mucho.

Interpretación simple:

- a `0.5` el modelo funciona perfecto,
- pero es más sensible al threshold que el baseline.

### Cómo comparar frente al baseline

Revisión rápida:

- ambos experimentos llegan a `accuracy = precision = recall = f1 = 1.0`,
- pero el baseline es más estable frente a variaciones amplias del threshold,
- mientras CLIP queda perfecto en una zona más acotada.

## 8. Prueba 4: ejecutar todo el pipeline

### Comando

```bash
python src/run_hito1.py
```

### Qué hace

Corre ambos experimentos:

- `baseline_pixels`
- `clip_logreg`

Luego genera una comparación final para revisar el hito completo de una sola vez.

### Archivos esperados

- `outputs/metrics/hito1_experiment_comparison.csv`
- `outputs/metrics/hito1_experiment_comparison.md`

### Qué debería revisar el ayudante

En la comparación final debería aparecer:

- una fila para `baseline_pixels`,
- una fila para `clip_logreg`,
- métricas en `1.0` para ambos,
- y `simple_sep` como clase positiva.

En el estado actual del repo, el resumen markdown muestra:

- `baseline_pixels: accuracy=1.0000, precision=1.0000, recall=1.0000, f1=1.0000`
- `clip_logreg: accuracy=1.0000, precision=1.0000, recall=1.0000, f1=1.0000`

Esto es útil para una revisión rápida porque en pocos segundos permite verificar:

- que ambos pipelines corrieron,
- que se generaron outputs,
- y que los resultados esperados siguen siendo consistentes.

## 9. Prueba 5: ejecutar el notebook

### Comando

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/hito1_doc_explore.ipynb --output hito1_doc_explore.executed.ipynb
```

### Para qué sirve

Sirve para validar reproducibilidad de forma académica:

- ejecuta el notebook completo,
- vuelve a correr las celdas,
- y deja un notebook ejecutado con outputs visibles.

### Qué archivo debería generarse

- `notebooks/hito1_doc_explore.executed.ipynb`

En este repo ese archivo ya existe. Si lo vuelves a ejecutar, debería actualizarse o regenerarse en la carpeta `notebooks/`.

## 10. Qué debería ver el ayudante si todo funciona bien

Si el repositorio está funcionando bien, debería observar al menos estas señales:

- el dataset se inspecciona sin errores,
- se detectan **35 clases**,
- se detectan **1500 páginas válidas**,
- se genera `outputs/metrics/class_counts.csv`,
- se genera `outputs/figures/class_distribution.png`,
- baseline corre sin fallar,
- CLIP corre sin fallar,
- ambos generan métricas y figuras,
- aparece la comparación final en `outputs/metrics/`,
- y el notebook se puede ejecutar o ya está disponible en versión ejecutada.

Una señal adicional positiva en este repo es que:

- ambos experimentos terminan con métricas perfectas,
- y no aparecen errores en las matrices de confusión.

## 11. Qué archivos son los más importantes para revisar

Lista priorizada para evaluación rápida:

1. `README.md`
   Explica de forma general qué es el repo, qué se implementó y cómo correrlo.

2. `docs/PROJECT_CONTEXT.md`
   Define el alcance exacto del Hito 1 y aclara qué no debía implementarse todavía.

3. `docs/Informe_Hito_1.md`
   Resume el trabajo realizado en formato de informe corto.

4. `src/run_hito1.py`
   Es el punto de entrada principal para correr los experimentos del hito.

5. `outputs/metrics/hito1_experiment_comparison.csv`
   Permite ver en una sola tabla si baseline y CLIP corrieron y qué resultados dieron.

6. `outputs/figures/class_distribution.png`
   Sirve para verificar rápido que la inspección del dataset fue realizada correctamente.

7. `outputs/figures/clip_logreg_confusion_matrix.png`
   Permite revisar visualmente el resultado del experimento más costoso del hito.

8. `notebooks/hito1_doc_explore.executed.ipynb`
   Muestra el flujo reproducible del trabajo con outputs visibles.

Si se quiere una revisión todavía más rápida, basta con mirar:

- `outputs/metrics/class_counts.csv`
- `outputs/metrics/hito1_experiment_comparison.csv`
- `outputs/figures/class_distribution.png`
- `outputs/figures/clip_logreg_confusion_matrix.png`

## 12. Problemas comunes

### Problema: `python: command not found`

Solución rápida:

```bash
python3 --version
```

Si `python3` sí existe, reemplaza `python` por `python3` en los comandos.

### Problema: `pip: command not found`

Solución rápida:

```bash
python3 -m pip --version
```

Y luego usa:

```bash
python3 -m pip install -r requirements.txt
```

### Problema: entorno virtual no activado

Señales típicas:

- `ModuleNotFoundError`
- `jupyter` no encontrado
- `transformers` o `torch` no instalados

Solución:

```bash
source .venv/bin/activate
```

### Problema: faltan dependencias

Solución:

```bash
python -m pip install -r requirements.txt
```

### Problema: CLIP tarda mucho al ejecutar

Esto es normal la primera vez porque puede estar:

- descargando el modelo,
- o extrayendo embeddings desde cero.

Si después de una primera ejecución quieres reutilizar embeddings, usa:

```bash
python src/run_hito1.py --experiment clip
```

En vez de:

```bash
python src/run_hito1.py --experiment clip --force-recompute-embeddings
```

### Problema: sospecha de caché CLIP inconsistente

Solución rápida:

```bash
python src/run_hito1.py --experiment clip --force-recompute-embeddings
```

### Problema: el notebook no ejecuta

Las causas más comunes son:

- dependencias faltantes,
- entorno virtual no activado,
- o falta de `jupyter`.

Solución:

```bash
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute notebooks/hito1_doc_explore.ipynb --output hito1_doc_explore.executed.ipynb
```

## 13. Cierre

Este repositorio demuestra que el Hito 1 deja una base experimental funcional y reproducible. El trabajo realizado valida una clasificación binaria sobre queries ya recortadas, con baseline de píxeles y CLIP + Regresión Logística, y deja métricas, figuras y notebook para revisión rápida.

Lo importante es no confundir este repositorio con la solución completa del proyecto. Aquí se resuelve una etapa inicial de experimentación controlada; el problema más grande de buscar queries dentro de páginas completas y localizar su posición todavía queda pendiente para hitos posteriores.
