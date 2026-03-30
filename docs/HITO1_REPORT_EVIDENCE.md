# HITO 1 Report Evidence

## 1. Resumen factual del proyecto

### Problema general del proyecto

- El proyecto trabaja con documentos históricos y queries visuales.
- El objetivo final es poder buscar una query visual dentro de páginas completas de documentos.
- En etapas futuras, el sistema debería identificar en qué página aparece la query y, eventualmente, estimar su ubicación dentro de la página.

### Alcance exacto del Hito 1

- El Hito 1 se centró en una clasificación binaria de queries visuales ya recortadas.
- El objetivo práctico fue construir una primera base experimental reproducible con Regresión Logística.
- Se compararon dos enfoques:
  - `baseline_pixels`
  - `clip_logreg`

### Qué NO se implementó todavía

- Retrieval completo sobre páginas.
- Ranking de páginas.
- Localización espacial dentro de una página.
- Bounding boxes.
- Sliding windows.
- mAP de detección o localización.

## 2. Descripción factual del dataset

### Contenido de cada dataset

- `DocExplore_images`: contiene páginas completas de documentos históricos.
- `DocExplore_queries_web`: contiene imágenes de queries organizadas por subcarpetas; cada subcarpeta corresponde a una clase.
- `evaluation_kit_v2`: contiene archivos auxiliares (`ctypes__.iso`, `im_example.txt`, `ps_example.txt`, `main`). Su uso exacto no se puede inferir con certeza a partir de los experimentos ejecutados del Hito 1, y no fue usado en el pipeline reportado.

### Conteos globales verificados

- Número total de clases detectadas en `DocExplore_queries_web`: **35**
- Número total de páginas válidas en `DocExplore_images`: **1500**

### Top 10 clases por cantidad de imágenes válidas

| Clase | Cantidad |
| --- | ---: |
| marqeur | 409 |
| simple_sep | 160 |
| S | 147 |
| losange | 117 |
| croix | 92 |
| double_sep | 87 |
| rubanlettrine | 68 |
| encadrement | 60 |
| T | 39 |
| D | 35 |

### Clases elegidas para el experimento

- Clase negativa: `marqeur`
- Clase positiva: `simple_sep`

### Justificación breve y factual de la elección

- Son las dos clases con mayor cantidad de imágenes válidas en `outputs/metrics/class_counts.csv`.
- Son las clases definidas por defecto en `src/config.py` mediante `DEFAULT_TARGET_CLASSES = ("marqeur", "simple_sep")`.

## 3. Detalles exactos del pipeline experimental

### Configuración común

- `random_state`: **42**
- `test_size`: **0.2**
- Script principal: `src/run_hito1.py`
- Modo por defecto del runner: ejecutar ambos experimentos (`baseline` y `clip`)

### Experimento `baseline_pixels`

- Tamaño de imagen: **64 x 64**
- Preprocesamiento:
  - carga de imagen en RGB
  - redimensionamiento a 64x64
  - flatten a vector de píxeles
  - `StandardScaler`
- Clasificador usado: `LogisticRegression`
- Train size: **455**
- Test size: **114**
- Clase positiva: `simple_sep`
- Clase negativa: `marqeur`

### Experimento `clip_logreg`

- Modelo CLIP usado: `openai/clip-vit-base-patch32`
- Dimensión del embedding: **512**
- Preprocesamiento:
  - procesamiento con `CLIPProcessor`
  - extracción de embeddings de imagen
  - normalización L2
- Clasificador usado: `LogisticRegression`
- Train size: **455**
- Test size: **114**
- Clase positiva: `simple_sep`
- Clase negativa: `marqeur`

## 4. Resultados exactos

### Métricas principales

| Experimento | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| baseline_pixels | 1.0 | 1.0 | 1.0 | 1.0 |
| clip_logreg | 1.0 | 1.0 | 1.0 | 1.0 |

- Hubo **empate exacto** entre ambos experimentos en las cuatro métricas principales.

### Matrices de confusión en números

Convención: filas = clase real, columnas = clase predicha, en el orden `[marqeur, simple_sep]`.

- `baseline_pixels`: `[[82, 0], [0, 32]]`
- `clip_logreg`: `[[82, 0], [0, 32]]`

### Classification report por clase

- En ambos experimentos:
  - `marqeur`: precision = 1.0, recall = 1.0, f1 = 1.0, support = 82
  - `simple_sep`: precision = 1.0, recall = 1.0, f1 = 1.0, support = 32
  - macro avg = 1.0
  - weighted avg = 1.0

### Thresholds: mejor threshold o rango razonable

- `baseline_pixels`:
  - no hay un único mejor threshold
  - el desempeño es perfecto para thresholds entre **0.10 y 0.95**
  - a threshold `0.05` baja levemente a accuracy `0.9912`, precision `0.9697`, recall `1.0`, f1 `0.9846`
- `clip_logreg`:
  - no hay un único mejor threshold
  - el desempeño es perfecto para thresholds entre **0.40 y 0.55**
  - por debajo de `0.40` aumentan falsos positivos
  - por encima de `0.55` cae el recall; entre `0.85` y `0.95` precision, recall y f1 quedan en `0.0`

### Ejemplos seleccionados

- `baseline_pixels_examples.csv`: 6 ejemplos `TP` y 6 ejemplos `TN`
- `clip_logreg_examples.csv`: 6 ejemplos `TP` y 6 ejemplos `TN`
- No hay ejemplos `FP` ni `FN` guardados, consistente con las matrices de confusión sin errores.

## 5. Observaciones técnicas útiles para redactar

- La interpretación prudente es que el resultado perfecto probablemente se debe a que el experimento trabaja con dos clases muy separables en un escenario simplificado.
- Este resultado **no** resuelve el problema completo del proyecto.
- El pipeline del Hito 1 trabaja con **recortes ya separados por clase**, no con páginas completas.
- Todavía no se evaluó retrieval sobre páginas, ranking ni localización espacial.
- En threshold `0.5`, ambos experimentos rinden perfecto.
- El baseline quedó perfecto en un rango mucho más amplio de thresholds que CLIP; CLIP mostró mayor sensibilidad fuera de la zona central.
- Si se redacta una discusión, conviene remarcar que el empate en métricas no implica equivalencia total en calibración.

## 6. Figuras disponibles para el informe

| Archivo | Qué muestra | Sección sugerida del informe |
| --- | --- | --- |
| `outputs/figures/class_distribution.png` | Distribución de las clases con más imágenes válidas | Descripción del dataset |
| `outputs/figures/baseline_pixels_confusion_matrix.png` | Matriz de confusión del baseline de píxeles | Resultados / Experimentos |
| `outputs/figures/clip_logreg_confusion_matrix.png` | Matriz de confusión del experimento CLIP + LogReg | Resultados / Experimentos |
| `outputs/figures/baseline_pixels_threshold_curves.png` | Curvas de precision, recall y F1 vs threshold para el baseline | Resultados / Análisis de threshold |
| `outputs/figures/clip_logreg_threshold_curves.png` | Curvas de precision, recall y F1 vs threshold para CLIP + LogReg | Resultados / Análisis de threshold |
| `outputs/figures/baseline_pixels_examples.png` | Ejemplos TP/TN del baseline | Resultados cualitativos / Análisis de casos |
| `outputs/figures/clip_logreg_examples.png` | Ejemplos TP/TN de CLIP + LogReg | Resultados cualitativos / Análisis de casos |

## 7. Tablas sugeridas para el informe

### Tabla sugerida 1: resumen del dataset

Objetivo: resumir el material disponible y justificar la selección de clases.

Columnas sugeridas:

- Dataset / subconjunto
- Contenido
- Cantidad relevante
- Uso en el Hito 1

Contenido sugerido:

- `DocExplore_images` | páginas completas | 1500 páginas válidas | contexto para trabajo futuro
- `DocExplore_queries_web` | queries por clase | 35 clases | base experimental del Hito 1
- `evaluation_kit_v2` | archivos auxiliares | uso no inferido con certeza en este hito | no usado en los experimentos

### Tabla sugerida 2: comparación de resultados

Objetivo: comparar ambos pipelines en una sola vista.

Columnas sugeridas:

- Experimento
- Representación de entrada
- Train size
- Test size
- Clase negativa
- Clase positiva
- Accuracy
- Precision
- Recall
- F1
- Rango de threshold perfecto

Contenido sugerido:

- `baseline_pixels` | RGB 64x64 + flatten + `StandardScaler` | 455 | 114 | `marqeur` | `simple_sep` | 1.0 | 1.0 | 1.0 | 1.0 | 0.10 a 0.95
- `clip_logreg` | embeddings CLIP 512-d | 455 | 114 | `marqeur` | `simple_sep` | 1.0 | 1.0 | 1.0 | 1.0 | 0.40 a 0.55
