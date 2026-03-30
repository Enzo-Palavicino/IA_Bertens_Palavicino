# Contexto del proyecto - Hito 1

## Objetivo general del proyecto
El proyecto final consiste en trabajar con documentos históricos y queries visuales. Más adelante se buscará encontrar una query dentro de páginas completas de documentos, idealmente determinando en qué página aparece y su ubicación.

## Alcance del Hito 1
En este hito NO se implementará todavía retrieval completo sobre páginas ni localización espacial.

El objetivo del Hito 1 es:
1. familiarizarse con el dataset,
2. explorar la estructura de las imágenes,
3. seleccionar 2 clases de queries,
4. extraer embeddings con CLIP,
5. entrenar una Regresión Logística,
6. evaluar resultados con métricas de clasificación,
7. dejar evidencia de experimentos similar al Lab3.

## Dataset
El dataset contiene:
- `data/raw/DocExplore_images/`: páginas completas de documentos históricos.
- `data/raw/DocExplore_queries_web/`: imágenes de queries organizadas por subcarpetas, donde cada subcarpeta representa una clase.
- `data/raw/evaluation_kit_v2/`: existe, pero no es el foco del Hito 1.

## Observaciones ya conocidas
- `DocExplore_queries_web` tiene clases explícitas por carpeta.
- Hay aproximadamente 35 clases.
- Las clases con más ejemplos observadas son:
  - marqeur
  - simple_sep
  - S
  - losange
  - croix
  - double_sep

## Experimento principal esperado
El experimento principal debe usar por defecto:
- clase 0: `marqeur`
- clase 1: `simple_sep`

Debe quedar parametrizable para cambiar esas clases luego.

## Referencia metodológica
Existe un notebook de referencia:
- `references/Lab3_RegresionLogistica.ipynb`

Ese notebook incluye:
- exploración de datos,
- clasificación binaria,
- split train/test,
- métricas,
- matriz de confusión,
- ejemplos correctos/incorrectos,
- análisis de threshold,
- y una sección de CLIP + Regresión Logística.

El nuevo código debe inspirarse en esa lógica, pero adaptado al dataset DocExplore.

## Qué NO hacer todavía
No implementar todavía:
- retrieval completo sobre todas las páginas,
- búsqueda de la query dentro de cada página,
- bounding boxes,
- sliding windows,
- mAP de detección o localización.

## Qué SÍ hacer
- exploración del dataset,
- conteo de clases,
- selección de dos clases,
- baseline simple opcional con píxeles,
- CLIP como extractor de embeddings,
- Regresión Logística como clasificador,
- evaluación con accuracy, precision, recall, f1, matriz de confusión y threshold analysis.