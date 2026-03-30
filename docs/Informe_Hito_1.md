# Informe Hito 1 - DocExplore

## 1. Definición del problema

El proyecto busca trabajar con documentos históricos y queries visuales. El objetivo final es que, dada una query visual, el sistema pueda determinar en qué página de un documento aparece y, en etapas posteriores, aproximar también su ubicación dentro de la página.

En este Hito 1 no se abordó todavía el problema completo de retrieval sobre páginas completas ni la localización espacial. El foco estuvo en construir una primera base experimental controlada para clasificación binaria de queries, de modo de validar el flujo de carga de datos, extracción de representaciones, entrenamiento de modelos y evaluación con métricas clásicas.

## 2. Definición de objetivos

### Objetivo general

Desarrollar una primera aproximación reproducible para clasificar queries visuales del dataset DocExplore mediante Regresión Logística, comparando un baseline basado en píxeles con una variante basada en embeddings CLIP.

### Objetivos específicos

- Familiarizarse con la estructura general del dataset.
- Explorar las clases disponibles en `DocExplore_queries_web`.
- Seleccionar dos clases de queries para un experimento inicial.
- Implementar una primera aproximación con Regresión Logística.
- Comparar un baseline con píxeles frente a CLIP + Regresión Logística.

## 3. Descripción de los datasets

El repositorio incluye tres fuentes de datos relevantes. `DocExplore_images` contiene las páginas completas de documentos históricos y corresponde al insumo que será importante en etapas futuras de retrieval. En la inspección realizada para este hito se verificaron 1500 páginas completas válidas. `DocExplore_queries_web` contiene recortes de queries organizados por carpetas, donde cada subcarpeta representa una clase. En esta colección se detectaron 35 clases válidas. Finalmente, `evaluation_kit_v2` está disponible como material de apoyo, pero no fue el foco de los experimentos del Hito 1.

De acuerdo con `outputs/metrics/class_counts.csv`, las clases con más ejemplos son `marqeur` (409), `simple_sep` (160), `S` (147), `losange` (117), `croix` (92) y `double_sep` (87). Para el experimento inicial se eligieron `marqeur` y `simple_sep`, principalmente porque son las clases con mayor cantidad de ejemplos y permiten partir con una comparación binaria simple y bien balanceada metodológicamente. Además, al revisar las muestras y los resultados obtenidos, ambas clases muestran diferencias visuales claras, lo que las hace adecuadas para una primera línea base.

## 4. Avances

Durante este hito se dejó estructurado el repositorio para trabajar de forma modular, se incorporó el dataset en la estructura esperada, se implementó la inspección inicial del conjunto de queries, se validó la carga correcta de imágenes y se construyeron scripts reutilizables para preprocesamiento, extracción de embeddings, entrenamiento y evaluación. También se generó un notebook reproducible como evidencia del trabajo realizado y se guardaron métricas, tablas y figuras en carpetas de salida para ambos experimentos.

## 5. Modelo de LogReg y experimentos iniciales

El experimento se planteó como una clasificación binaria entre las clases `marqeur` y `simple_sep`. A partir de las 569 imágenes válidas de estas dos clases, se realizó un train/test split estratificado de 80/20, lo que produjo 455 muestras de entrenamiento y 114 de prueba. En el conjunto de prueba hubo 82 ejemplos de `marqeur` y 32 de `simple_sep`.

Se evaluaron dos variantes. La primera fue `baseline_pixels`, donde cada imagen se convirtió a RGB, se redimensionó a 64x64, se aplanó a un vector de píxeles, se escaló con `StandardScaler` y luego se entrenó una Regresión Logística. La segunda fue `clip_logreg`, donde cada imagen se procesó con `openai/clip-vit-base-patch32` para extraer embeddings de 512 dimensiones, y sobre esos embeddings se entrenó otra Regresión Logística.

Los resultados obtenidos fueron los siguientes:

| Experimento | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| baseline_pixels | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| clip_logreg | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Ambos experimentos alcanzaron exactamente el mismo desempeño en el conjunto de prueba. Las matrices de confusión también fueron idénticas: 82 ejemplos de `marqeur` y 32 de `simple_sep` fueron clasificados correctamente, sin falsos positivos ni falsos negativos. Esto coincide con los archivos de ejemplos guardados, donde solo aparecen casos TP y TN, ya que no hubo errores para ilustrar como FP o FN.

El análisis por threshold mostró un matiz interesante. En el baseline, el desempeño se mantuvo perfecto entre thresholds 0.10 y 0.95, y solo cayó levemente en 0.05. En CLIP, el resultado perfecto se mantuvo entre 0.40 y 0.55; por debajo de ese rango aumentaron los falsos positivos y por encima disminuyó el recall. Es decir, a threshold 0.5 ambos métodos rinden igual, pero el baseline quedó más estable frente a variaciones amplias del umbral, mientras que CLIP mostró una calibración más sensible.

El resultado perfecto en ambos enfoques probablemente se explica por la elección de dos clases visualmente muy distinguibles y por el hecho de que este experimento trabaja con recortes de queries ya separados, no con páginas completas. Por lo mismo, estos resultados no implican que el problema general del proyecto esté resuelto. Todavía no se aborda retrieval sobre páginas completas, ranking de páginas ni localización dentro de la página.

## 6. Conclusión breve

En este Hito 1 se logró construir una base experimental completa y reproducible para trabajar con DocExplore. Se inspeccionó el dataset, se seleccionaron dos clases adecuadas para una primera prueba, se implementaron dos pipelines con Regresión Logística y se obtuvo un desempeño perfecto en ambos casos para el escenario binario elegido. Este avance deja lista una base metodológica y de código para enfrentar en los próximos hitos el problema más difícil: buscar queries dentro de páginas completas y, más adelante, estimar también su ubicación.
