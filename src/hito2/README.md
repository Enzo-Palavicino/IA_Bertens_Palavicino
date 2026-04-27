# Hito 2

Base de trabajo para la etapa multiclase single-label del proyecto.

## Alcance de esta carpeta

- `data/`: manifest, splits y datasets reutilizables.
- `models/`: reservado para `CLIP encoder + MLP` y `CNN`.
- `trainers/`: reservado para los loops de entrenamiento posteriores.
- `utils/`: reservado para helpers compartidos del Hito 2.

## Supuestos del dataset

- Cada subdirectorio inmediato bajo `data/raw/DocExplore_queries_web/` se interpreta como una clase.
- Cada imagen válida dentro de ese subdirectorio es una muestra de esa clase.
- `data/raw/DocExplore_images/` contiene páginas completas del corpus, pero no se usa todavía en este loader multiclase.
- El manifest guarda `image_path` relativo al repositorio para evitar rutas absolutas dependientes de una máquina.

## Política de `class_id`

- Los `class_id` se asignan con orden lexicográfico sobre `class_name`.
- Esa regla hace la asignación reproducible mientras no cambie el conjunto de clases detectadas.

## Política de splits

- Se intenta un split `train/val/test` balanceado por clase.
- Si una clase tiene 3 o más imágenes, cada split recibe al menos una muestra.
- Si una clase tiene 2 imágenes, no es posible un split estratificado perfecto en tres partes; el fallback actual es `1 train + 1 test`.
- Si una clase tiene 1 imagen, el fallback actual es dejarla en `train`.
- Estas excepciones quedan registradas en el resumen JSON de splits.

## Loader listo para conectar

- `DocExploreClipDataset`: entrega imagen RGB cruda o `clip_inputs` si se pasa un `processor`.
- `DocExploreCNNDataset`: aplica resize directo a un tamaño fijo configurable y devuelve tensor `float32` en rango `[0, 1]`.

## Pipeline CNN

- `python -m src.hito2.train_cnn`
- Usa `query_splits.csv` existente; no recalcula splits dentro del entrenamiento.
- Usa resize directo sin padding de entrada; el tamano por defecto es `128x128`.
- Entrena una CNN baseline con bloques convolucionales `Conv-BN-ReLU-MaxPool`, `AdaptiveAvgPool` y un clasificador fully connected con dropout.
- Por defecto aplica pesos inversos a la frecuencia de clases calculados solo sobre train para reducir el sesgo por desbalance.
- No usa augmentations por defecto, para que la comparacion con CLIP + MLP sea mas limpia.

## Outputs del experimento CNN

- `outputs/hito2/cnn/metrics/`
- `outputs/hito2/cnn/figures/`
- `outputs/hito2/cnn/models/`
- `outputs/hito2/cnn/predictions/`

Archivos principales:

- `metrics/val_metrics.json`
- `metrics/test_metrics.json`
- `metrics/val_classification_report.json`
- `metrics/test_classification_report.json`
- `metrics/training_history.csv`
- `metrics/experiment_summary.md`
- `figures/val_confusion_matrix.png`
- `figures/test_confusion_matrix.png`
- `models/best_model.pt`
- `models/experiment_config.json`
- `predictions/val_predictions.csv`
- `predictions/test_predictions.csv`

## Comparacion Final

- `python -m src.hito2.compare_models`
- Reutiliza unicamente los artefactos ya existentes en `outputs/hito2/clip_mlp/` y `outputs/hito2/cnn/`.
- No reentrena ningun modelo.
- Verifica que ambos enfoques usen los mismos splits y el mismo orden de clases antes de comparar.

Outputs principales:

- `outputs/hito2/comparison/comparison_metrics.csv`
- `outputs/hito2/comparison/comparison_metrics.json`
- `outputs/hito2/comparison/comparison_summary.md`

## Pipeline CLIP + MLP

- `python -m src.hito2.train_clip_mlp`
- Usa `query_splits.csv` existente; no recalcula splits dentro del entrenamiento.
- Extrae embeddings con `openai/clip-vit-base-patch32` usando el preprocess oficial de CLIP.
- Cachea embeddings por split en `data/processed/clip_embeddings/hito2/clip_mlp/`.
- Entrena solo una cabeza MLP sobre embeddings congelados, con `CrossEntropyLoss`, `Adam`, seed fija y early stopping por `val_loss`.
- Por defecto aplica pesos inversos a la frecuencia de clases calculados solo sobre train para reducir el sesgo por desbalance.

## Outputs del experimento CLIP + MLP

- `outputs/hito2/clip_mlp/metrics/`
- `outputs/hito2/clip_mlp/figures/`
- `outputs/hito2/clip_mlp/models/`
- `outputs/hito2/clip_mlp/predictions/`

Archivos principales:

- `metrics/val_metrics.json`
- `metrics/test_metrics.json`
- `metrics/val_classification_report.json`
- `metrics/test_classification_report.json`
- `metrics/training_history.csv`
- `metrics/experiment_summary.md`
- `figures/val_confusion_matrix.png`
- `figures/test_confusion_matrix.png`
- `models/best_model.pt`
- `models/experiment_config.json`
- `predictions/val_predictions.csv`
- `predictions/test_predictions.csv`

## Consideraciones metodologicas

- La CNN y CLIP + MLP usan exactamente el mismo `query_splits.csv`, para que la comparacion sea justa y sin leakage entre enfoques.
- El resize directo sin padding mantiene el preprocesamiento simple y reproducible, pero puede distorsionar la relacion de aspecto de algunas queries; esa limitacion debe mencionarse al comparar contra CLIP.
- La CNN no usa augmentations por defecto, para evitar introducir una fuente extra de variacion al compararla con un MLP sobre embeddings congelados.
- No hay data leakage entre splits: el MLP se ajusta solo con train y el early stopping usa exclusivamente validacion.
- El cache de embeddings no altera los splits; solo evita recomputar el encoder CLIP cuando las rutas del split no cambian.
- Validacion no contiene `obj_3`, `obj_36`, `obj_37` ni `obj_42`, porque esas clases tienen solo 2 muestras totales; por eso el reporte de val incluye clases con soporte 0.
- La comparacion futura contra CNN debe considerar esta misma limitacion del dataset raro para que el contraste sea justo.

## Comando base

```bash
python -m src.hito2.prepare_data
```

Esto genera por defecto:

- `outputs/hito2/manifests/query_manifest.csv`
- `outputs/hito2/splits/query_splits.csv`
- `outputs/hito2/splits/query_splits_summary.json`
