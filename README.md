# IA_Bertens_Palavicino

## Hito 1

El Hito 1 implementa un pipeline binario reproducible sobre queries visuales del dataset DocExplore. El experimento por defecto compara las clases `marqeur` y `simple_sep` con dos enfoques:

- `baseline_pixels`: RGB redimensionado + flatten + `StandardScaler` + `LogisticRegression`
- `clip_logreg`: embeddings CLIP (`openai/clip-vit-base-patch32`) + `LogisticRegression`

En este hito si se implementa:

- inspeccion inicial del dataset
- clasificacion binaria entre dos clases de queries
- metricas de clasificacion
- matriz de confusion
- analisis por threshold
- seleccion de ejemplos TP / FP / TN / FN
- notebook de evidencia

Todavia no se implementa:

- retrieval sobre paginas completas
- ranking de paginas
- localizacion espacial dentro de pagina
- bounding boxes
- sliding windows
- mAP de deteccion o localizacion

## Estructura relevante

```text
.
├── data/
│   ├── raw/
│   │   ├── DocExplore_images/
│   │   ├── DocExplore_queries_web/
│   │   └── evaluation_kit_v2/
│   └── processed/
│       └── clip_embeddings/
├── notebooks/
│   └── hito1_doc_explore.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
└── src/
```

## Dataset

El dataset debe existir localmente en:

- `data/raw/DocExplore_queries_web/`
- `data/raw/DocExplore_images/`
- `data/raw/evaluation_kit_v2/`

La inspeccion inicial ya confirmo 35 clases y el experimento por defecto usa:

- clase 0: `marqeur`
- clase 1: `simple_sep`

## Instalacion

1. Crear entorno virtual:

```bash
python3 -m venv .venv
```

2. Activarlo:

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

Si tu entorno no expone `python`, usa `python3` en los comandos de ejecucion.

## Ejecucion

### 1. Inspeccion del dataset

```bash
python src/dataset_inspection.py
```

### 2. Pipeline completo del Hito 1

Corre ambos experimentos por defecto:

```bash
python src/run_hito1.py
```

Solo baseline:

```bash
python src/run_hito1.py --experiment baseline
```

Solo CLIP:

```bash
python src/run_hito1.py --experiment clip
```

Cambiar clases:

```bash
python src/run_hito1.py --class-a marqeur --class-b simple_sep
```

Forzar recomputo de embeddings CLIP:

```bash
python src/run_hito1.py --experiment clip --force-recompute-embeddings
```

## Outputs generados

### Inspeccion

- `outputs/metrics/class_counts.csv`
- `outputs/figures/class_distribution.png`

### Baseline de pixeles

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

### CLIP + Logistic Regression

- `data/processed/clip_embeddings/...`
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

### Comparacion final

- `outputs/metrics/hito1_experiment_comparison.csv`
- `outputs/metrics/hito1_experiment_comparison.md`

## Notebook de evidencia

El notebook `notebooks/hito1_doc_explore.ipynb` resume:

- objetivo y alcance del Hito 1
- exploracion del dataset
- seleccion de clases
- baseline con pixeles
- CLIP + Logistic Regression
- metricas
- matriz de confusion
- threshold analysis
- ejemplos TP / FP / TN / FN
- comparacion breve entre ambos enfoques

## Nota sobre CLIP

La primera ejecucion de `clip_logreg` puede tardar mas porque necesita descargar el modelo `openai/clip-vit-base-patch32` si no esta cacheado localmente. Luego los embeddings quedan reutilizables en `data/processed/clip_embeddings/`.
