# IA_Bertens_Palavicino

## Hito 1

El Hito 1 se enfoca en preparar una primera base de trabajo sobre el dataset DocExplore. En esta fase inicial se deja lista la estructura del proyecto y una inspeccion del dataset para identificar clases disponibles, cantidad de ejemplos por clase y cantidad de paginas completas disponibles.

Todavia no se implementa:
- retrieval sobre paginas completas
- localizacion dentro de pagina
- bounding boxes
- CLIP
- entrenamiento del pipeline final

## Estructura del repositorio

```text
.
├── data/
│   └── raw/
│       ├── DocExplore_images/
│       ├── DocExplore_queries_web/
│       └── evaluation_kit_v2/
├── docs/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── references/
└── src/
```

## Dataset

El dataset debe estar disponible localmente en:

- `data/raw/DocExplore_queries_web/`: queries organizadas por clase en subcarpetas
- `data/raw/DocExplore_images/`: paginas completas del corpus
- `data/raw/evaluation_kit_v2/`: material auxiliar de evaluacion

La configuracion base del proyecto usa por defecto las clases:

- `marqeur`
- `simple_sep`

## Inspeccion inicial del dataset

En esta fase ya existe un script de inspeccion para:

- detectar clases por nombre de carpeta
- contar imagenes validas por clase
- ignorar archivos no imagen y archivos corruptos
- guardar conteos en `outputs/metrics/class_counts.csv`
- guardar un grafico en `outputs/figures/class_distribution.png`
- contar paginas completas validas en `data/raw/DocExplore_images/`

## Como ejecutar

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Ejecuta la inspeccion:

```bash
python src/dataset_inspection.py
```

Si tu entorno no expone el alias `python`, usa:

```bash
python3 src/dataset_inspection.py
```

## Nota sobre el alcance

Esta etapa solo deja lista la base del proyecto y la inspeccion del dataset. La etapa de retrieval y la busqueda/localizacion de queries dentro de una pagina completa quedan fuera del alcance actual.
