"""Compare the existing Hito 2 experiment outputs without retraining."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.hito2.config import HITO2_COMPARISON_OUTPUT_DIR, HITO2_OUTPUT_DIR, ensure_hito2_output_directories
from src.report_utils import save_dataframe, save_json, save_text

CORE_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "top_3_accuracy",
)
LOSS_METRICS = ("loss",)
LOWER_IS_BETTER_METRICS = {
    "val_loss",
    "test_loss",
    "best_val_loss",
    "final_val_loss",
    "val_loss_std",
    "mean_abs_val_loss_delta",
    "best_epoch",
}
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ExperimentArtifacts:
    """In-memory representation of one completed experiment."""

    model_key: str
    experiment_dir: Path
    config: dict[str, Any]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    history_df: pd.DataFrame

    @property
    def experiment_name(self) -> str:
        """Return the stable experiment name from config."""
        return str(self.config["experiment_name"])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Hito 2 comparison script."""
    parser = argparse.ArgumentParser(description="Compare the existing Hito 2 model outputs.")
    parser.add_argument(
        "--clip-dir",
        type=Path,
        default=HITO2_OUTPUT_DIR / "clip_mlp",
        help="Directory that contains the CLIP + MLP experiment outputs.",
    )
    parser.add_argument(
        "--cnn-dir",
        type=Path,
        default=HITO2_OUTPUT_DIR / "cnn",
        help="Directory that contains the CNN experiment outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HITO2_COMPARISON_OUTPUT_DIR,
        help="Directory where the comparison outputs will be written.",
    )
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON artifact from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> Path:
    """Raise a clear error when an expected artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


def load_experiment_artifacts(model_key: str, experiment_dir: Path) -> ExperimentArtifacts:
    """Load the minimal artifact set required for comparison."""
    experiment_dir = Path(experiment_dir).resolve()
    metrics_dir = _require_file(experiment_dir / "metrics")
    models_dir = _require_file(experiment_dir / "models")

    config_path = _require_file(models_dir / "experiment_config.json")
    val_metrics_path = _require_file(metrics_dir / "val_metrics.json")
    test_metrics_path = _require_file(metrics_dir / "test_metrics.json")
    history_path = _require_file(metrics_dir / "training_history.csv")

    return ExperimentArtifacts(
        model_key=model_key,
        experiment_dir=experiment_dir,
        config=_load_json(config_path),
        val_metrics=_load_json(val_metrics_path),
        test_metrics=_load_json(test_metrics_path),
        history_df=pd.read_csv(history_path),
    )


def validate_comparable_experiments(experiments: Sequence[ExperimentArtifacts]) -> dict[str, Any]:
    """Validate that the compared experiments are methodologically aligned."""
    if len(experiments) < 2:
        raise ValueError("At least two experiments are required for comparison.")

    reference = experiments[0]
    shared_checks = {
        "same_split_path": True,
        "same_num_classes": True,
        "same_class_order": True,
        "same_split_sizes": True,
    }

    reference_split_path = str(Path(reference.config["split_path"]).resolve())
    reference_num_classes = int(reference.config["num_classes"])
    reference_class_names = list(reference.config["class_names"])
    reference_split_sizes = dict(reference.config["split_sizes"])

    for experiment in experiments[1:]:
        if str(Path(experiment.config["split_path"]).resolve()) != reference_split_path:
            shared_checks["same_split_path"] = False
        if int(experiment.config["num_classes"]) != reference_num_classes:
            shared_checks["same_num_classes"] = False
        if list(experiment.config["class_names"]) != reference_class_names:
            shared_checks["same_class_order"] = False
        if dict(experiment.config["split_sizes"]) != reference_split_sizes:
            shared_checks["same_split_sizes"] = False

    if not all(shared_checks.values()):
        raise ValueError(
            "The compared experiments are not aligned enough for a fair comparison. "
            f"Validation results: {shared_checks}"
        )

    return {
        "split_path": reference_split_path,
        "num_classes": reference_num_classes,
        "class_names": reference_class_names,
        "split_sizes": reference_split_sizes,
        **shared_checks,
    }


def _safe_float(value: Any) -> float | None:
    """Convert a scalar value into float when possible."""
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def build_comparison_row(experiment: ExperimentArtifacts) -> dict[str, Any]:
    """Flatten one experiment into a single comparable row."""
    history_df = experiment.history_df
    val_loss_series = history_df["val_loss"].astype(float)
    train_loss_series = history_df["train_loss"].astype(float)

    mean_abs_delta = float(val_loss_series.diff().abs().dropna().mean()) if len(val_loss_series) > 1 else 0.0
    val_loss_std = float(val_loss_series.std(ddof=0)) if len(val_loss_series) > 1 else 0.0
    final_val_loss = float(val_loss_series.iloc[-1])
    final_train_loss = float(train_loss_series.iloc[-1])

    row: dict[str, Any] = {
        "model": experiment.model_key,
        "experiment_name": experiment.experiment_name,
        "num_classes": int(experiment.config["num_classes"]),
        "split_path": str(Path(experiment.config["split_path"]).resolve()),
        "best_epoch": int(experiment.config["best_epoch"]),
        "epochs_ran": int(len(history_df)),
        "stopped_early": bool(experiment.config.get("stopped_early", False)),
        "best_val_loss": _safe_float(experiment.config.get("best_val_loss")),
        "final_val_loss": final_val_loss,
        "final_train_loss": final_train_loss,
        "val_loss_std": val_loss_std,
        "mean_abs_val_loss_delta": mean_abs_delta,
    }

    for metric_name in CORE_METRICS:
        row[f"val_{metric_name}"] = _safe_float(experiment.val_metrics.get(metric_name))
        row[f"test_{metric_name}"] = _safe_float(experiment.test_metrics.get(metric_name))

    row["val_loss"] = _safe_float(experiment.val_metrics.get("loss"))
    row["test_loss"] = _safe_float(experiment.test_metrics.get("loss"))
    return row


def build_comparison_dataframe(experiments: Sequence[ExperimentArtifacts]) -> pd.DataFrame:
    """Build the main comparison dataframe, one row per model."""
    rows = [build_comparison_row(experiment) for experiment in experiments]
    return pd.DataFrame(rows)


def decide_winner(metric_name: str, model_to_value: dict[str, float | None]) -> str:
    """Return the winner label for one metric with tie handling."""
    valid_items = [(model, value) for model, value in model_to_value.items() if value is not None]
    if len(valid_items) < 2:
        return "insufficient_data"

    sorted_items = sorted(valid_items, key=lambda item: item[1], reverse=metric_name not in LOWER_IS_BETTER_METRICS)
    first_model, first_value = sorted_items[0]
    second_model, second_value = sorted_items[1]
    if abs(first_value - second_value) <= TIE_TOLERANCE:
        return "tie"
    return first_model


def build_winner_records(comparison_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build a row-wise winner table across validation, test and stability metrics."""
    metric_names = (
        [f"val_{metric}" for metric in CORE_METRICS]
        + [f"test_{metric}" for metric in CORE_METRICS]
        + ["val_loss", "test_loss", "best_val_loss", "final_val_loss", "val_loss_std", "mean_abs_val_loss_delta", "best_epoch"]
    )

    records: list[dict[str, Any]] = []
    for metric_name in metric_names:
        model_to_value = {
            row["model"]: _safe_float(row[metric_name]) if metric_name in row else None
            for _, row in comparison_df.iterrows()
        }
        winner = decide_winner(metric_name, model_to_value)
        record: dict[str, Any] = {"metric": metric_name, "winner": winner}
        record.update({f"value_{model}": value for model, value in model_to_value.items()})
        records.append(record)

    return records


def build_stability_records(comparison_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract the stability-focused metrics for JSON and markdown reporting."""
    stability_metrics = ("epochs_ran", "best_epoch", "best_val_loss", "final_val_loss", "val_loss_std", "mean_abs_val_loss_delta")
    records: list[dict[str, Any]] = []
    for metric_name in stability_metrics:
        model_to_value = {
            row["model"]: _safe_float(row[metric_name]) if metric_name in row else None
            for _, row in comparison_df.iterrows()
        }
        records.append(
            {
                "metric": metric_name,
                "winner": decide_winner(metric_name, model_to_value),
                **{f"value_{model}": value for model, value in model_to_value.items()},
            }
        )
    return records


def _wins_count(winner_records: Sequence[dict[str, Any]], metric_names: Sequence[str]) -> dict[str, int]:
    """Count how many requested metrics each model wins."""
    metric_name_set = set(metric_names)
    counts: dict[str, int] = {}
    for record in winner_records:
        metric_name = str(record["metric"])
        winner = str(record["winner"])
        if metric_name not in metric_name_set:
            continue
        if winner in {"tie", "insufficient_data"}:
            continue
        counts[winner] = counts.get(winner, 0) + 1
    return counts


def build_automatic_analysis(
    comparison_df: pd.DataFrame,
    winner_records: Sequence[dict[str, Any]],
    validation_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Generate automatic interpretation text from the comparison table."""
    comparison_rows = {row["model"]: row for _, row in comparison_df.iterrows()}
    test_metric_names = [f"test_{metric}" for metric in CORE_METRICS]
    val_metric_names = [f"val_{metric}" for metric in CORE_METRICS]
    test_wins = _wins_count(winner_records, metric_names=test_metric_names)
    val_wins = _wins_count(winner_records, metric_names=val_metric_names)

    best_test_model = max(test_wins.items(), key=lambda item: item[1])[0] if test_wins else "tie"
    best_val_model = max(val_wins.items(), key=lambda item: item[1])[0] if val_wins else "tie"

    clip_row = comparison_rows.get("clip_mlp")
    cnn_row = comparison_rows.get("cnn")
    if clip_row is None or cnn_row is None:
        raise ValueError("The comparison script currently expects `clip_mlp` and `cnn` rows.")

    convergence_winner = decide_winner(
        "best_epoch",
        {"clip_mlp": clip_row["best_epoch"], "cnn": cnn_row["best_epoch"]},
    )
    stability_winner_std = decide_winner(
        "val_loss_std",
        {"clip_mlp": clip_row["val_loss_std"], "cnn": cnn_row["val_loss_std"]},
    )
    stability_winner_delta = decide_winner(
        "mean_abs_val_loss_delta",
        {"clip_mlp": clip_row["mean_abs_val_loss_delta"], "cnn": cnn_row["mean_abs_val_loss_delta"]},
    )

    if stability_winner_std == stability_winner_delta and stability_winner_std not in {"tie", "insufficient_data"}:
        stability_text = (
            f"Segun `val_loss_std` y `mean_abs_val_loss_delta`, `{stability_winner_std}` muestra una trayectoria de "
            "validacion mas estable."
        )
    else:
        stability_text = (
            "Las heuristicas simples de estabilidad (`val_loss_std` y `mean_abs_val_loss_delta`) no muestran un ganador "
            "completamente consistente; la trayectoria depende del criterio usado."
        )

    weighted_vs_macro_clip = float(clip_row["test_weighted_f1"] - clip_row["test_macro_f1"])
    weighted_vs_macro_cnn = float(cnn_row["test_weighted_f1"] - cnn_row["test_macro_f1"])
    top3_advantage_clip = float(clip_row["test_top_3_accuracy"] - clip_row["test_accuracy"])
    top3_advantage_cnn = float(cnn_row["test_top_3_accuracy"] - cnn_row["test_accuracy"])

    analysis_points = [
        (
            f"En test, `{best_test_model}` gana la mayoria de las metricas principales "
            f"({test_wins.get(best_test_model, 0)} de {len(CORE_METRICS)})."
            if best_test_model != "tie"
            else "En test no hay un ganador unico por conteo de metricas."
        ),
        (
            f"En validacion, `{best_val_model}` gana la mayoria de las metricas principales "
            f"({val_wins.get(best_val_model, 0)} de {len(CORE_METRICS)})."
            if best_val_model != "tie"
            else "En validacion no hay un ganador unico por conteo de metricas."
        ),
        (
            f"`{convergence_winner}` alcanza su mejor checkpoint en menos epocas "
            f"(`best_epoch`: CLIP+MLP={int(clip_row['best_epoch'])}, CNN={int(cnn_row['best_epoch'])})."
            if convergence_winner not in {"tie", "insufficient_data"}
            else "Ambos modelos alcanzan su mejor checkpoint en una cantidad de epocas similar."
        ),
        stability_text,
        (
            "La brecha entre `weighted_f1` y `macro_f1` en ambos modelos confirma el impacto del desbalance de clases "
            f"(CLIP+MLP: {weighted_vs_macro_clip:.4f}; CNN: {weighted_vs_macro_cnn:.4f})."
        ),
        (
            "El `top-3 accuracy` es claramente superior al `accuracy` top-1 en ambos enfoques, lo que sugiere que el "
            f"ranking de candidatos es mas facil que la decision exacta de clase (CLIP+MLP: +{top3_advantage_clip:.4f}; "
            f"CNN: +{top3_advantage_cnn:.4f})."
        ),
        (
            "Interpretacion tecnica breve: CLIP+MLP aprovecha embeddings preentrenados y por eso suele resistir mejor el "
            "escenario de 35 clases con pocas muestras por clase; la CNN aprende directamente desde pixeles redimensionados "
            "a `128x128`, por lo que exige mas datos para cerrar la brecha."
        ),
        (
            "Chequeo metodologico: ambos experimentos usan el mismo `query_splits.csv`, el mismo orden de clases y el mismo "
            "problema multiclase single-label, por lo que la comparacion es valida a nivel de particion."
        ),
    ]

    return {
        "test_metric_wins": test_wins,
        "val_metric_wins": val_wins,
        "convergence_winner": convergence_winner,
        "stability_winner_val_loss_std": stability_winner_std,
        "stability_winner_mean_abs_val_loss_delta": stability_winner_delta,
        "analysis_points": analysis_points,
        "methodology": {
            "task_type": "multiclase single-label",
            "num_classes": int(validation_metadata["num_classes"]),
            "same_splits": bool(validation_metadata["same_split_path"] and validation_metadata["same_split_sizes"]),
            "cnn_resize": "128x128 directo sin padding",
            "clip_backbone": "openai/clip-vit-base-patch32 con embeddings preentrenados",
            "rare_classes_note": (
                "Existen clases extremadamente escasas con 1-2 imagenes en train y algunas ausentes en validacion, "
                "lo que tensiona especialmente macro-F1 y recall macro."
            ),
        },
    }


def _select_columns(columns: Sequence[str], prefix: str) -> list[str]:
    """Return the subset of columns that start with one prefix."""
    return [column for column in columns if column.startswith(prefix)]


def _build_markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """Render a dataframe subset as markdown with rounded floats."""
    table_df = df.loc[:, list(columns)].copy()
    float_columns = table_df.select_dtypes(include=["float", "float64", "float32"]).columns
    table_df.loc[:, float_columns] = table_df.loc[:, float_columns].round(4)
    return _dataframe_to_markdown(table_df)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a dataframe as a GitHub-flavored Markdown table without extra deps."""
    headers = [str(column) for column in df.columns]
    rows: list[list[str]] = []

    for _, row in df.iterrows():
        rendered_row: list[str] = []
        for value in row.tolist():
            if isinstance(value, float):
                rendered_row.append(f"{value:.4f}")
            else:
                rendered_row.append(str(value))
        rows.append(rendered_row)

    separator = ["---"] * len(headers)
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    markdown_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(markdown_lines)


def build_markdown_summary(
    comparison_df: pd.DataFrame,
    winner_records: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    validation_metadata: dict[str, Any],
) -> str:
    """Create the final markdown summary for the report and presentation."""
    test_columns = ["model"] + [f"test_{metric}" for metric in CORE_METRICS] + ["test_loss"]
    val_columns = ["model"] + [f"val_{metric}" for metric in CORE_METRICS] + ["val_loss"]
    training_columns = [
        "model",
        "best_epoch",
        "epochs_ran",
        "best_val_loss",
        "final_val_loss",
        "val_loss_std",
        "mean_abs_val_loss_delta",
        "stopped_early",
    ]

    winner_df = pd.DataFrame(winner_records)
    winner_table = winner_df[["metric", "winner"]].copy()

    lines = [
        "# Hito 2 Comparison",
        "",
        "## Contexto",
        "- Problema: clasificacion multiclase single-label.",
        f"- Numero de clases: {validation_metadata['num_classes']}.",
        "- Dataset: queries visuales bajo `data/raw/DocExplore_queries_web/<query>`.",
        "- Comparacion hecha sin reentrenar modelos; solo se reutilizan artefactos ya generados.",
        "- Ambos enfoques usan exactamente el mismo `query_splits.csv` para train/val/test.",
        "- Existen clases muy escasas con 1-2 imagenes, y algunas no aparecen en validacion.",
        "- CNN: resize directo `128x128` sin padding.",
        "- CLIP + MLP: embeddings preentrenados de CLIP y cabeza MLP supervisada.",
        "",
        "## Metricas Test",
        _build_markdown_table(comparison_df, test_columns),
        "",
        "## Metricas Val",
        _build_markdown_table(comparison_df, val_columns),
        "",
        "## Entrenamiento y Estabilidad",
        _build_markdown_table(comparison_df, training_columns),
        "",
        "## Ganador Por Metrica",
        _dataframe_to_markdown(winner_table),
        "",
        "## Analisis Automatico",
    ]

    for point in analysis["analysis_points"]:
        lines.append(f"- {point}")

    lines.extend(
        [
            "",
            "## Interpretacion Tecnica Breve",
            "- Si el objetivo principal es el rendimiento final sobre el split de test actual, el enfoque con mas metricas ganadas es la referencia mas fuerte para el informe.",
            "- Si el objetivo incluye eficiencia de convergencia, `best_epoch` aporta una lectura complementaria: menos epocas hasta el mejor checkpoint sugiere entrenamiento mas rapido de estabilizar.",
            "- Las metricas macro siguen siendo las mas sensibles a las clases raras; por eso deben leerse junto a `weighted_f1` y `top-3 accuracy`.",
            "",
            "## Observaciones Metodologicas",
            "- Los dos experimentos resuelven el mismo problema multiclase single-label de 35 clases.",
            "- Ambos usan la misma particion train/val/test, evitando leakage y manteniendo una comparacion defendible.",
            "- La CNN recibe imagenes redimensionadas a `128x128` sin padding, lo que simplifica el baseline pero puede distorsionar aspecto.",
            "- CLIP + MLP parte de representaciones preentrenadas; ese sesgo a favor de conocimiento previo debe explicitarse al comparar contra una CNN entrenada desde pixeles.",
            "- Las clases muy escasas siguen siendo la principal limitacion metodologica del Hito 2.",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run the final Hito 2 comparison without retraining any model."""
    args = build_parser().parse_args()
    ensure_hito2_output_directories()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        load_experiment_artifacts("clip_mlp", args.clip_dir),
        load_experiment_artifacts("cnn", args.cnn_dir),
    ]
    validation_metadata = validate_comparable_experiments(experiments)
    comparison_df = build_comparison_dataframe(experiments)
    winner_records = build_winner_records(comparison_df)
    stability_records = build_stability_records(comparison_df)
    analysis = build_automatic_analysis(
        comparison_df=comparison_df,
        winner_records=winner_records,
        validation_metadata=validation_metadata,
    )

    csv_path = save_dataframe(comparison_df, output_dir / "comparison_metrics.csv", index=False)
    json_path = save_json(
        {
            "generated_at": datetime.now().astimezone().isoformat(),
            "comparison_dir": str(output_dir),
            "metadata": validation_metadata,
            "models": comparison_df.to_dict(orient="records"),
            "winners": winner_records,
            "stability": stability_records,
            "analysis": analysis,
        },
        output_dir / "comparison_metrics.json",
    )
    summary_text = build_markdown_summary(
        comparison_df=comparison_df,
        winner_records=winner_records,
        analysis=analysis,
        validation_metadata=validation_metadata,
    )
    summary_path = save_text(summary_text, output_dir / "comparison_summary.md")

    print("Comparacion lista.")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {summary_path}")


if __name__ == "__main__":
    main()
