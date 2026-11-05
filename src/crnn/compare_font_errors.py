import csv
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.paths import DATA_DIR, PROJECT_ROOT


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def compare_font_errors(errors_path, usage_path, fonts_path, output_path):
    errors = read_csv(errors_path)
    usage = read_csv(usage_path)
    fonts = read_csv(fonts_path)

    font_by_sample = {row["sample_key"]: row["font_id"] for row in usage}
    path_by_font = {row["font_id"]: row["font_path"] for row in fonts}
    sample_counts = Counter(font_by_sample.values())
    errors_by_font = Counter()
    distance_by_font = Counter()
    edits_by_font = defaultdict(Counter)

    for row in errors:
        sample_key = Path(row["image_path"]).stem
        font_id = font_by_sample.get(sample_key)
        if font_id is None:
            raise ValueError(f"No font metadata found for sample {sample_key}")

        errors_by_font[font_id] += 1
        distance_by_font[font_id] += int(row["distance"])
        for edit in ("replace", "delete", "insert"):
            edits_by_font[font_id][edit] += int(row[edit])

    headers = [
        "font_id",
        "font_path",
        "samples",
        "errors",
        "error_rate",
        "exact_match_rate",
        "total_edit_distance",
        "mean_distance_per_error",
        "mean_distance_per_sample",
        "replace",
        "delete",
        "insert",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for font_id, sample_count in sorted(
            sample_counts.items(),
            key=lambda item: (
                errors_by_font[item[0]] / item[1] if item[1] else 0,
                distance_by_font[item[0]] / item[1] if item[1] else 0,
            ),
            reverse=True,
        ):
            error_count = errors_by_font[font_id]
            total_distance = distance_by_font[font_id]
            writer.writerow(
                [
                    font_id,
                    path_by_font.get(font_id, ""),
                    sample_count,
                    error_count,
                    error_count / sample_count if sample_count else 0,
                    (sample_count - error_count) / sample_count if sample_count else 0,
                    total_distance,
                    total_distance / error_count if error_count else 0,
                    total_distance / sample_count if sample_count else 0,
                    edits_by_font[font_id]["replace"],
                    edits_by_font[font_id]["delete"],
                    edits_by_font[font_id]["insert"],
                ]
            )


if __name__ == "__main__":
    dataset_dir = DATA_DIR / "words-dataset"
    metadata_dir = dataset_dir / "metadata"
    compare_font_errors(
        PROJECT_ROOT / "artifacts" / "evaluate_errors.csv",
        metadata_dir / "test_font_usage.csv",
        metadata_dir / "test_fonts.csv",
        PROJECT_ROOT / "artifacts" / "font_error_summary.csv",
    )
