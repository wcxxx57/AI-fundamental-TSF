from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd


DEFAULT_TEXT_COLUMN = "Random_Text"
DEFAULT_SOURCE_TEXT_COLUMN = "Final_Search_4"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
SENTENCE_PUNCTUATION = [".", ".", ".", "!", "?"]
INLINE_PUNCTUATION = [",", "", "", "", ""]


def resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def split_columns(value: str) -> list[str]:
    return [col.strip() for col in value.split(",") if col.strip()]


def stable_seed(seed: int, row_index: int, salt: str) -> int:
    raw = f"{seed}:{row_index}:{salt}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def source_text_for_row(row: pd.Series, source_columns: list[str]) -> str:
    parts = []
    for col in source_columns:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))
    return " ".join(parts)


def random_char_sentence_text(target_chars: int, rng: random.Random) -> str:
    """Generate English-like random character text from letters, spaces, and punctuation only."""
    parts: list[str] = []
    sentence_words = 0
    current_len = 0
    while current_len < target_chars:
        word_len = rng.randint(2, 9)
        word = "".join(rng.choice(LETTERS) for _ in range(word_len))
        if sentence_words == 0:
            word = word.capitalize()

        sentence_words += 1
        if sentence_words >= rng.randint(7, 16):
            word += rng.choice(SENTENCE_PUNCTUATION)
            sentence_words = 0
        elif rng.random() < 0.08:
            word += rng.choice(INLINE_PUNCTUATION)

        candidate = word if not parts else " " + word
        parts.append(candidate)
        current_len += len(candidate)

    text = "".join(parts).strip()
    if len(text) > target_chars:
        text = text[:target_chars].rstrip(" ,")
    if text and text[-1] not in ".!?":
        if len(text) >= target_chars:
            text = text[:-1].rstrip(" ,")
        text += "."
    return text


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic random-text negative-control CSV for fighting MM-TSFlib fusion runs."
    )
    parser.add_argument("--input", required=True, help="Input Time-MMD-style CSV path.")
    parser.add_argument("--output", default="", help="Output CSV path. Defaults to <input>_random_text.csv.")
    parser.add_argument("--audit-output", default="", help="Audit JSONL path.")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument(
        "--source-text-column",
        default=DEFAULT_SOURCE_TEXT_COLUMN,
        help="Comma-separated source text column(s) used only for optional length matching.",
    )
    parser.add_argument("--seed", type=int, default=20240624)
    parser.add_argument(
        "--mode",
        choices=["char_noise"],
        default="char_noise",
        help="char_noise samples only letters, spaces, and punctuation.",
    )
    parser.add_argument(
        "--length-mode",
        choices=["source", "fixed"],
        default="source",
        help="source matches each row's source text character length; fixed uses --fixed-chars.",
    )
    parser.add_argument("--fixed-chars", type=int, default=320)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    input_path = resolve_path(args.input, cwd)
    output_path = (
        resolve_path(args.output, cwd)
        if args.output
        else input_path.with_name(input_path.stem + "_random_text.csv")
    )
    audit_path = (
        resolve_path(args.audit_output, cwd)
        if args.audit_output
        else Path(__file__).resolve().parents[1]
        / "results"
        / "random_text_control"
        / f"{input_path.stem}_{args.text_column}_seed{args.seed}_audit.jsonl"
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to rebuild.")
    if args.min_chars <= 0 or args.max_chars < args.min_chars:
        raise ValueError("--min-chars/--max-chars are inconsistent.")

    df = pd.read_csv(input_path)
    source_columns = split_columns(args.source_text_column)
    missing_source = [col for col in source_columns if col not in df.columns]
    if missing_source:
        print(f"Warning: missing source text columns {missing_source}; source-based lengths will use --fixed-chars.")

    audit_rows: list[dict[str, object]] = []
    random_values: list[str] = []
    for row_index, row in df.iterrows():
        rng = random.Random(stable_seed(args.seed, int(row_index), args.text_column))
        source_text = source_text_for_row(row, source_columns)
        source_char_count = len(source_text)

        if args.length_mode == "source":
            target_chars = max(args.min_chars, min(args.max_chars, source_char_count or args.fixed_chars))
        else:
            target_chars = max(args.min_chars, min(args.max_chars, args.fixed_chars))
        text = random_char_sentence_text(target_chars, rng)

        random_values.append(text)
        audit_rows.append(
            {
                "row_index": int(row_index),
                "input_csv": str(input_path),
                "output_csv": str(output_path),
                "text_column": args.text_column,
                "source_text_columns": source_columns,
                "seed": int(args.seed),
                "mode": args.mode,
                "length_mode": args.length_mode,
                "source_char_count": int(source_char_count),
                "random_char_count": int(len(text)),
                "random_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "leakage_check": {
                    "uses_target_values": False,
                    "uses_future_rows": False,
                    "rule": "Text is generated from deterministic random letters, spaces, and punctuation; source columns are used only for length control.",
                },
            }
        )

    df[args.text_column] = random_values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    write_jsonl(audit_path, audit_rows)

    print(f"Wrote CSV: {output_path}")
    print(f"Wrote audit JSONL: {audit_path}")
    print(f"Rows: {len(df)}")
    print(f"Random text column: {args.text_column}")
    print("Random text mode: char_noise (letters, spaces, punctuation)")


if __name__ == "__main__":
    main()
