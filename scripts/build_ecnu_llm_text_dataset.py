from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TEXT_COLUMN = "ECNU_LLM_Text"
DEFAULT_FACT_COLUMN = "ECNU_LLM_Fact"
DEFAULT_PRED_COLUMN = "ECNU_LLM_Pred"
DEFAULT_SYSTEM_PROMPT = (
    "You create leak-free textual features for time-series forecasting. "
    "Use only the historical sequence and variable meanings supplied in the prompt. "
    "Do not use future values, future events, hidden labels, external search, or prior knowledge not grounded in the prompt. "
    "Return valid JSON only."
)


def getenv_first(names: list[str], default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def save_outputs(
    df: pd.DataFrame,
    output_path: Path,
    audit_path: Path,
    text_column: str,
    fact_column: str,
    pred_column: str,
    text_values: list[str],
    fact_values: list[str],
    pred_values: list[str],
    generated_rows: list[dict[str, Any]],
    merge_existing_audit: bool,
) -> None:
    df[fact_column] = fact_values
    df[pred_column] = pred_values
    df[text_column] = text_values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(output_path.name + ".tmp")
    df.to_csv(tmp_output_path, index=False)
    os.replace(tmp_output_path, output_path)

    rows_to_write = generated_rows
    if merge_existing_audit and audit_path.exists():
        with audit_path.open("r", encoding="utf-8") as f:
            previous_rows = [json.loads(line) for line in f if line.strip()]
        by_index = {int(row["row_index"]): row for row in previous_rows}
        for row in generated_rows:
            by_index[int(row["row_index"])] = row
        rows_to_write = [by_index[i] for i in sorted(by_index)]
    write_jsonl(audit_path, rows_to_write)


def load_existing_audit(path: Path, text_column: str, fact_column: str, pred_column: str) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    existing: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_index = int(row["row_index"])
            fact = str(row.get("fact", row.get(fact_column, ""))).strip()
            pred = str(row.get("pred", row.get(pred_column, ""))).strip()
            text = str(row.get("combined_text", row.get(text_column, ""))).strip()
            if not text:
                response = str(row.get("response", "")).strip()
                if response:
                    parsed = parse_fact_pred_response(response)
                    fact = fact or parsed["fact"]
                    pred = pred or parsed["pred"]
                    text = combine_fact_pred(fact, pred)
            if fact or pred or text:
                existing[row_index] = {
                    "fact": fact,
                    "pred": pred,
                    "combined_text": text or combine_fact_pred(fact, pred),
                }
    return existing


def parse_value_columns(df: pd.DataFrame, value_cols: str, target_col: str) -> list[str]:
    if value_cols.strip().lower() == "all_numeric":
        excluded_prefixes = ("Final_", "ECNU_")
        excluded = {"date", "start_date", "end_date"}
        cols = []
        for col in df.columns:
            if col in excluded or col.startswith(excluded_prefixes):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
        return cols
    cols = [col.strip() for col in value_cols.split(",") if col.strip()]
    if not cols:
        cols = [target_col]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing value columns: {missing}")
    return cols


def load_variable_meanings(raw: str, value_cols: list[str]) -> dict[str, str]:
    meanings: dict[str, str] = {}
    if raw:
        content = raw
        try:
            raw_stripped = raw.strip()
            raw_path = Path(raw)
            if raw_stripped.startswith("{"):
                content = raw
            elif raw_path.exists():
                content = raw_path.read_text(encoding="utf-8")
            else:
                content = raw
            parsed = json.loads(content)
        except Exception as exc:
            parsed = {}
            loose_content = content.strip()
            if loose_content.startswith("{") and loose_content.endswith("}") and ":" in loose_content:
                key, value = loose_content[1:-1].split(":", 1)
                parsed[key.strip().strip('"').strip("'")] = value.strip().strip('"').strip("'")
            elif "=" in loose_content:
                for item in loose_content.split("|"):
                    if not item.strip():
                        continue
                    if "=" not in item:
                        raise ValueError(
                            "--variable-meanings items must be JSON, a JSON file, or pipe-separated key=value pairs"
                        ) from exc
                    key, value = item.split("=", 1)
                    parsed[key.strip()] = value.strip()
            else:
                raise ValueError(
                    "--variable-meanings must be JSON, a JSON file, or pipe-separated key=value pairs"
                ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("--variable-meanings JSON must be an object mapping column name to meaning")
        meanings = {str(k): str(v) for k, v in parsed.items()}

    for col in value_cols:
        meanings.setdefault(col, f"{col}. The column name is the only provided variable meaning.")
    return meanings


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.6g}"
    text = str(value).strip()
    return text if text else "NA"


def format_history_sequence(
    hist: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    max_history_rows: int,
) -> tuple[str, int, int]:
    if max_history_rows and len(hist) > max_history_rows:
        hist = hist.iloc[-max_history_rows:]
    lines = []
    for idx, row in hist.iterrows():
        values = ", ".join(f"{col}={format_value(row[col])}" for col in value_cols)
        lines.append(f"row={idx}, date={format_value(row[date_col])}, {values}")
    return "\n".join(lines), int(hist.index[0]), int(hist.index[-1])


def combine_fact_pred(fact: str, pred: str) -> str:
    fact = str(fact).strip()
    pred = str(pred).strip()
    parts = []
    if fact:
        parts.append("Fact: " + fact)
    if pred:
        parts.append("Pred: " + pred)
    return " ".join(parts).strip()


def parse_fact_pred_response(response: str) -> dict[str, str]:
    text = str(response).strip()
    json_text = text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        json_text = fenced.group(1)
    elif "{" in text and "}" in text:
        json_text = text[text.find("{") : text.rfind("}") + 1]

    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            fact = str(parsed.get("fact", parsed.get("facts", ""))).strip()
            pred = str(parsed.get("pred", parsed.get("prediction", parsed.get("analysis", "")))).strip()
            if fact or pred:
                return {"fact": fact, "pred": pred}
    except json.JSONDecodeError:
        pass

    json_like_fact = re.search(r'"fact"\s*:\s*"(.*?)"\s*,\s*"pred"\s*:', text, flags=re.S | re.I)
    json_like_pred = re.search(r'"pred"\s*:\s*"(.*?)(?:"\s*\}|\Z)', text, flags=re.S | re.I)
    if json_like_fact or json_like_pred:
        fact = json_like_fact.group(1).strip() if json_like_fact else ""
        pred = json_like_pred.group(1).strip() if json_like_pred else ""
        return {"fact": fact, "pred": pred}

    fact_match = re.search(r"(?:^|\n)\s*fact\s*:\s*(.*?)(?=\n\s*pred\s*:|\Z)", text, flags=re.S | re.I)
    pred_match = re.search(r"(?:^|\n)\s*pred\s*:\s*(.*)", text, flags=re.S | re.I)
    fact = fact_match.group(1).strip() if fact_match else ""
    pred = pred_match.group(1).strip() if pred_match else ""
    if fact or pred:
        return {"fact": fact, "pred": pred}
    return {"fact": text, "pred": ""}


def build_prompt(
    df: pd.DataFrame,
    row_index: int,
    seq_len: int,
    pred_len: int,
    date_col: str,
    value_cols: list[str],
    target_col: str,
    domain: str,
    max_columns: int,
    variable_meanings: dict[str, str],
    domain_context: str,
    max_history_rows: int,
) -> tuple[str, dict[str, Any]]:
    start_index = max(0, row_index - seq_len + 1)
    hist = df.iloc[start_index : row_index + 1]
    origin_date = str(df.iloc[row_index][date_col])
    history_start_date = str(hist.iloc[0][date_col])
    history_end_date = str(hist.iloc[-1][date_col])

    selected_cols = value_cols[:max_columns]
    sequence_text, serialized_start_index, serialized_end_index = format_history_sequence(
        hist=hist,
        date_col=date_col,
        value_cols=selected_cols,
        max_history_rows=max_history_rows,
    )
    meaning_text = "\n".join(f"- {col}: {variable_meanings[col]}" for col in selected_cols)
    context_text = domain_context.strip() if domain_context.strip() else "No extra domain context is provided."

    prompt = (
        "Create two short English text fields for a multimodal time-series forecasting dataset.\n"
        "Hard leakage rule: use only the historical sequence and variable meanings shown in this prompt. "
        "Do not use future target values, future statistics, future dates, hidden labels, external search, or outside facts.\n"
        "Return valid JSON with exactly these string keys: \"fact\" and \"pred\".\n\n"
        "\"fact\" should describe only observed historical facts in the given sequence: changes, trend shape, volatility, "
        "turning points, anomalies, and level shifts.\n"
        "\"pred\" should be a qualitative forecast-oriented analysis based only on the same sequence and the given variable "
        "meanings: likely continuation/reversal, uncertainty, and domain interpretation. Do not output exact future values.\n\n"
        "Keep each field under 55 words. Do not mention external indicators, events, policies, or facts unless they are explicitly "
        "present in the historical sequence, variable meanings, or domain/context line.\n\n"
        f"Domain: {domain}\n"
        f"Domain/context supplied by the dataset builder: {context_text}\n"
        f"Target column: {target_col}\n"
        f"Forecast horizon length: {pred_len} steps\n"
        f"Historical window row indices: {start_index}..{row_index}\n"
        f"Historical window dates: {history_start_date}..{history_end_date}\n"
        f"Origin timestamp: {origin_date}\n"
        f"Serialized historical sequence row indices: {serialized_start_index}..{serialized_end_index}\n\n"
        "Variable meanings:\n"
        f"{meaning_text}\n\n"
        "Historical sequence, ordered from older to newer:\n"
        f"{sequence_text}\n\n"
        "Output format example:\n"
        "{\"fact\":\"...\", \"pred\":\"...\"}"
    )
    meta = {
        "history_start_index": int(start_index),
        "history_end_index": int(row_index),
        "serialized_start_index": int(serialized_start_index),
        "serialized_end_index": int(serialized_end_index),
        "history_start_date": history_start_date,
        "history_end_date": history_end_date,
        "origin_date": origin_date,
        "value_columns": selected_cols,
        "variable_meanings": {col: variable_meanings[col] for col in selected_cols},
        "domain_context": context_text,
    }
    return prompt, meta


def call_openai_compatible(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> str:
    url = api_base.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}")
            if exc.code == 429 and attempt < retries:
                retry_after = exc.headers.get("Retry-After")
                try:
                    sleep_seconds = float(retry_after) if retry_after else retry_sleep * (attempt + 1) * 6
                except ValueError:
                    sleep_seconds = retry_sleep * (attempt + 1) * 6
                sleep_seconds = max(sleep_seconds, 30.0)
                print(
                    f"LLM request hit HTTP 429 on attempt {attempt + 1}/{retries + 1}; "
                    f"slowing down for {sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)
                continue
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = RuntimeError(f"LLM request failed: {exc}")
        if attempt < retries:
            sleep_seconds = retry_sleep * (attempt + 1)
            print(f"LLM request failed on attempt {attempt + 1}/{retries + 1}; retrying in {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)
    else:
        raise last_error or RuntimeError("LLM request failed")

    choice = data["choices"][0]
    if "message" in choice and "content" in choice["message"]:
        return str(choice["message"]["content"]).strip()
    if "text" in choice:
        return str(choice["text"]).strip()
    raise RuntimeError(f"Unsupported chat completion response shape: {data}")


def mock_response(prompt_meta: dict[str, Any]) -> str:
    return json.dumps(
        {
            "fact": (
                "The supplied historical sequence ends at {origin_date}; this mock fact field only confirms "
                "that the observed rows in the prompt are available."
            ).format(origin_date=prompt_meta["origin_date"]),
            "pred": (
                "Based only on the prompted historical sequence and variable meanings, this mock pred field would discuss "
                "trend persistence, volatility risk, and uncertainty without using future values."
            ),
        },
        ensure_ascii=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ECNU LLM-generated self-text CSV for fighting experiments.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", default="", help="Output CSV path. Defaults to <input>_ecnu_llm.csv.")
    parser.add_argument("--audit-output", default="", help="Prompt/response JSONL path.")
    parser.add_argument("--domain", default="", help="Human-readable domain name. Defaults to input parent folder.")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target-col", default="OT")
    parser.add_argument("--value-cols", default="OT", help="Comma-separated numeric columns or all_numeric.")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--fact-column", default=DEFAULT_FACT_COLUMN)
    parser.add_argument("--pred-column", default=DEFAULT_PRED_COLUMN)
    parser.add_argument(
        "--variable-meanings",
        default="",
        help='JSON object or path to JSON file mapping column names to meanings, e.g. {"OT":"US unemployment rate"}.',
    )
    parser.add_argument("--domain-context", default="", help="Optional domain context shown to the LLM.")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--min-history", type=int, default=0, help="Minimum history length. Defaults to seq_len.")
    parser.add_argument("--max-columns", type=int, default=4)
    parser.add_argument(
        "--max-history-rows",
        type=int,
        default=0,
        help="Maximum history rows serialized in the prompt. 0 means serialize the full historical window.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0, help="Exclusive end row index. 0 means end of file.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="Do not call ECNU; write deterministic mock text.")
    parser.add_argument("--resume", action="store_true", help="Reuse responses from an existing audit JSONL.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--api-base", default=getenv_first(["ECNU_LLM_BASE_URL", "OPENAI_BASE_URL"]))
    parser.add_argument("--api-key-env", default="ECNU_LLM_API_KEY")
    parser.add_argument("--model", default=getenv_first(["ECNU_LLM_MODEL", "OPENAI_MODEL"], "ecnu-llm"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25, help="Incrementally save CSV/audit after this many new rows. 0 disables incremental saves.")
    args = parser.parse_args()

    cwd = Path.cwd()
    input_path = resolve_path(args.input, cwd)
    output_path = (
        resolve_path(args.output, cwd)
        if args.output
        else input_path.with_name(input_path.stem + "_ecnu_llm.csv")
    )
    audit_path = (
        resolve_path(args.audit_output, cwd)
        if args.audit_output
        else Path(__file__).resolve().parents[1]
        / "results"
        / "ecnu_llm_text"
        / f"{input_path.stem}_H{args.seq_len}_F{args.pred_len}_{args.text_column}_audit.jsonl"
    )
    if output_path.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite or --resume.")

    if not args.mock and not args.api_base:
        raise RuntimeError("Set ECNU_LLM_BASE_URL or pass --api-base. Use --mock for local smoke tests.")
    api_key = getenv_first([args.api_key_env, "OPENAI_API_KEY"])
    if not args.mock and not api_key:
        raise RuntimeError(f"Set {args.api_key_env} or OPENAI_API_KEY. Use --mock for local smoke tests.")

    df = pd.read_csv(input_path)
    for col in [args.date_col, args.target_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    value_cols = parse_value_columns(df, args.value_cols, args.target_col)
    variable_meanings = load_variable_meanings(args.variable_meanings, value_cols)
    domain = args.domain or input_path.parent.name
    min_history = args.min_history or args.seq_len
    end_index = args.end_index or len(df)
    end_index = min(end_index, len(df))

    existing = (
        load_existing_audit(audit_path, args.text_column, args.fact_column, args.pred_column)
        if args.resume
        else {}
    )
    generated_rows: list[dict[str, Any]] = []
    text_values = df[args.text_column].astype(str).tolist() if args.text_column in df.columns else [""] * len(df)
    fact_values = df[args.fact_column].astype(str).tolist() if args.fact_column in df.columns else [""] * len(df)
    pred_values = df[args.pred_column].astype(str).tolist() if args.pred_column in df.columns else [""] * len(df)
    completed = 0

    for row_index in range(max(0, args.start_index), end_index):
        history_len = row_index - max(0, row_index - args.seq_len + 1) + 1
        if history_len < min_history:
            fact_values[row_index] = "Insufficient historical context before this timestamp."
            pred_values[row_index] = "No prediction-oriented text is generated because the historical context is incomplete."
            text_values[row_index] = combine_fact_pred(fact_values[row_index], pred_values[row_index])
            continue
        if row_index in existing:
            fact_values[row_index] = existing[row_index]["fact"]
            pred_values[row_index] = existing[row_index]["pred"]
            text_values[row_index] = existing[row_index]["combined_text"]
            continue

        prompt, meta = build_prompt(
            df=df,
            row_index=row_index,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            date_col=args.date_col,
            value_cols=value_cols,
            target_col=args.target_col,
            domain=domain,
            max_columns=args.max_columns,
            variable_meanings=variable_meanings,
            domain_context=args.domain_context,
            max_history_rows=args.max_history_rows,
        )
        if args.mock:
            response = mock_response(meta)
        else:
            response = call_openai_compatible(
                api_base=args.api_base,
                api_key=api_key,
                model=args.model,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            if args.sleep > 0:
                time.sleep(args.sleep)
        parsed_response = parse_fact_pred_response(response)
        fact = parsed_response["fact"]
        pred = parsed_response["pred"]
        combined_text = combine_fact_pred(fact, pred)
        fact_values[row_index] = fact
        pred_values[row_index] = pred
        text_values[row_index] = combined_text
        generated_rows.append(
            {
                "row_index": int(row_index),
                "domain": domain,
                "source_csv": str(input_path),
                "output_csv": str(output_path),
                "model": args.model if not args.mock else "mock",
                "text_column": args.text_column,
                "fact_column": args.fact_column,
                "pred_column": args.pred_column,
                "seq_len": int(args.seq_len),
                "pred_len": int(args.pred_len),
                "temperature": float(args.temperature),
                "max_tokens": int(args.max_tokens),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt": prompt,
                "response": response,
                "fact": fact,
                "pred": pred,
                "combined_text": combined_text,
                "leakage_check": {
                    "uses_y_values": False,
                    "max_prompt_row_index": int(meta["history_end_index"]),
                    "origin_row_index": int(row_index),
                    "max_prompt_time": meta["history_end_date"],
                    "origin_time": meta["origin_date"],
                    "uses_external_or_future_information": False,
                    "rule": (
                        "max_prompt_row_index <= origin_row_index; prompt contains only serialized historical rows "
                        "and user-supplied variable meanings/domain context"
                    ),
                },
                **meta,
            }
        )
        completed += 1
        if args.save_every and completed % args.save_every == 0:
            save_outputs(
                df=df,
                output_path=output_path,
                audit_path=audit_path,
                text_column=args.text_column,
                fact_column=args.fact_column,
                pred_column=args.pred_column,
                text_values=text_values,
                fact_values=fact_values,
                pred_values=pred_values,
                generated_rows=generated_rows,
                merge_existing_audit=bool(args.resume),
            )
            print(f"Saved incremental progress: {completed} new rows -> {output_path}")
        if args.limit and completed >= args.limit:
            break

    save_outputs(
        df=df,
        output_path=output_path,
        audit_path=audit_path,
        text_column=args.text_column,
        fact_column=args.fact_column,
        pred_column=args.pred_column,
        text_values=text_values,
        fact_values=fact_values,
        pred_values=pred_values,
        generated_rows=generated_rows,
        merge_existing_audit=bool(args.resume),
    )
    print(f"Wrote CSV: {output_path}")
    print(f"Wrote audit JSONL: {audit_path}")
    print(f"Generated or reused text for {sum(bool(str(x).strip()) for x in text_values)} / {len(text_values)} rows")


if __name__ == "__main__":
    main()
