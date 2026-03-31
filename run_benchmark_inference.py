"""
使用訓練好的 LoRA 權重，對 benchmark.csv 逐題生成預測並輸出提交用 CSV（欄位：question_id, pred；pred 為 0–3）。

使用前請修改下方「路徑與模型設定」區塊的變數。
需已登入 Hugging Face（Llama 3 需授權）或本機已快取該 base 模型。

記憶體／解碼行為（環境變數，可選）：
  INFERENCE_BATCH_SIZE  一次送進 model.generate 的題數（整數，>=1）。GPU 未設時預設 16（大幅加速）；CPU 預設 1。
  INFERENCE_NUM_BEAMS   覆寫 beam 數量（整數，>=1）。未設時：GPU 預設 4，CPU 預設 1。**設為 1 通常可再快約 3–4 倍**（略影響品質）。
  INFERENCE_MAX_NEW_TOKENS  覆寫 max_new_tokens（整數）。未設時預設 24（答案只需少量 token；過大會拖慢 beam）。
  INFERENCE_LOW_MEMORY  設為 1 / true / yes 時強制 num_beams=1、batch=1，且於 CPU 每 100 題 gc 一次。

Kaggle Notebook：偵測到 Kaggle 時預設
  ADAPTER_DIR=/kaggle/input/models/nycu113101011/ai-hw1-model1/transformers/default/1/kaggle/working/saved_models/lora_finetuned
    （Kaggle Models 掛載實際路徑；若你發布新版本請改數字 1 或整段設 ADAPTER_DIR）
  若 LoRA 在 working，請設 ADAPTER_DIR=/kaggle/working/saved_models/lora_finetuned
  BENCHMARK_CSV=<KAGGLE_HW_DATASET_ROOT>/benchmark.csv；OUTPUT_CSV=/kaggle/working/benmark_prediction2.csv
  並嘗試從 Secrets 讀取 HF_TOKEN。覆寫：ADAPTER_DIR、KAGGLE_HW_DATASET_ROOT、BENCHMARK_CSV、OUTPUT_CSV
"""

from __future__ import annotations

import gc
import os
import re

import pandas as pd
import torch
from tqdm.auto import tqdm
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _is_kaggle_kernel() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle")


def _bootstrap_hf_token_from_kaggle_secrets() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"):
        return
    if not _is_kaggle_kernel():
        return
    try:
        from kaggle_secrets import UserSecretsClient

        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t:
            os.environ["HF_TOKEN"] = t
    except Exception:
        pass


_bootstrap_hf_token_from_kaggle_secrets()

# ── 路徑與模型設定（依你的環境修改）────────────────────────────────────────

# 與訓練時相同的 Hugging Face base model（需與 adapter 訓練時一致）
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

if _is_kaggle_kernel():
    # Kaggle Models 掛載路徑（含 transformers/default/<version>/…）；以環境變數 ADAPTER_DIR 可整段覆寫。
    _KAGGLE_DEFAULT_ADAPTER = (
        "/kaggle/input/models/nycu113101011/ai-hw1-model1/transformers/default/1/"
        "kaggle/working/saved_models/lora_finetuned"
    )
    _KAGGLE_HW_DATA = os.environ.get(
        "KAGGLE_HW_DATASET_ROOT",
        "/kaggle/input/datasets/nycu113101011/aihw1-dataset-splitted",
    )
    ADAPTER_DIR = os.environ.get("ADAPTER_DIR", _KAGGLE_DEFAULT_ADAPTER)
    BENCHMARK_CSV = os.environ.get(
        "BENCHMARK_CSV",
        os.path.join(_KAGGLE_HW_DATA, "benchmark.csv"),
    )
    OUTPUT_CSV = os.environ.get(
        "OUTPUT_CSV",
        "/kaggle/working/benmark_prediction2.csv",
    )
else:
    # 本機
    ADAPTER_DIR = r"C:\Users\weini\Downloads\lora_finetuned\kaggle\working\saved_models\lora_finetuned"
    BENCHMARK_CSV = os.path.join("dataset", "benchmark.csv")
    OUTPUT_CSV = "benmark_prediction2.csv"

# Hugging Face token：環境變數；Kaggle 已嘗試從 Secrets 帶入 HF_TOKEN
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# 生成參數（GPU 預設與 main.generate_predictions 對齊；CPU 預設見 _resolved_num_beams）
MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 24
NUM_BEAMS = 4
DEFAULT_GPU_BATCH_SIZE = 16

_GC_EVERY_N_STEPS_CPU = 100


def _log(msg: str) -> None:
    """立即輸出到終端（避免緩衝導致看起來當住）。"""
    print(msg, flush=True)


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _resolved_num_beams() -> int:
    """CPU 預設 beam=1 降低峰值記憶體；GPU 預設 NUM_BEAMS。皆可被環境變數覆寫。"""
    raw = os.environ.get("INFERENCE_NUM_BEAMS")
    if raw is not None and str(raw).strip() != "":
        return max(1, int(raw))
    if _env_truthy("INFERENCE_LOW_MEMORY"):
        return 1
    if torch.cuda.is_available():
        return NUM_BEAMS
    return 1


def _resolved_max_new_tokens() -> int:
    raw = os.environ.get("INFERENCE_MAX_NEW_TOKENS")
    if raw is not None and str(raw).strip() != "":
        return max(1, int(raw))
    return MAX_NEW_TOKENS


def _resolved_batch_size() -> int:
    raw = os.environ.get("INFERENCE_BATCH_SIZE")
    if raw is not None and str(raw).strip() != "":
        return max(1, int(raw))
    if _env_truthy("INFERENCE_LOW_MEMORY"):
        return 1
    if torch.cuda.is_available():
        return DEFAULT_GPU_BATCH_SIZE
    return 1


def _format_prompt(row: pd.Series, is_test: bool = True) -> str:
    question = row["question"]
    opa = row["opa"]
    opb = row["opb"]
    opc = row["opc"]
    opd = row["opd"]
    prompt = (
        f"You are a helpful medical expert. Please answer the following pathology "
        f"multiple-choice question.\n\nQuestion: {question}\n"
        f"A) {opa}\nB) {opb}\nC) {opc}\nD) {opd}\n\n"
        f"Please select the correct option (A, B, C, or D) and provide the answer.\nAnswer:"
    )
    if not is_test and "ans" in row and pd.notna(row.get("ans")):
        prompt += f' {row["ans"]}'
    return prompt


def _extract_predicted_index(text: str) -> int | None:
    """與 dataset.csv 一致：標籤為 0–3；若模型輸出 A–D 則轉成 0–3。"""
    t = text.strip()
    digit = re.search(r"\b([0-3])\b", t)
    if digit:
        return int(digit.group(1))
    letter = re.search(r"\b([A-D])\b", t.upper())
    if letter:
        return ord(letter.group(1)) - ord("A")
    return None


def load_model_for_inference(
    base_model_name: str,
    adapter_dir: str,
    hf_token: str | None,
):
    _log("[載入 1/5] Hugging Face login（若有設定 HF_TOKEN）…")
    if hf_token:
        login(token=hf_token)

    _log(f"[載入 2/5] 從 LoRA 目錄載入 tokenizer：{adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 勿使用 device_map="auto" 在本機 VRAM 吃緊時：會觸發權重 offload 到磁碟 / meta device，
    # PeftModel 載入 adapter 時對應不到模組路徑（KeyError: ...embed_tokens）。
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    _log(
        f"[載入 3/5] 載入底模 {base_model_name}（若本機無快取會下載，可能停很久；CPU 載入也較慢）…"
    )
    load_kw = dict(
        token=hf_token,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kw)
    except TypeError:
        load_kw.pop("dtype", None)
        load_kw["torch_dtype"] = dtype
        base = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kw)

    _log(f"[載入 4/5] 將底模搬到 {device}，並掛載 LoRA…")
    base = base.to(device)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.to(device)
    model.eval()
    _log("[載入 5/5] 模型就緒。")
    return model, tokenizer


def run_benchmark_inference(
    base_model_name: str = BASE_MODEL_NAME,
    adapter_dir: str = ADAPTER_DIR,
    benchmark_csv: str = BENCHMARK_CSV,
    output_csv: str = OUTPUT_CSV,
    hf_token: str | None = HF_TOKEN,
    max_seq_length: int = MAX_SEQ_LENGTH,
    max_new_tokens: int | None = None,
    num_beams: int | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame:
    _log("========== Benchmark 推論：開始 ==========")
    _log("[階段 A] 檢查檔案是否存在…")
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"找不到 LoRA 目錄: {adapter_dir}")
    if not os.path.isfile(benchmark_csv):
        raise FileNotFoundError(f"找不到 benchmark 檔: {benchmark_csv}")
    _log(f"  LoRA：{adapter_dir}")
    _log(f"  題目：{benchmark_csv}")
    _log(f"  輸出：{output_csv}")

    beams = num_beams if num_beams is not None else _resolved_num_beams()
    new_tokens_limit = max_new_tokens if max_new_tokens is not None else _resolved_max_new_tokens()
    bs = batch_size if batch_size is not None else _resolved_batch_size()
    use_low_mem_gc = _env_truthy("INFERENCE_LOW_MEMORY") or (
        not torch.cuda.is_available() and beams == 1
    )

    _log(
        f"[階段 B] 解碼設定：batch_size={bs} | num_beams={beams} | "
        f"max_new_tokens={new_tokens_limit} | max_seq_length={max_seq_length}"
    )
    cuda_on = torch.cuda.is_available()
    if not cuda_on:
        _log(
            "  （目前為 CPU：會很慢，進度條仍會動；若長時間停在「載入底模」代表在讀檔或下載。）"
        )

    _log("[階段 C] 載入模型（通常最久；請對照上方 [載入 x/5]）…")
    model, tokenizer = load_model_for_inference(base_model_name, adapter_dir, hf_token)
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    _log(
        f"[階段 C 完成] 推論裝置: {device} | batch_size={bs} | num_beams={beams} | max_new_tokens={new_tokens_limit}"
        + (
            "（CPU：beam=1；若要 4 請設 INFERENCE_NUM_BEAMS=4）"
            if beams == 1 and not cuda_on
            else ""
        )
    )

    _log(f"[階段 D] 讀取 benchmark CSV 並展開列…")
    benchmark_df = pd.read_csv(benchmark_csv, dtype=str)
    results: list[dict] = []
    step = 0
    rows_list = [row for _, row in benchmark_df.iterrows()]
    n = len(rows_list)
    _log(f"[階段 D 完成] 共 {n} 題，即將開始 generate。")

    _log("[階段 E] 逐批推論（下方為進度條；首批可能較久為正常現象）…")
    first_batch = True
    with tqdm(
        total=n,
        desc="Benchmark inference",
        unit="q",
        mininterval=0.5,
        dynamic_ncols=True,
    ) as pbar:
        for start in range(0, n, bs):
            if first_batch:
                tqdm.write(
                    "[推論] 正在跑第一批 model.generate（CUDA 首次呼叫可能多等幾十秒）…"
                )
                first_batch = False
            batch_rows = rows_list[start : start + bs]
            prompts = [_format_prompt(r, is_test=True) for r in batch_rows]
            inputs = tokenizer(
                prompts,
                truncation=True,
                max_length=max_seq_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            in_w = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=new_tokens_limit,
                    do_sample=False,
                    num_beams=beams,
                    early_stopping=beams > 1,
                    pad_token_id=pad_id,
                )

            for j, row in enumerate(batch_rows):
                gen_ids = outputs[j, in_w:]
                generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                pred = _extract_predicted_index(generated_text)
                if pred is None:
                    pred = 0
                results.append({"question_id": row.get("question_id", None), "pred": int(pred)})

            del outputs
            step += len(batch_rows)
            pbar.update(len(batch_rows))
            if device.type == "cpu" and use_low_mem_gc and step % _GC_EVERY_N_STEPS_CPU == 0:
                gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    _log(f"[階段 F] 寫入 CSV：{output_csv}")
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_csv, index=False, encoding="utf-8")
    _log(f"========== 完成：共 {len(submission_df)} 筆 pred，已寫入 ==========")
    return submission_df


if __name__ == "__main__":
    run_benchmark_inference()
