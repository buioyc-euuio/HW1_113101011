"""
使用訓練好的 LoRA 權重，對 benchmark.csv 逐題生成預測並輸出提交用 CSV。

使用前請修改下方「路徑與模型設定」區塊的變數。
需已登入 Hugging Face（Llama 3 需授權）或本機已快取該 base 模型。

記憶體／解碼行為（環境變數，可選）：
  INFERENCE_NUM_BEAMS   覆寫 beam 數量（整數，>=1）。未設時：GPU 預設 4，CPU 預設 1（省 RAM）。
  INFERENCE_LOW_MEMORY  設為 1 / true / yes 時強制 num_beams=1，且於 CPU 每 100 題 gc 一次。

Kaggle Notebook：偵測到 Kaggle 時預設
  ADAPTER_DIR=/kaggle/working/saved_models/lora_finetuned
  BENCHMARK_CSV=<KAGGLE_HW_DATASET_ROOT>/benchmark.csv
  OUTPUT_CSV=/kaggle/working/benchmark_predictions.csv
  並嘗試從 Secrets 讀取 HF_TOKEN。可用 KAGGLE_HW_DATASET_ROOT、ADAPTER_DIR、BENCHMARK_CSV、OUTPUT_CSV 覆寫。
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
    # Kaggle：與 main.py 相同預設；可全程用環境變數覆寫。
    _KAGGLE_HW_DATA = os.environ.get(
        "KAGGLE_HW_DATASET_ROOT",
        "/kaggle/input/datasets/nycu113101011/aihw1-dataset-splitted",
    )
    ADAPTER_DIR = os.environ.get(
        "ADAPTER_DIR",
        "/kaggle/working/saved_models/lora_finetuned",
    )
    BENCHMARK_CSV = os.environ.get(
        "BENCHMARK_CSV",
        os.path.join(_KAGGLE_HW_DATA, "benchmark.csv"),
    )
    OUTPUT_CSV = os.environ.get(
        "OUTPUT_CSV",
        "/kaggle/working/benchmark_predictions.csv",
    )
else:
    # 本機
    ADAPTER_DIR = r"C:\Users\weini\Downloads\lora_finetuned\kaggle\working\saved_models\lora_finetuned"
    BENCHMARK_CSV = os.path.join("dataset", "benchmark.csv")
    OUTPUT_CSV = "benchmark_predictions.csv"

# Hugging Face token：環境變數；Kaggle 已嘗試從 Secrets 帶入 HF_TOKEN
HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# 生成參數（GPU 預設與 main.generate_predictions 對齊；CPU 預設見 _resolved_num_beams）
MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 64
NUM_BEAMS = 4

_GC_EVERY_N_STEPS_CPU = 100


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


def _extract_predicted_option(text: str) -> str | None:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None


def load_model_for_inference(
    base_model_name: str,
    adapter_dir: str,
    hf_token: str | None,
):
    if hf_token:
        login(token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 勿使用 device_map="auto" 在本機 VRAM 吃緊時：會觸發權重 offload 到磁碟 / meta device，
    # PeftModel 載入 adapter 時對應不到模組路徑（KeyError: ...embed_tokens）。
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

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

    base = base.to(device)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def run_benchmark_inference(
    base_model_name: str = BASE_MODEL_NAME,
    adapter_dir: str = ADAPTER_DIR,
    benchmark_csv: str = BENCHMARK_CSV,
    output_csv: str = OUTPUT_CSV,
    hf_token: str | None = HF_TOKEN,
    max_seq_length: int = MAX_SEQ_LENGTH,
    max_new_tokens: int = MAX_NEW_TOKENS,
    num_beams: int | None = None,
) -> pd.DataFrame:
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"找不到 LoRA 目錄: {adapter_dir}")
    if not os.path.isfile(benchmark_csv):
        raise FileNotFoundError(f"找不到 benchmark 檔: {benchmark_csv}")

    beams = num_beams if num_beams is not None else _resolved_num_beams()
    use_low_mem_gc = _env_truthy("INFERENCE_LOW_MEMORY") or (
        not torch.cuda.is_available() and beams == 1
    )

    model, tokenizer = load_model_for_inference(base_model_name, adapter_dir, hf_token)
    device = next(model.parameters()).device
    print(
        f"推論裝置: {device} | num_beams={beams}"
        + ("（CPU 省記憶體預設為 1；若要 4 請設 INFERENCE_NUM_BEAMS=4）" if beams == 1 and not torch.cuda.is_available() else "")
    )

    benchmark_df = pd.read_csv(benchmark_csv, dtype=str)
    results = []
    step = 0

    for _, row in tqdm(
        benchmark_df.iterrows(),
        total=len(benchmark_df),
        desc="Benchmark inference",
        unit="q",
    ):
        prompt = _format_prompt(row, is_test=True)
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=beams,
                early_stopping=beams > 1,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        del outputs
        predicted = _extract_predicted_option(generated_text)
        if predicted is None:
            predicted = "A"

        qid = row.get("question_id", None)
        results.append({"question_id": qid, "ans": predicted})

        step += 1
        if device.type == "cpu" and use_low_mem_gc and step % _GC_EVERY_N_STEPS_CPU == 0:
            gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"共 {len(submission_df)} 筆預測，已寫入: {output_csv}")
    return submission_df


if __name__ == "__main__":
    run_benchmark_inference()
