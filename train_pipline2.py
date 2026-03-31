"""
train_pipeline.py 的加強版（檔名依需求：train_pipline2.py）。

為何先前準確率容易偏弱（本版對策）
────────────────────────────────────────
1) **Prompt／標籤格式**：改為 choice 用 0–3、模型在 ### Response 後輸出
   「ans statement: …,」與「ans is {i}) {option}」，與推論時格式一致；loss 僅算在 Response 段。
2) **LoRA**：較大 rank + attention 與 MLP 模組。
3) **解析**：優先從「ans is K)」抽類別，再 fallback 數字／A–D。
4) **三欄提交**：可用 generate_benchmark_predictions_cot() 輸出 question_id, cot, ans。

Kaggle 雙 T4 GPU
────────────────
- 若只跑 `python train_pipline2.py`，預設仍只用 **單卡**（較簡單）。
- 要用兩張 T4：請用 torchrun（Accelerator / 雙程序各綁一張卡），例如：
    torchrun --nproc_per_node=2 train_pipline2.py
  Notebook 可：`!cd /kaggle/working/HW1_113101011 && torchrun --nproc_per_node=2 train_pipline2.py`

環境變數（可選）
────────────────
  TRAIN_USE_DDP=0          強制關閉 DDP（即使使用 torchrun）。
  KAGGLE_INPUT_DATASET     train/val/test 所在資料目錄（預設見下方常數）。
"""

from __future__ import annotations

import inspect
import os
import re
import sys

# 與原 train_pipeline 相同：讓 Kaggle working 目錄可被 import
_repo = "/kaggle/working/HW1_113101011"
if os.path.isdir("/kaggle/working") and _repo not in sys.path:
    sys.path.insert(0, _repo)

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# ── Kaggle 路徑（與 train_pipeline 對齊，可多一個環境變數覆寫）──
_DEFAULT_INPUT = "/kaggle/input/datasets/nycu113101011/aihw1-dataset-splitted"
KAGGLE_INPUT_DIR = os.environ.get("KAGGLE_INPUT_DATASET", _DEFAULT_INPUT)
KAGGLE_OUTPUT_DIR = "/kaggle/working/saved_models/lora_finetuned_v2"

BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# 與訓練 loss 遮罩對齊：只可改字串若同步改 _marker_token_end_index 的搜尋字串
_INSTRUCTION = "### Instruction\n"
_RESPONSE = "### Response\n"


def format_prompt_cot(row: pd.Series, is_test: bool = True) -> str:
    """
    0–3 選項 + CoT 風格輸出。訓練時在 ### Response 後接上金標兩行。
    """
    q = str(row["question"]).strip()
    o0, o1, o2, o3 = (
        str(row["opa"]),
        str(row["opb"]),
        str(row["opc"]),
        str(row["opd"]),
    )
    user_block = (
        f"{_INSTRUCTION}"
        f"Question: {q}\n"
        f"choice:\n"
        f"0) {o0}\n"
        f"1) {o1}\n"
        f"2) {o2}\n"
        f"3) {o3}\n\n"
        f"{_RESPONSE}"
    )
    if is_test:
        return user_block
    ai = int(row["ans"])
    opts = [o0, o1, o2, o3]
    opt = opts[ai]
    stem_first = q.split("\n")[0].strip()
    stmt = f"{stem_first.rstrip('?')} {opt}"
    gold = f"ans statement: {stmt},\nans is {ai}) {opt}\n"
    return user_block + gold


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _use_ddp_flag() -> bool:
    """WORLD_SIZE>1（例如 torchrun --nproc_per_node=2）時預設走 DDP；可設 TRAIN_USE_DDP=0 關閉。"""
    ex = os.environ.get("TRAIN_USE_DDP", "").strip().lower()
    if ex in ("0", "false", "no", "off"):
        return False
    if _world_size() > 1:
        return True
    return ex in ("1", "true", "yes", "on")


def setup_model_and_lora_v2(
    model_name: str = BASE_MODEL_NAME,
    hf_token: str | None = None,
):
    """
    較強 LoRA：較大 rank、含 MLP 線性層；DDP 時每進程綁一張卡，不用 device_map=auto 跨卡切一份模型。
    """
    from huggingface_hub import login

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        login(token=tok)
    elif hf_token:
        login(token=hf_token)

    print(f"[v2] Loading tokenizer / base model: {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token or tok)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lr = _local_rank()
    use_ddp = _use_ddp_flag() and lr >= 0 and _world_size() > 1

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        # T4 (sm_75)：bf16 硬體支援弱，推論／訓練較穩用 fp16
        dtype = torch.float16

    if use_ddp:
        torch.cuda.set_device(lr)
        device_map = {"": lr}
        low_mem = False
    elif torch.cuda.is_available():
        device_map = "auto"
        low_mem = True
    else:
        device_map = None
        low_mem = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token or tok,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_mem,
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def _marker_token_end_index(
    input_ids: list[int],
    tokenizer,
    marker: str = _RESPONSE,
) -> int:
    """在 input_ids 中找到 marker（預設 ### Response\\n），取**最後一處**之後為訓練 loss 區。"""
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    if not marker_ids:
        return 0
    end = 0
    for start in range(len(input_ids) - len(marker_ids) + 1):
        if input_ids[start : start + len(marker_ids)] == marker_ids:
            end = start + len(marker_ids)
    return end


def _build_causal_dataset_completion_only(
    texts: list[str],
    tokenizer,
    max_length: int = 512,
    marker: str = _RESPONSE,
):
    """
    Causal LM，**僅對 ### Response 之後（CoT + ans is…）計算 loss**。
    """
    input_ids_l: list[list[int]] = []
    attn_l: list[list[int]] = []
    labels_l: list[list[int]] = []

    for text in texts:
        enc = tokenizer(
            text,
            truncation=True,
            truncation_side="left",
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=True,
        )
        ids = enc["input_ids"]
        att = enc["attention_mask"]
        labs = list(ids)

        split = _marker_token_end_index(ids, tokenizer, marker=marker)
        for j in range(len(labs)):
            if j < split or att[j] == 0:
                labs[j] = -100

        input_ids_l.append(ids)
        attn_l.append(att)
        labels_l.append(labs)

    return Dataset.from_dict(
        {
            "input_ids": input_ids_l,
            "attention_mask": attn_l,
            "labels": labels_l,
        }
    )


def _extract_predicted_index(text: str) -> int | None:
    """優先對齊訓練格式「ans is {0-3}) …」。"""
    t = text.strip()
    m = re.search(r"(?i)ans\s*is\s*([0-3])\s*\)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([0-3])\b", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([A-D])\b", t.upper())
    if m:
        return ord(m.group(1)) - ord("A")
    return None


def train_model_v2(
    train_csv: str | None = None,
    val_csv: str | None = None,
    output_dir: str | None = None,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-4,
    max_seq_length: int = 768,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    logging_steps: int = 20,
    save_total_limit: int = 2,
):
    train_csv = train_csv or os.path.join(KAGGLE_INPUT_DIR, "train.csv")
    val_csv = val_csv or os.path.join(KAGGLE_INPUT_DIR, "val.csv")
    output_dir = output_dir or KAGGLE_OUTPUT_DIR

    lr = _local_rank()
    is_main = lr in (-1, 0)

    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = setup_model_and_lora_v2()

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    train_texts = [format_prompt_cot(r, is_test=False) for _, r in train_df.iterrows()]
    val_texts = [format_prompt_cot(r, is_test=False) for _, r in val_df.iterrows()]

    train_dataset = _build_causal_dataset_completion_only(
        train_texts, tokenizer, max_length=max_seq_length
    )
    eval_dataset = _build_causal_dataset_completion_only(
        val_texts, tokenizer, max_length=max_seq_length
    )

    use_ddp = _use_ddp_flag() and lr >= 0 and _world_size() > 1

    fp16 = torch.cuda.is_available() and dtype_is_fp16(model)
    bf16 = torch.cuda.is_available() and not fp16 and torch.cuda.get_device_capability()[0] >= 8

    training_args_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "logging_steps": logging_steps,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "save_total_limit": save_total_limit,
        "fp16": fp16,
        "bf16": bf16,
        "optim": "adamw_torch",
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "gradient_checkpointing": True,
        "ddp_find_unused_parameters": False,
        "dataloader_pin_memory": torch.cuda.is_available(),
    }

    if use_ddp:
        training_args_kwargs["local_rank"] = lr

    valid_params = inspect.signature(TrainingArguments.__init__).parameters
    training_args_kwargs = {k: v for k, v in training_args_kwargs.items() if k in valid_params}
    if "evaluation_strategy" not in training_args_kwargs and "eval_strategy" in valid_params:
        training_args_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    model.enable_input_require_grads()

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    if is_main:
        print("[v2] Starting training …")
    trainer.train()

    if is_main:
        print(f"[v2] Saving adapter to {output_dir} …")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("[v2] Training complete.")
    return trainer


def dtype_is_fp16(model) -> bool:
    try:
        return model.get_input_embeddings().weight.dtype == torch.float16
    except Exception:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8


def plot_training_history(trainer, output_path: str | None = None):
    output_path = output_path or os.path.join(KAGGLE_OUTPUT_DIR, "training_history_v2.png")
    history = trainer.state.log_history
    train_epochs, train_losses = [], []
    eval_epochs, eval_losses = [], []

    for entry in history:
        if (
            "loss" in entry
            and "epoch" in entry
            and "step" in entry
            and entry.get("eval_loss") is None
        ):
            train_epochs.append(entry["epoch"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "epoch" in entry:
            eval_epochs.append(entry["epoch"])
            eval_losses.append(entry["eval_loss"])

    if not train_losses and not eval_losses:
        print("無法取得 loss 紀錄。")
        return

    plt.figure(figsize=(8, 5))
    if train_losses:
        plt.plot(train_epochs, train_losses, label="Train Loss", marker="o")
    if eval_losses:
        plt.plot(eval_epochs, eval_losses, label="Validation Loss", marker="o")
    plt.title("Training / Validation Loss (v2)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"已儲存 {output_path}")


def evaluate_accuracy_v2(
    adapter_dir: str,
    test_csv: str,
    max_seq_length: int = 768,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    output_dir: str | None = None,
):
    """與資料相同的 0–3 標籤比對；載入 base + LoRA adapter。"""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=tok)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        token=tok,
        torch_dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    test_df = pd.read_csv(test_csv)
    correct = 0
    total = 0

    for _, row in tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc="Eval test (v2)",
        unit="q",
    ):
        prompt = format_prompt_cot(row, is_test=True)
        inputs = tokenizer(
            prompt,
            truncation=True,
            truncation_side="left",
            max_length=max_seq_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                early_stopping=num_beams > 1,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        pred = _extract_predicted_index(gen)
        if pred is None:
            pred = 0
        expected = int(row["ans"])

        if pred == expected:
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    print(f"[v2] Test Accuracy: {acc:.4f} ({correct}/{total})")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        p = os.path.join(output_dir, "test_accuracy_v2.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"accuracy={acc:.6f}\ncorrect={correct}\ntotal={total}\n")
    return acc


def generate_benchmark_predictions_cot(
    adapter_dir: str,
    benchmark_csv: str,
    output_csv: str,
    base_model_name: str = BASE_MODEL_NAME,
    max_seq_length: int = 768,
    max_new_tokens: int = 128,
    num_beams: int = 4,
):
    """
    產生 question_id, cot, ans；ans 為 0–3 整數字串，cot 為模型完整生成（需脫敏可自行後處理）。
    """
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=tok)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=tok,
        torch_dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = base.to(device)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    benchmark_df = pd.read_csv(benchmark_csv, dtype=str)
    rows_out: list[dict] = []

    for _, row in tqdm(
        benchmark_df.iterrows(),
        total=len(benchmark_df),
        desc="Benchmark CoT predict",
        unit="q",
    ):
        prompt = format_prompt_cot(row, is_test=True)
        inputs = tokenizer(
            prompt,
            truncation=True,
            truncation_side="left",
            max_length=max_seq_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                early_stopping=num_beams > 1,
                pad_token_id=tokenizer.pad_token_id,
            )
        cot = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        pred = _extract_predicted_index(cot)
        if pred is None:
            pred = 0
        rows_out.append(
            {
                "question_id": row.get("question_id", None),
                "cot": cot.replace("\r\n", "\n").strip(),
                "ans": int(pred),
            }
        )

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[v2] 已寫入 {len(out_df)} 筆 → {output_csv}")
    return out_df


if __name__ == "__main__":
    trained = train_model_v2(
        train_csv=os.path.join(KAGGLE_INPUT_DIR, "train.csv"),
        val_csv=os.path.join(KAGGLE_INPUT_DIR, "val.csv"),
        output_dir=KAGGLE_OUTPUT_DIR,
    )
    # DDP 時僅 rank 0 畫圖與評估，避免重複
    if _local_rank() in (-1, 0):
        plot_training_history(
            trained,
            output_path=os.path.join(KAGGLE_OUTPUT_DIR, "training_history_v2.png"),
        )
        evaluate_accuracy_v2(
            adapter_dir=KAGGLE_OUTPUT_DIR,
            test_csv=os.path.join(KAGGLE_INPUT_DIR, "test.csv"),
            output_dir=KAGGLE_OUTPUT_DIR,
        )
