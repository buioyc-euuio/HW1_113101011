# `train_pipeline.py` 與 `train_pipline2.py` 差異說明

> **檔名備註**：Repository 內原版腳本為 **`train_pipeline.py`**（`pipeline` 拼字）。`train_pipline2.py` 為第二版訓練腳本（檔名依專案慣例保留 `pipline` 拼法）。

## 總覽對照

| 項目 | `train_pipeline.py` | `train_pipline2.py` |
|------|---------------------|----------------------|
| Prompt 來源 | `split_dataset.format_prompt`（A–D 選項、`Answer:` + 數字 0–3） | 檔內 `format_prompt_cot`（0–3 選項、`### Instruction` / `### Response` + CoT 兩行金標） |
| 訓練標籤 | 接在 `Answer:` 後一個 token（數字） | `ans statement: …` 與 `ans is i) 選項全文` |
| Loss 範圍 | 幾乎整段序列（僅 padding 為 -100） | **僅 `### Response\n` 之後**（前綴 mask 為 -100） |
| 截斷策略 | 預設由 tokenizer（通常截右側） | **`truncation_side="left"`**，避免長題目切掉尾部 Response／答案 |
| 模型／LoRA 載入 | `load_model_from_hf.setup_model_and_lora()` | 檔內 `setup_model_and_lora_v2()`（見下表） |
| Trainer | **TRL `SFTTrainer`** | **`transformers.Trainer`** |
| 雙卡 DDP | 未內建 | 支援 `torchrun` + `WORLD_SIZE`；可 `TRAIN_USE_DDP=0` 關閉 |
| 預設輸出目錄 | `/kaggle/working/saved_models/lora_finetuned` | `/kaggle/working/saved_models/lora_finetuned_v2` |
| 資料路徑覆寫 | 寫死在函式或 `__main__` | 環境變數 **`KAGGLE_INPUT_DATASET`** 可覆寫 |

## 1. Prompt 與資料格式

### `train_pipeline.py`

- 使用 **`split_dataset.py`** 的模板：英文 medical expert 前綴、`Question`、**`A) B) C) D)`**、結尾 **`Answer:`**。
- 訓練時：`Answer:` 後接 **`ans` 欄位**（CSV 為 **0–3** 整數，與「A–D 字母」在文字上不一致，但訓練目標是續寫該數字）。

### `train_pipline2.py`

- **`format_prompt_cot`**：
  - **`### Instruction`** 區：`Question:` + **`choice:`** 下列 **0) 1) 2) 3)**。
  - **`### Response`**：推論時停在此處；訓練時再接兩行，例如：
    - `ans statement: {題幹首行去 ?} {正確選項},`
    - `ans is {i}) {正確選項全文}`

## 2. Dataset 與 Loss（為何 v2 較對準「答題」）

| | `train_pipeline.py` | `train_pipline2.py` |
|--|---------------------|----------------------|
| 建資料函式 | `_build_causal_dataset`：labels ≈ input_ids | `_build_causal_dataset_completion_only`：在 token 層找 **`### Response\n`**，**此前** labels 全 -100 |
| 效果 | 題幹、選項也參與語言建模 loss | 梯度主要壓在 **Response 內（敘述 + ans is）** |

## 3. LoRA 與載入方式

### `load_model_from_hf.py`（`train_pipeline` 使用）

- `LoraConfig`：`r=16`, `lora_alpha=32`
- `target_modules`：`q_proj`, `k_proj`, `v_proj`, `o_proj`
- `device_map="auto"`（單進程多卡時可能跨卡切模型）

### `setup_model_and_lora_v2`（`train_pipline2`）

- `r=32`, `lora_alpha=64`
- `target_modules`：attention 四個 + **`gate_proj`, `up_proj`, `down_proj`**
- **T4**：偵測算力 &lt; sm80 時用 **fp16**；較新 GPU 可用 bf16
- **DDP**：每進程 `device_map={"": local_rank}`，避免與 DDP 衝突的跨卡 auto 切分

## 4. 訓練超參數（預設值）

| | `train_pipeline.train_model` | `train_pipline2.train_model_v2` |
|--|------------------------------|----------------------------------|
| Epochs | 3 | 5 |
| per_device train batch | 2 | 4 |
| gradient_accumulation | 無 | **2** |
| learning_rate | 2e-5 | **1e-4** |
| max_seq_length | 512 | **768** |
| warmup / weight_decay | 無 | **warmup_ratio=0.06**, **weight_decay=0.01** |
| gradient_checkpointing | 無 | **True** |
| 挑最佳 checkpoint | `metric_for_best_model='loss'` | **`eval_loss`**（並設 `greater_is_better=False`） |

## 5. 評估（Test accuracy）

### `train_pipeline.evaluate_accuracy`

- 傳入已載入的 `model`, `tokenizer`（`__main__` 用 **`AutoModelForCausalLM.from_pretrained(output_dir)`** 載入 adapter 目錄，與典型 **Peft 輸出**不完全一致時可能載入異常）。
- Prompt：`format_prompt`（A–D / `Answer:`）。
- 預測：正則取 **第一個 A–D**。
- 標籤：`_normalize_answer(row["ans"]) `→ 對數字 `0–3` 會變成字串 **`"0".."3"`**，與預測的 **`"A".."D"`** **無法對齊**，離線 accuracy **可能嚴重失真**（除非另行改欄位或評估脚本）。

### `train_pipline2.evaluate_accuracy_v2`

- 自 **`adapter_dir`** 載入 **Tokenizer**，**Base 模型 + `PeftModel.from_pretrained`**。
- Prompt：`format_prompt_cot(..., is_test=True)`。
- 生成：**left truncate**、`max_new_tokens=128`（給 CoT 用）。
- 預測：**優先** `ans is K)`，再 `\b[0-3]\b`，再 A–D。
- 標籤：**`int(row["ans"])`**，與資料 CSV 一致。

評估迴圈有 **`tqdm`** 進度條。

## 6. Benchmark／提交 CSV

| | `train_pipeline.generate_predictions` | `train_pipline2` |
|--|--------------------------------------|------------------|
| 函式 | `generate_predictions` | **`generate_benchmark_predictions_cot`** |
| 輸出欄位 | `question_id`, `ans`（**A–D 字母**） | **`question_id`, `cot`, `ans`（0–3 整數）** |
| Prompt | 舊 A–D | `format_prompt_cot` |

若作業只收 **`question_id` + `pred`**，需從 v2 的 **`ans`** 欄另存，或自行改名；且推論必須與 **v2 訓練同一套 prompt**。

## 7. `__main__` 與路徑

| | `train_pipeline.py` | `train_pipline2.py` |
|--|---------------------|----------------------|
| 預設 input | `kaggle_input_dir = "/kaggle/input/aihw1_dataset_splitted"`（與上方常數路徑 **不一致**，使用時請自行對齊） | **`KAGGLE_INPUT_DIR`**：`/kaggle/input/datasets/nycu113101011/aihw1-dataset-splitted`，可環境變數覆寫 |
| 訓練後 | `plot_training_history` + `evaluate_accuracy`（載入方式見上） | `plot_training_history` + **`evaluate_accuracy_v2`**（僅 rank 0） |

## 8. 依賴

- **`train_pipeline.py`**：依賴 **TRL**（`SFTTrainer`）。
- **`train_pipline2.py`**：僅需 **`transformers.Trainer`**（不需 TRL）。

---

**實務建議**：若要以 v2 模型交作業或使用 `run_benchmark_inference.py`，請確認該腳本是否改為 **`format_prompt_cot`**（或呼叫 **`generate_benchmark_predictions_cot`**）；否則 prompt 與訓練不一致，分數會掉。
