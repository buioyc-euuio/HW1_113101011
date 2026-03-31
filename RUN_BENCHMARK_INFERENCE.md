# `run_benchmark_inference.py` 使用說明

本文件說明如何修改路徑、在**本機**執行推理，以及在 **Kaggle** 上放 `saved_models`（LoRA 輸出資料夾）該怎麼對應到程式里的 `ADAPTER_DIR`。

---

## 你需要準備什麼

- **Python 環境**：建議使用專案內 `requirements.txt`（含 `transformers`、`peft`、`torch` 等）。
- **Base 模型**：預設為 `meta-llama/Llama-3.2-1B-Instruct`，需與訓練 LoRA 時相同；首次使用通常要在 [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 同意條款，並準備 **Access Token**。
- **LoRA 資料夾**：內含至少 `adapter_config.json`、`adapter_model.safetensors`，以及訓練時一併儲存的 **tokenizer** 檔（例如 `tokenizer.json`、`tokenizer_config.json` 等）。程式會從這個資料夾讀 tokenizer，並把權重掛在 base 模型上。

`ADAPTER_DIR` 請指向「**含有上述檔案的那一層資料夾**」，名稱不一定要叫 `saved_models`，重點是路徑要對。

---

## 路徑要改哪裡？

請打開 `run_benchmark_inference.py`，修改檔案上方 **`# ── 路徑與模型設定`** 區塊：

| 變數 | 用途 |
|------|------|
| `BASE_MODEL_NAME` | Hugging Face 上的 base 模型 ID，需與訓練一致。 |
| `ADAPTER_DIR` | LoRA（adapter + tokenizer）所在目錄的**絕對或相對路徑**。 |
| `BENCHMARK_CSV` | `benchmark.csv` 路徑（欄位需含 `question_id`, `question`, `opa`, `opb`, `opc`, `opd`）。 |
| `OUTPUT_CSV` | 預測結果輸出檔（預設 `question_id`, `ans`）。 |

**Hugging Face Token**：程式會讀環境變數 `HF_TOKEN` 或 `HUGGINGFACE_TOKEN`。若已在本機 `huggingface-cli login` 且模型已快取，有時可不設；Llama 授權未過時請務必設定 token。

### Windows 本機路徑範例

若你把整包從 Kaggle 下載後放在「下載」資料夾，可改成原始字串路徑，避免反斜線轉義問題：

```python
ADAPTER_DIR = r"C:\Users\你的使用者名稱\Downloads\lora_finetuned\kaggle\working\saved_models\lora_finetuned"
BENCHMARK_CSV = r"C:\Users\你的使用者名稱\Desktop\HW1_113101011\dataset\benchmark.csv"
OUTPUT_CSV = r"C:\Users\你的使用者名稱\Desktop\HW1_113101011\benchmark_predictions.csv"
```

若 LoRA 與程式在**同一專案目錄**下，也可用相對路徑（與目前預設類似）：

```python
ADAPTER_DIR = os.path.join("saved_models", "lora_finetuned")
BENCHMARK_CSV = os.path.join("dataset", "benchmark.csv")
```

---

## 在本機怎麼跑？

1. **進入專案目錄**（含 `run_benchmark_inference.py` 的資料夾）。
2. **建立並啟用虛擬環境**（若尚未建立）：

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **安裝依賴**：

   ```bash
   pip install -r requirements.txt
   ```

4. **（建議）設定 HF Token**（PowerShell 範例）：

   ```powershell
   $env:HF_TOKEN = "hf_xxxxxxxx"
   ```

5. 依上一節修改 `ADAPTER_DIR`、`BENCHMARK_CSV` 等。
6. **執行**：

   ```bash
   python run_benchmark_inference.py
   ```

完成後檢查 `OUTPUT_CSV` 指定的檔案是否有約與 `benchmark.csv` 相同筆數的列。若有 **NVIDIA GPU**，請安裝對應的 CUDA 版 PyTorch 以加速；僅 CPU 亦可跑，但 900 題會較慢。

---

## 在 Kaggle 怎麼跑？`saved_models` 要放哪？

Kaggle 沒有「固定只能放某一個資料夾名稱」，重要的是 **`ADAPTER_DIR` 要等於 Notebook 執行時「實際看得到 adapter 的那個路徑」**。常見做法如下。

### Kaggle 快速步驟（`run_benchmark_inference.py` 已內建預設路徑）

在 Kaggle Notebook 里，只要偵測到在 Kaggle 上執行，腳本會**自動**使用（可用環境變數覆寫）：

| 項目 | 預設 |
|------|------|
| `ADAPTER_DIR` | `/kaggle/input/models/nycu113101011/ai-hw1-model1/transformers/default/1/kaggle/working/saved_models/lora_finetuned`（**Kaggle Models** 實際掛載路徑；版本號 `1` 若升版請改路徑或設 `ADAPTER_DIR`） |
| `BENCHMARK_CSV` | `/kaggle/input/datasets/nycu113101011/aihw1-dataset-splitted/benchmark.csv` |
| `OUTPUT_CSV` | `/kaggle/working/benchmark_predictions.csv` |
| `HF_TOKEN` | 若環境變數未設定，會嘗試自 Kaggle **Secrets** 讀取名為 `HF_TOKEN` 的 secret |

若 LoRA 是同一個 Notebook 訓練後直接存在 `working`，請設 `ADAPTER_DIR=/kaggle/working/saved_models/lora_finetuned`。若 Models 路徑不同，請從 Notebook 左側 Input 複製完整路徑設成 `ADAPTER_DIR`。

**你需要做的：**

1. Notebook 右側 **Accelerator** 選 **GPU**（建議）。
2. **Add Data** 掛上作業的 **Input Dataset**（路徑需與上表一致；若 Kaggle 上資料夾名稱不同，請在執行腳本前設定 `KAGGLE_HW_DATASET_ROOT`，見下）。
3. **Settings → Secrets**：新增 `HF_TOKEN`（Hugging Face Access Token，且帳號已接受 Llama 3.2 使用條款）。
4. 將本專案的 `run_benchmark_inference.py` 上傳到 Notebook（**Add Input** 或 **Upload** 到工作目錄），或使用 `!wget` / 手動新增檔案。
5. 第一個程式碼儲存格安裝依賴（Kaggle 預裝 torch，其餘補齊即可）：

   ```python
   !pip install -q "transformers>=4.34.0" "peft>=0.5.0" accelerate tqdm huggingface_hub pandas
   ```

6. 若作業資料集在 Input 里的實際路徑**不是** `.../datasets/nycu113101011/aihw1-dataset-splitted`，在**匯入腳本前**設定：

   ```python
   import os
   os.environ["KAGGLE_HW_DATASET_ROOT"] = "/kaggle/input/你在_Add_Data_看到的資料夾名稱"
   ```

7. 若 LoRA 是**另上傳的 Dataset**（不是 `working` 里訓練產物），在匯入腳本前設定：

   ```python
   import os
   os.environ["ADAPTER_DIR"] = "/kaggle/input/你的-lora-dataset/.../lora_finetuned"
   ```

8. 執行推理：

   ```python
   # 若腳本在當前工作目錄
   %run run_benchmark_inference.py
   ```

   或直接：

   ```python
   from run_benchmark_inference import run_benchmark_inference
   run_benchmark_inference()
   ```

完成後到 **Output** 或 `/kaggle/working/` 下載 `benchmark_predictions.csv`。Secrets 會在 `import run_benchmark_inference` 時嘗試注入；若你更喜歡手動，仍可在最頂格自行 `UserSecretsClient().get_secret("HF_TOKEN")` 設到 `os.environ["HF_TOKEN"]` 後再 import。

### 做法 A：同一個 Notebook 裡剛訓練完（路徑已在 working 下）

若訓練程式把模型存成（與你 `main.py` 預設類似）：

`/kaggle/working/saved_models/lora_finetuned`

則推理時可設：

```python
ADAPTER_DIR = "/kaggle/working/saved_models/lora_finetuned"
BENCHMARK_CSV = "/kaggle/input/你的資料集資料夾名稱/benchmark.csv"
OUTPUT_CSV = "/kaggle/working/benchmark_predictions.csv"
```

`benchmark.csv` 來自作業的 **Input Dataset** 時，路徑請到 Notebook 左側 **Input** 檔案樹展開，複製實際路徑（通常在 `/kaggle/input/...` 底下）。

### 做法 B：另開 Notebook，只跑推理（LoRA 已另存上傳）

1. 將本機的 **`lora_finetuned` 整包資料夾**（或整個 `saved_models` 壓縮後上傳，解壓後結構一致）做成 **New Dataset** 上傳到 Kaggle。
2. 在 Notebook **Add Data** 掛上該 Dataset。
3. 假設 Dataset 掛載後，adapter 所在路徑為：

   `/kaggle/input/你上傳的資料集名稱/lora_finetuned`

   則設定：

   ```python
   ADAPTER_DIR = "/kaggle/input/你上傳的資料集名稱/lora_finetuned"
   ```

若你上傳時 zip 內多一層目錄（例如解開後是 `saved_models/lora_finetuned`），`ADAPTER_DIR` 就必須寫到**含有 `adapter_config.json` 的那層**，例如：

```python
ADAPTER_DIR = "/kaggle/input/我的-lora-dataset/saved_models/lora_finetuned"
```

### Kaggle 上的 Hugging Face Token

在 Notebook 的 **Secrets** 新增例如 `HF_TOKEN`，在程式最前面讀取並設成環境變數後再 import 推理邏輯，或直接在 Notebook 儲存格：

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

然後再執行 `run_benchmark_inference.py` 里的 `run_benchmark_inference()`，或 `%run run_benchmark_inference.py`（需先確保工作目錄有該檔案）。

### Kaggle GPU

在 Notebook 右側 **Accelerator** 選 **GPU**，可明顯縮短 900 題推理時間。

---

## 簡短對照

| 環境 | `ADAPTER_DIR` 概念 |
|------|-------------------|
| 本機 | 下載的 LoRA 資料夾完整路徑，或專案內 `saved_models/lora_finetuned`。 |
| Kaggle 訓練同一 session | 多半是 `/kaggle/working/saved_models/lora_finetuned`。 |
| Kaggle 另掛 Dataset | `/kaggle/input/<資料集名稱>/...` 直到看到 `adapter_config.json` 的那層。 |

若執行時出現找不到 `adapter_config.json`，代表 `ADAPTER_DIR` 多或少一層目錄，請對照檔案總管或 Kaggle Input 樹狀結構調整即可。
