import os
import re
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

try:
    from trl import SFTTrainer
except ImportError as e:
    raise ImportError(
        "trl not installed. Install it with `pip install trl` and retry."
    ) from e

from split_dataset import format_prompt
from load_model_from_hf import setup_model_and_lora


def _build_causal_dataset(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    # Causal LM label = input_ids
    labels = encodings.input_ids.clone()

    # 將 padding 部分標記為 -100，避免對 loss 造成影響
    attention_mask = encodings.attention_mask
    labels[attention_mask == 0] = -100

    dataset = torch.utils.data.TensorDataset(
        encodings.input_ids,
        encodings.attention_mask,
        labels,
    )
    return dataset


def train_model(
    train_csv=None,
    val_csv=None,
    test_csv=None,
    output_dir=None,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    max_seq_length=512,
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
):
    """訓練函式：建立 TrainingArguments、SFTTrainer，並訓練模型。

    主要依作業要求：設定學習率、Batch Size、Epoch 數，並儲存至 saved_models/。
    """

    # Kaggle paths
    kaggle_input_dir = "/kaggle/input/aihw1_dataset_splitted"
    kaggle_output_dir = "/kaggle/working/saved_models/lora_finetuned"

    train_csv = train_csv or os.path.join(kaggle_input_dir, "train.csv")
    val_csv = val_csv or os.path.join(kaggle_input_dir, "val.csv")
    test_csv = test_csv or os.path.join(kaggle_input_dir, "test.csv")
    output_dir = output_dir or kaggle_output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. 載入 LoRA 已設定模型與 tokenizer
    model, tokenizer = setup_model_and_lora()

    # 2. 載入資料並轉換為訓練文本
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_texts = [format_prompt(r, is_test=False) for _, r in train_df.iterrows()]
    val_texts = [format_prompt(r, is_test=False) for _, r in val_df.iterrows()]

    # 3. 建立 Dataset
    train_dataset = _build_causal_dataset(train_texts, tokenizer, max_length=max_seq_length)
    eval_dataset = _build_causal_dataset(val_texts, tokenizer, max_length=max_seq_length)

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy,
        save_total_limit=save_total_limit,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    # 5. 建立 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        tokenizer=tokenizer,
    )

    # 6. 開始訓練
    print("Starting training...")
    trainer.train()

    # 7. 儲存模型與 tokenizer
    print(f"Saving fine-tuned model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete.")

    return trainer


def plot_training_history(trainer, output_path="saved_models/training_history.png"):
    """從 Trainer logging history 畫出 Training/Validation loss。"""
    history = trainer.state.log_history

    train_epochs = []
    train_losses = []
    eval_epochs = []
    eval_losses = []

    for entry in history:
        if "loss" in entry and "epoch" in entry and "step" in entry and entry.get("eval_loss") is None:
            train_epochs.append(entry["epoch"])
            train_losses.append(entry["loss"])

        if "eval_loss" in entry and "epoch" in entry:
            eval_epochs.append(entry["epoch"])
            eval_losses.append(entry["eval_loss"])

    if len(train_epochs) == 0 and len(eval_epochs) == 0:
        print("無法從 trainer state 取得 loss 資訊，請確認是否開啟 logging/eval。")
        return

    plt.figure(figsize=(8, 5))
    if len(train_losses) > 0:
        plt.plot(train_epochs, train_losses, label="Train Loss", marker="o")
    if len(eval_losses) > 0:
        plt.plot(eval_epochs, eval_losses, label="Validation Loss", marker="o")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved training history plot to {output_path}")


def _normalize_answer(ans):
    return ans.strip().upper()


def _extract_predicted_option(text):
    # 抓取首個 A/B/C/D
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None


def evaluate_accuracy(
    model,
    tokenizer,
    test_csv="dataset/test.csv",
    max_seq_length=512,
    max_new_tokens=32,
    output_dir=None,
):
    """在測試集上做 generation，計算回答選項的準確率。"""
    test_df = pd.read_csv(test_csv)

    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(model_device)
    model.eval()

    correct = 0
    total = 0

    for _, row in test_df.iterrows():
        prompt = format_prompt(row, is_test=True)
        inputs = tokenizer(prompt, truncation=True, max_length=max_seq_length, return_tensors="pt").to(model_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        predicted = _extract_predicted_option(decoded)
        expected = _normalize_answer(row["ans"])

        if predicted is not None and predicted == expected:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")

    if output_dir:
        with open(os.path.join(output_dir, "test_accuracy.txt"), "w", encoding="utf-8") as f:
            f.write(f"accuracy={accuracy:.6f}\ncorrect={correct}\ntotal={total}\n")

    return accuracy


if __name__ == "__main__":
    # Kaggle 命名空間路徑
    kaggle_input_dir = "/kaggle/input/aihw1_dataset_splitted"
    kaggle_output_dir = "/kaggle/working/saved_models/lora_finetuned"

    trained = train_model(
        train_csv=os.path.join(kaggle_input_dir, "train.csv"),
        val_csv=os.path.join(kaggle_input_dir, "val.csv"),
        output_dir=kaggle_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        max_seq_length=512,
    )

    plot_training_history(trained, output_path=os.path.join(kaggle_output_dir, "training_history.png"))

    # 重新載入最佳模型以進行測試
    best_model = AutoModelForCausalLM.from_pretrained(kaggle_output_dir)
    best_tokenizer = AutoTokenizer.from_pretrained(kaggle_output_dir)

    evaluate_accuracy(
        best_model,
        best_tokenizer,
        test_csv=os.path.join(kaggle_input_dir, "test.csv"),
        max_seq_length=512,
        max_new_tokens=32,
        output_dir=kaggle_output_dir,
    )
