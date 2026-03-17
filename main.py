import os

import pandas as pd

from split_dataset import load_and_split_data, format_prompt
from train_pipeline import (
    train_model,
    plot_training_history,
    evaluate_accuracy,
    generate_predictions,
)


def load_raw_dataset(dataset_csv):
    """Step 1: 讀取原始 dataset.csv，回傳 DataFrame（不一定要立刻寫檔）。"""
    print(f"Loading raw dataset from {dataset_csv}...")
    df = pd.read_csv(dataset_csv, dtype=str)
    print(f"Raw dataset shape: {df.shape}")
    return df


def run_full_pipeline(use_kaggle=True):
    """Step 6: 主要入口，串接 Step1~Step5。"""

    # --------------------------------------------------
    # 路徑設定 (Kaggle vs local)
    # --------------------------------------------------
    if use_kaggle:
        data_root = "/kaggle/input/aihw1_dataset_splitted"
        dataset_csv = "/kaggle/input/aihw1_dataset_splitted/dataset.csv"
        benchmark_csv = "/kaggle/input/aihw1_dataset_splitted/benchmark.csv"
        output_dir = "/kaggle/working/saved_models/lora_finetuned"
    else:
        data_root = "dataset"
        dataset_csv = os.path.join(data_root, "dataset.csv")
        benchmark_csv = os.path.join(data_root, "benchmark.csv")
        output_dir = "saved_models/lora_finetuned"

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Step 1: 切分 dataset (讀取檔案並產生 train/val/test)
    # --------------------------------------------------
    raw_df = load_raw_dataset(dataset_csv)

    # load_and_split_data 會輸出 train/val/test csv (現有函式)
    train_df, val_df, test_df = load_and_split_data(dataset_csv)

    # --------------------------------------------------
    # Step 2: prompt formatting (由 split_dataset.format_prompt 自動完成)
    # --------------------------------------------------
    # 此處我們不需額外操作，因爲 train/eval/generate 都會用 format_prompt
    sample_prompt = format_prompt(test_df.iloc[0], is_test=True)
    print("Sample prompt:\n", sample_prompt)

    # --------------------------------------------------
    # Step 3: LoRA 訓練
    # --------------------------------------------------
    trainer = train_model(
        train_csv=os.path.join(data_root, "train.csv"),
        val_csv=os.path.join(data_root, "val.csv"),
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        max_seq_length=512,
    )

    # Step 4: training history 繪圖 + 測試集驗證
    plot_training_history(trainer, output_path=os.path.join(output_dir, "training_history.png"))

    best_model_path = output_dir
    test_csv = os.path.join(data_root, "test.csv")

    evaluate_accuracy(
        model=None,
        tokenizer=None,
        test_csv=test_csv,
        max_seq_length=512,
        max_new_tokens=32,
        output_dir=output_dir,
    )

    # Step 5: benchmark 推論 (Kaggle 提交檔案)
    submission_csv = os.path.join(output_dir, "benchmark_submission.csv")
    generate_predictions(
        model=None,
        tokenizer=None,
        benchmark_csv=benchmark_csv,
        output_csv=submission_csv,
        output_dir=output_dir,
        max_seq_length=512,
        max_new_tokens=64,
        num_beams=4,
    )

    print("Pipeline finished.")
    print("Submission file:", submission_csv)


if __name__ == "__main__":
    # Kaggle 執行環境預設 True；若在本機為 False
    run_full_pipeline(use_kaggle=True)
