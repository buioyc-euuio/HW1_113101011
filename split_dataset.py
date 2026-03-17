import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_data(file_path):
    """
    讀取資料集並切分為 Training, Validation, Testing sets。
    """
    # 1. 載入資料：讀取 dataset.csv，包含 question_id, question, opa, opb, opc, opd, ans [cite: 54, 55, 56, 57, 58]
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 2. 資料切分 
    # 先切分出 80% 作為訓練集，20% 作為暫存集 (用來分給驗證與測試)
    # 使用 stratify 保持每個 ans 類別比例一致
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ans'])
    
    # 將剩下的 20% 平分為驗證集 (10%) 與測試集 (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['ans'])
    
    print(f"Data split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 儲存為 CSV 檔案
    train_df.to_csv("dataset/train.csv", index=False)
    val_df.to_csv("dataset/val.csv", index=False)
    test_df.to_csv("dataset/test.csv", index=False)
    print("Saved split files: dataset/train.csv, dataset/val.csv, dataset/test.csv")

    return train_df, val_df, test_df

def format_prompt(row, is_test=False):
    """
    將資料中的問題與選項格式化為結構化的提示 (Prompt) 。
    如果是測試階段 (is_test=True)，則不放入正確答案。
    """
    # 提取各個欄位的內容 [cite: 56, 57]
    question = row['question']
    opa = row['opa']
    opb = row['opb']
    opc = row['opc']
    opd = row['opd']
    
    # 設計基礎的 Prompt 模板 (Zero-Shot 範例)
    prompt = f"""You are a helpful medical expert. Please answer the following pathology multiple-choice question.

Question: {question}
A) {opa}
B) {opb}
C) {opc}
D) {opd}

Please select the correct option (A, B, C, or D) and provide the answer.
Answer:"""

    # 如果是訓練或驗證階段，我們需要把 Ground Truth 答案附加上去 
    if not is_test:
        ans = row['ans']
        prompt += f" {ans}"
        
    return prompt


# ==========================================
# 實際執行範例
# ==========================================
if __name__ == "__main__":
    # 假設你的資料放在 dataset/ 資料夾下
    dataset_path = "dataset/dataset.csv"
    
    # 執行切分
    train_data, val_data, test_data = load_and_split_data(dataset_path)
    
    # 測試 Prompt 格式化 (以訓練集的第一筆資料為例)
    sample_row = train_data.iloc[0]
    sample_prompt = format_prompt(sample_row, is_test=False)
    
    print("\n--- Sample Formatted Prompt ---")
    print(sample_prompt)