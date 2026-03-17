import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

"""
我會想知道 這邊每一段的細節是甚麼，背後的運作模式，以及他確切來說到底會從HF下載下來什麼檔案，
為什麼 llama 願意開源? 那它的商業機密怎麼半? 他會開源什麼 但不公開甚麼?
為什麼我能下載他的model 然後甚至能用LoRA去 finetune? 
> lora_config = LoraConfig() 
這個 config 裡面在幹嘛?
> # 5. 將 LoRA 外掛應用到模型上
> model = get_peft_model(model, lora_config)
具體來說，數學上，這是怎麼做到的？ 怎麼將 LoRA 外掛應用到模型上? 怎麼插入的?
lora 具體來說它的運作機制是甚麼？(這就要看老師的講義了)
這份檔案裏面LORA設定在幹嘛?
"""

def setup_model_and_lora():
    """
    載入 Llama-3.2-1B-Instruct 模型，並套用 LoRA 微調設定。
    """
    # 1. 指定作業規定的模型名稱
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"Loading tokenizer and model: {model_name}...")
    
    # 2. 載入 Tokenizer (分詞器：負責把人類文字切成模型看得懂的數字)
    # Llama 3 預設沒有 padding token，我們通常會把 eos_token 設為 padding token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 訓練時通常靠右對齊
    
    # 3. 載入基礎模型 (Base Model)
    # 為了節省 VRAM (顯示卡記憶體)，我們通常會用 bfloat16 或 float16 載入
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", # 自動將模型分配到 GPU 上
        torch_dtype=torch.bfloat16, # 使用 bfloat16 可以節省記憶體並加速
    )
    
    # 4. 設定 LoRA 參數 (LoraConfig)
    print("Setting up LoRA configuration...")
    lora_config = LoraConfig(
        r=16,               # r (Rank): 便利貼的大小。數值越大能學到越多細節，但越耗資源 (通常設 8, 16, 32)
        lora_alpha=32,      # lora_alpha: 縮放係數，通常設定為 r 的 2 倍
        target_modules=[    # 指定要貼便利貼的注意力機制模組 (Llama 通常是 q_proj, v_proj 等)
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj"
        ],
        lora_dropout=0.05,  # Dropout 比例，用來防止過度擬合 (Overfitting)
        bias="none",        # 通常不需要訓練 bias
        task_type="CAUSAL_LM" # 我們的任務是語言模型生成 (Causal Language Modeling)
    )
    
    # 5. 將 LoRA 外掛應用到模型上
    model = get_peft_model(model, lora_config)
    
    # 印出可以訓練的參數比例，你會發現我們只需要訓練不到 1% 的參數！
    model.print_trainable_parameters()
    
    return model, tokenizer

# ==========================================
# 實際執行範例
# ==========================================
if __name__ == "__main__":
    # 呼叫函式來測試是否能成功載入
    # 注意：這一步會需要下載約 2GB~3GB 的模型權重，請確保網路暢通
    model, tokenizer = setup_model_and_lora()
    print("模型與 LoRA 設定完成！準備進入訓練階段。")