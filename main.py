# !pip install trl
import os
import re
import inspect

import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

try:
    from trl import SFTTrainer
except ImportError as e:
    raise ImportError('trl not installed. Install it with `pip install trl` and retry.') from e


def load_raw_dataset(dataset_csv):
    """Step 1: 讀取原始 dataset.csv，回傳 DataFrame（不一定要立刻寫檔）。"""
    print(f'Loading raw dataset from {dataset_csv}...')
    df = pd.read_csv(dataset_csv, dtype=str)
    print(f'Raw dataset shape: {df.shape}')
    return df


def load_and_split_data(file_path):
    print(f'Loading data from {file_path}...')
    df = pd.read_csv(file_path)

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['ans'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['ans'])

    print(f'Data split completed: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}')

    os.makedirs('dataset', exist_ok=True)
    train_df.to_csv('dataset/train.csv', index=False)
    val_df.to_csv('dataset/val.csv', index=False)
    test_df.to_csv('dataset/test.csv', index=False)
    print('Saved split files: dataset/train.csv, dataset/val.csv, dataset/test.csv')

    return train_df, val_df, test_df


def format_prompt(row, is_test=False):
    question = row['question']
    opa = row['opa']
    opb = row['opb']
    opc = row['opc']
    opd = row['opd']

    prompt = f"""You are a helpful medical expert. Please answer the following pathology multiple-choice question.\n\nQuestion: {question}\nA) {opa}\nB) {opb}\nC) {opc}\nD) {opd}\n\nPlease select the correct option (A, B, C, or D) and provide the answer.\nAnswer:"""

    if not is_test:
        ans = row['ans']
        prompt += f' {ans}'

    return prompt


def setup_model_and_lora():
    model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    print(f'Loading tokenizer and model: {model_name}...')

    # 從 Kaggle Secrets 讀取 HF_TOKEN，並 login Hugging Face
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret('HF_TOKEN')
    if not hf_token:
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

    if not hf_token:
        raise RuntimeError(
            'HF_TOKEN not found in Kaggle secrets or environment. ' 
            'Please set it in Kaggle Secrets as HF_TOKEN.'
        )

    login(token=hf_token)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )

    print('Setting up LoRA configuration...')
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


from datasets import Dataset


def _build_causal_dataset(texts, tokenizer, max_length=512):
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt',
    )
    labels = encodings.input_ids.clone()
    attention_mask = encodings.attention_mask
    labels[attention_mask == 0] = -100

    # 轉為 HuggingFace Dataset 以符合 SFTTrainer 需求
    dataset = Dataset.from_dict({
        'input_ids': encodings.input_ids.tolist(),
        'attention_mask': encodings.attention_mask.tolist(),
        'labels': labels.tolist(),
    })
    return dataset


def train_model(
    train_csv=None,
    val_csv=None,
    output_dir=None,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    max_seq_length=512,
    logging_steps=20,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    save_total_limit=2,
):
    kaggle_input_dir = '/kaggle/input/datasets/nycu113101011/aihw1-dataset-splitted'
    kaggle_output_dir = '/kaggle/working/saved_models/lora_finetuned'

    train_csv = train_csv or os.path.join(kaggle_input_dir, 'train.csv')
    val_csv = val_csv or os.path.join(kaggle_input_dir, 'val.csv')
    output_dir = output_dir or kaggle_output_dir

    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = setup_model_and_lora()

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_texts = [format_prompt(r, is_test=False) for _, r in train_df.iterrows()]
    val_texts = [format_prompt(r, is_test=False) for _, r in val_df.iterrows()]

    train_dataset = _build_causal_dataset(train_texts, tokenizer, max_length=max_seq_length)
    eval_dataset = _build_causal_dataset(val_texts, tokenizer, max_length=max_seq_length)

    training_args_kwargs = {
        'output_dir': output_dir,
        'num_train_epochs': num_train_epochs,
        'per_device_train_batch_size': per_device_train_batch_size,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'learning_rate': learning_rate,
        'logging_steps': logging_steps,
        'save_strategy': save_strategy,
        'evaluation_strategy': evaluation_strategy,
        'save_total_limit': save_total_limit,
        'fp16': torch.cuda.is_available(),
        'optim': 'adamw_torch',
        'report_to': 'none',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'loss',
    }

    # 兼容不同 transformers 版本（evaluation_strategy 參數可能被改名或刪除）
    valid_params = inspect.signature(TrainingArguments.__init__).parameters
    training_args_kwargs = {
        k: v
        for k, v in training_args_kwargs.items()
        if k in valid_params
    }

    if 'evaluation_strategy' not in training_args_kwargs and 'eval_strategy' in valid_params:
        training_args_kwargs['eval_strategy'] = evaluation_strategy

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # 方便後續評估/推論直接使用 tokenizer
    trainer.tokenizer = tokenizer

    print('Starting training...')
    trainer.train()

    print(f'Saving fine-tuned model to {output_dir}...')
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer, model, tokenizer


def plot_training_history(trainer, output_path='saved_models/training_history.png'):
    history = trainer.state.log_history
    train_epochs, train_losses, eval_epochs, eval_losses = [], [], [], []

    for entry in history:
        if 'loss' in entry and 'epoch' in entry and 'step' in entry and entry.get('eval_loss') is None:
            train_epochs.append(entry['epoch'])
            train_losses.append(entry['loss'])
        if 'eval_loss' in entry and 'epoch' in entry:
            eval_epochs.append(entry['epoch'])
            eval_losses.append(entry['eval_loss'])

    if len(train_epochs) == 0 and len(eval_epochs) == 0:
        print('無法從 trainer state 取得 loss 資訊，請確認是否開啟 logging/eval。')
        return

    plt.figure(figsize=(8, 5))
    if len(train_losses) > 0:
        plt.plot(train_epochs, train_losses, label='Train Loss', marker='o')
    if len(eval_losses) > 0:
        plt.plot(eval_epochs, eval_losses, label='Validation Loss', marker='o')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f'Saved training history plot to {output_path}')


def _normalize_answer(ans):
    return ans.strip().upper()


def _extract_predicted_option(text):
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None


def evaluate_accuracy(
    model=None,
    tokenizer=None,
    test_csv='dataset/test.csv',
    max_seq_length=512,
    max_new_tokens=32,
    output_dir=None,
):
    if model is None or tokenizer is None:
        model, tokenizer = setup_model_and_lora()
        if output_dir:
            model = PeftModel.from_pretrained(model, output_dir)

    if model is None:
        raise RuntimeError('evaluate_accuracy: model is None after setup. Please check setup_model_and_lora or load_model_from_hf.')
    if tokenizer is None:
        raise RuntimeError('evaluate_accuracy: tokenizer is None after setup. Please check setup_model_and_lora.')

    test_df = pd.read_csv(test_csv)

    model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(model_device)
    model.eval()

    correct = 0
    total = 0

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Test eval', unit='row'):
        prompt = format_prompt(row, is_test=True)
        inputs = tokenizer(prompt, truncation=True, max_length=max_seq_length, return_tensors='pt').to(model_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        predicted = _extract_predicted_option(decoded)
        expected = _normalize_answer(row['ans'])

        if predicted is not None and predicted == expected:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f'Test Accuracy: {accuracy:.4f} ({correct}/{total})')

    if output_dir:
        with open(os.path.join(output_dir, 'test_accuracy.txt'), 'w', encoding='utf-8') as f:
            f.write(f'accuracy={accuracy:.6f}\ncorrect={correct}\ntotal={total}\n')

    return accuracy


def generate_predictions(
    model=None,
    tokenizer=None,
    benchmark_csv=None,
    output_csv='benchmark_predictions.csv',
    output_dir=None,
    max_seq_length=512,
    max_new_tokens=64,
    num_beams=4,
):
    benchmark_csv = benchmark_csv or os.path.join('dataset', 'benchmark.csv')

    if model is None or tokenizer is None:
        model, tokenizer = setup_model_and_lora()
        if output_dir:
            model = PeftModel.from_pretrained(model, output_dir)

    model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(model_device)
    model.eval()

    benchmark_df = pd.read_csv(benchmark_csv, dtype=str)

    results = []
    for _, row in tqdm(benchmark_df.iterrows(), total=len(benchmark_df), desc='Benchmark inference', unit='q'):
        prompt = format_prompt(row, is_test=True)
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_tensors='pt',
        ).to(model_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        predicted = _extract_predicted_option(generated_text)
        if predicted is None:
            predicted = 'A'

        question_id = row.get('question_id', None)
        results.append({'question_id': question_id, 'ans': predicted})

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f'Saved Kaggle submission predictions to {output_csv}')
    return submission_df


def run_full_pipeline(use_kaggle=True):
    if use_kaggle:
        data_root = '/kaggle/input/datasets/nycu113101011/aihw1-dataset-splitted'
        dataset_csv = '/kaggle/input/datasets/nycu113101011/aihw1-dataset-splitted/dataset.csv'
        benchmark_csv = '/kaggle/input/datasets/nycu113101011/aihw1-dataset-splitted/benchmark.csv'
        output_dir = '/kaggle/working/saved_models/lora_finetuned'
    else:
        data_root = 'dataset'
        dataset_csv = os.path.join(data_root, 'dataset.csv')
        benchmark_csv = os.path.join(data_root, 'benchmark.csv')
        output_dir = 'saved_models/lora_finetuned'

    os.makedirs(output_dir, exist_ok=True)

    raw_df = load_raw_dataset(dataset_csv)
    train_df, val_df, test_df = load_and_split_data(dataset_csv)

    sample_prompt = format_prompt(test_df.iloc[0], is_test=True)
    print('Sample prompt:\n', sample_prompt)

    trainer, model, tokenizer = train_model(
        train_csv=os.path.join(data_root, 'train.csv'),
        val_csv=os.path.join(data_root, 'val.csv'),
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        max_seq_length=512,
    )

    plot_training_history(trainer, output_path=os.path.join(output_dir, 'training_history.png'))

    evaluate_accuracy(
        model=model,
        tokenizer=tokenizer,
        test_csv=os.path.join(data_root, 'test.csv'),
        max_seq_length=512,
        max_new_tokens=32,
        output_dir=output_dir,
    )

    submission_csv = os.path.join(output_dir, 'benchmark_submission.csv')
    generate_predictions(
        model=model,
        tokenizer=tokenizer,
        benchmark_csv=benchmark_csv,
        output_csv=submission_csv,
        output_dir=output_dir,
        max_seq_length=512,
        max_new_tokens=64,
        num_beams=4,
    )

    print('Pipeline finished.')
    print('Submission file:', submission_csv)


if __name__ == '__main__':
    run_full_pipeline(use_kaggle=True)
