import os
import argparse
import json
import logging
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from evaluate import load as load_metric
import time



# ======================================================
# 1. 强制进入离线模式（防止意外联网）
# ======================================================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# ======================================================
# 2. 日志系统
# ======================================================
def setup_logging(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train.log")

    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    console.setFormatter(formatter)

    logging.getLogger("").handlers = []
    logging.getLogger("").addHandler(console)
    logging.getLogger("").addHandler(logging.FileHandler(log_path))


# ======================================================
# 3. 本地 SST-2 数据集加载（你自己上传的 tsv 文件）
# ======================================================
from datasets import Dataset, DatasetDict

def load_local_sst2(tokenizer, data_dir="./sst-2", max_length=128):

    def read_tsv(path):
        sentences = []
        labels = []
        with open(path, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                text, label = line.strip().split("\t")
                sentences.append(text)
                labels.append(int(label))
        return sentences, labels

    # 读取 tsv 文件
    train_texts, train_labels = read_tsv(os.path.join(data_dir, "train.tsv"))
    dev_texts, dev_labels = read_tsv(os.path.join(data_dir, "dev.tsv"))

    # tokenize
    train_enc = tokenizer(train_texts, truncation=True, padding=False, max_length=max_length)
    dev_enc = tokenizer(dev_texts, truncation=True, padding=False, max_length=max_length)

    # 构建 HuggingFace Dataset（离线无需 load_dataset）
    train_dataset = Dataset.from_dict({
        "input_ids": train_enc["input_ids"],
        "attention_mask": train_enc["attention_mask"],
        "label": train_labels,
    })

    dev_dataset = Dataset.from_dict({
        "input_ids": dev_enc["input_ids"],
        "attention_mask": dev_enc["attention_mask"],
        "label": dev_labels,
    })

    # 返回 DatasetDict
    return DatasetDict({
        "train": train_dataset,
        "validation": dev_dataset
    })

# ======================================================
# 4. LoRA 配置
# ======================================================
def build_lora_config(rank, alpha, dropout, improved=False):

    if improved:
        target_modules = ["query", "value", "intermediate.dense"]  # 改进版：Attention + MLP
    else:
        target_modules = ["query", "value"]  # 原始论文：只对 q, v (RoBERTa uses "query" and "value" instead of "q_proj" and "v_proj")

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

# ======================================================
# 5. 评估指标统计总参数量和可训练参数量
# ======================================================
def count_params(model):
    """
    Returns (total_params, trainable_params)
    """
    total = 0
    trainable = 0
    for p in model.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    return total, trainable


# ======================================================
# 6. 主训练函数（离线版）
# ======================================================
def main(args):
    setup_logging(args.output_dir)

    logging.info("===== Offline Training Mode =====")
    logging.info("Using local model and local dataset only.")

    # 模型路径（已上传）
    model_path = "./robeata"
    
    # Convert to absolute path - this is crucial for transformers to recognize it as a local path
    model_path = os.path.abspath(model_path)
    
    # Verify the path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Please ensure the roberta model directory exists."
        )
    
    # Check for required files
    required_files = ["config.json", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        raise FileNotFoundError(
            f"Model directory is missing required files: {missing_files}\n"
            f"Model path: {model_path}"
        )
    
    logging.info(f"Loading model from local path: {model_path}")

    # 加载 tokenizer 和模型（完全离线）
    # Use local_files_only=True to prevent any network access attempts
    tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    model = RobertaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        local_files_only=True,
    )

    # 加载本地 SST-2
    dataset = load_local_sst2(tokenizer, data_dir=args.data_dir)

    data_collator = DataCollatorWithPadding(tokenizer)

    # 应用 LoRA 或 full fine-tune
    if args.use_lora:
        logging.info(
            f"Using LoRA (rank={args.rank}, alpha={args.alpha}, "
            f"dropout={args.dropout}, improved={args.improved})"
        )

        lora_config = build_lora_config(args.rank, args.alpha, args.dropout, args.improved)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logging.info("Full Fine-tuning all parameters.")

    # ========= 统计参数量 =========
    total_params, trainable_params = count_params(model)
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
    logging.info(f"Total params: {total_params}")
    logging.info(f"Trainable params: {trainable_params} "
                 f"({trainable_ratio * 100:.4f}% of total)")

    # 评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        # 注意：如果 logits/labels 是 numpy，这里保持你的原写法不动
        accuracy = (preds == labels).astype(float).mean().item()
        return {"accuracy": accuracy}

    # 训练配置
    # Only enable FP16 on CUDA devices (not supported on CPU/MPS)
    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        logging.info("CUDA detected: Enabling FP16 mixed precision training")
    else:
        logging.info("CUDA not available: Using full precision (FP32) training")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 保存超参数（先把参数统计也写进去）
    hyper = vars(args).copy()
    hyper["total_params"] = int(total_params)
    hyper["trainable_params"] = int(trainable_params)
    hyper["trainable_ratio"] = float(trainable_ratio)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "hyperparams.json"), "w") as f:
        json.dump(hyper, f, indent=4)

    # ========= 训练前记录时间和显存 =========
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = torch.cuda.memory_allocated()
    else:
        gpu_mem_before = None

    logging.info("=== Start Training ===")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    logging.info("=== Training Finished ===")

    training_time = end_time - start_time
    logging.info(f"Total training time: {training_time:.2f} seconds")

    # ========= 训练后统计显存 =========
    if torch.cuda.is_available():
        gpu_mem_after = torch.cuda.memory_allocated()
        gpu_mem_peak = torch.cuda.max_memory_allocated()
        logging.info(f"GPU memory before training: {gpu_mem_before / 1024**2:.2f} MB")
        logging.info(f"GPU memory after training: {gpu_mem_after / 1024**2:.2f} MB")
        logging.info(f"GPU peak memory during training: {gpu_mem_peak / 1024**2:.2f} MB")
    else:
        gpu_mem_after = None
        gpu_mem_peak = None
        logging.info("CUDA not available. Skip GPU memory stats.")

    # 评估
    final_metrics = trainer.evaluate()
    logging.info(f"Final Eval: {final_metrics}")

    # 把时间 / 显存信息也写进 results.json 里
    results = dict(final_metrics)
    results["total_params"] = int(total_params)
    results["trainable_params"] = int(trainable_params)
    results["trainable_ratio"] = float(trainable_ratio)
    results["training_time_sec"] = float(training_time)

    if gpu_mem_before is not None:
        results["gpu_mem_before_mb"] = float(gpu_mem_before / 1024**2)
        results["gpu_mem_after_mb"] = float(gpu_mem_after / 1024**2)
        results["gpu_mem_peak_mb"] = float(gpu_mem_peak / 1024**2)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # 保存模型
    save_dir = os.path.join(args.output_dir, "best_model")
    model.save_pretrained(save_dir)
    logging.info(f"Saved best model to: {save_dir}")


# ======================================================
# 启动入口
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--improved", action="store_true")

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--data_dir", type=str, default="./SST-2")  # 你的本地 SST-2 路径

    args = parser.parse_args()
    main(args)
