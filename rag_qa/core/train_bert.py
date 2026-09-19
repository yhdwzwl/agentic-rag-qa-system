#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/3/21 15:17
@File    : train_bert.py
@Function :
"""
import os
import json
import torch
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from typing import Dict, List, Union
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset

# 配置标签映射 (根据需求固定)
LABEL_MAP = {"通用无关": 0, "股票财经": 1}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


class BERTClassifier:
    def __init__(self, model_name_or_path: str, output_dir: str = "./results"):
        """
        初始化分类器
        :param model_name_or_path: BERT 模型路径或名称 (如 'bert-base-chinese')
        :param output_dir: 模型保存和训练输出目录
        """
        self.model_name = model_name_or_path
        self.output_dir = output_dir
        self.label_map = LABEL_MAP
        self.id2label = ID2LABEL

        # 加载分词器和模型
        print(f"Loading model from {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=NUM_LABELS,
            id2label=self.id2label,
            label2id=self.label_map
        )

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    def _preprocess_function(self, examples):
        """数据集预处理函数"""
        # 标签是数字0/1，直接用
        labels = examples["label"]
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=128
        )
        tokenized["labels"] = labels
        return tokenized

    def load_data(self, file_path: str):
        """加载 JSON/JSONL 数据"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # 使用 datasets 库加载 json 文件，支持 .json 和 .jsonl
        dataset = load_dataset('json', data_files=file_path)
        # 通常 load_dataset 会返回一个字典，如果是单文件通常键为 'train'
        # 这里我们强制取第一个 split 或者如果是单文件直接处理
        if 'train' in dataset:
            raw_dataset = dataset['train']
        else:
            raw_dataset = dataset[list(dataset.keys())[0]]

        # 映射标签以验证数据合法性
        def check_label(example):
            if example['label'] not in [0, 1]:
                raise ValueError(f"Unknown label: {example['label']}. Allowed: [0, 1]")
            return example

        raw_dataset = raw_dataset.map(check_label)
        return raw_dataset

    def load_data_from_mongo(self, db_name: str = "rag_doc_data", collection: str = "bert_2_data"):
        """直接从MongoDB加载数据"""
        from pymongo import MongoClient
        from datasets import Dataset

        client = MongoClient("mongodb://localhost:27017/")
        col = client[db_name][collection]

        # 查所有数据
        docs = list(col.find({}, {"_id": 0, "text": 1, "label": 1}))
        client.close()

        print(f"从MongoDB加载了 {len(docs)} 条数据")

        # 转成Dataset
        dataset = Dataset.from_list(docs)
        return dataset

    def train(self, train_dataset, eval_dataset, args: TrainingArguments = None):
        """
        训练模型
        :param train_dataset: 训练集Dataset对象
        :param eval_dataset: 验证集Dataset对象
        :param args: TrainingArguments 配置
        """
        print("Tokenizing datasets...")
        train_dataset = train_dataset.map(self._preprocess_function, batched=True)
        eval_dataset = eval_dataset.map(self._preprocess_function, batched=True)
        print(train_dataset)
        # 默认训练参数
        if args is None:
            args = TrainingArguments(
                output_dir=self.output_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
                load_best_model_at_end=True,
                logging_steps=10,
                save_total_limit=2,
                fp16=torch.cuda.is_available(),  # 自动开启混合精度
                report_to="none"  # 关闭 wandb 等报告，避免报错
            )

        # 定义评估指标计算函数
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            # 简单指标
            from sklearn.metrics import accuracy_score, f1_score
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds, average='weighted')
            }

        data_collator = DataCollatorWithPadding(self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        print("Start training...")
        self.trainer.train()

        # 保存最佳模型
        self.save_model()
        print(f"Training finished. Model saved to {self.output_dir}")

    def evaluate(self, test_file: str):
        """
        在测试集上评估，输出分类报告和混淆矩阵
        :param test_file: 测试集路径
        """
        print(f"Evaluating on {test_file}...")
        if not hasattr(self, 'trainer'):
            # 如果没有 trainer 实例（例如直接加载模型评估），重新创建一个
            self.trainer = Trainer(model=self.model, tokenizer=self.tokenizer)

        test_dataset = self.load_data(test_file)
        test_dataset = test_dataset.map(self._preprocess_function, batched=True)

        predictions = self.trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        labels = predictions.label_ids

        # 1. 分类报告
        report = classification_report(
            labels,
            preds,
            target_names=list(self.label_map.keys()),
            digits=4
        )
        print("\n--- Classification Report ---")
        print(report)

        # 保存报告到文件
        with open(os.path.join(self.output_dir, "classification_report.txt"), "w", encoding='utf-8') as f:
            f.write(report)

        # 2. 混淆矩阵
        cm = confusion_matrix(labels, preds)
        print("\n--- Confusion Matrix ---")
        print(cm)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        #             xticklabels=list(self.label_map.keys()),
        #             yticklabels=list(self.label_map.keys()))
        # plt.title('Confusion Matrix')
        # plt.ylabel('True Label')
        # plt.xlabel('Predicted Label')
        #
        # cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        # plt.savefig(cm_path)
        # print(f"Confusion Matrix saved to {cm_path}")
        # plt.close()

    def save_model(self):
        """保存模型和分词器"""
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        # 保存标签映射以便推理时使用
        with open(os.path.join(self.output_dir, "label_map.json"), "w", encoding='utf-8') as f:
            json.dump(self.label_map, f, ensure_ascii=False)

    def load_model_for_inference(self, path: str = None):
        """加载已保存的模型用于推理"""
        load_path = path if path else self.output_dir
        print(f"Loading model for inference from {load_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

        # 加载标签映射
        label_map_path = os.path.join(load_path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding='utf-8') as f:
                self.label_map = json.load(f)
            self.id2label = {v: k for k, v in self.label_map.items()}

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text: str) -> str:
        """
        单条文本推理
        :param text: 输入文本
        :return: 分类标签名称
        """
        if not hasattr(self, 'device'):
            self.load_model_for_inference()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_id = torch.max(probs, dim=-1)

        label_name = self.id2label[predicted_id.item()]
        return label_name


# ==========================================
# 训练入口
# ==========================================
if __name__ == "__main__":
    # 1. 配置模型路径
    MODEL_PATH = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\rag_qa\models\nlp_bert_document-segmentation_chinese-base"
    OUTPUT_DIR = r"F:\small_project\aiagent\ailearn\rag\Itcast_qa_system\rag_qa\models\bert_classification_output"

    # 2. 初始化分类器
    classifier = BERTClassifier(model_name_or_path=MODEL_PATH, output_dir=OUTPUT_DIR)

    # 3. 从MongoDB加载所有数据
    print("加载数据...")
    all_dataset = classifier.load_data_from_mongo()

    # 4. 拆分训练集和验证集（80%训练，20%验证）
    split = all_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条")

    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=10,
        save_total_limit=2,
    )

    # 6. 开始训练
    print("\n开始训练...")
    classifier.train(train_dataset, eval_dataset, training_args)

    print("\n训练完成！")
