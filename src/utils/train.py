import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import random

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def get_tokenizer_model(model_id, checkpoint=None, dropout=0.0, labels="hard"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    configuration = AutoConfig.from_pretrained(model_id)
    configuration.hidden_dropout_prob = dropout
    configuration.attention_probs_dropout_prob = dropout
    configuration.classifier_dropout = dropout
    configuration.num_labels = 1 if labels == "soft" else 2

    if checkpoint:
        model_id = checkpoint

    model = AutoModelForSequenceClassification.from_pretrained(model_id, config=configuration)
    return tokenizer, model


def tokenize(examples, tokenizer, max_tokens):
    tokenized_inputs = tokenizer(
        examples["text_clean"], padding="max_length", truncation=True, max_length=max_tokens
    )
    return tokenized_inputs


def preprocessing_data(data, tokenizer, max_tokens):
    dt = Dataset.from_pandas(data)
    tokenized_dt = dt.map(
        lambda x: tokenize(x, tokenizer, max_tokens), remove_columns=["text_clean"], batched=True
    )
    return tokenized_dt.with_format("torch")


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_data(tokenizer, df_train, df_test=None, labels="hard", max_tokens=512):
    if labels == "hard":
        col = "sexist"
    elif labels == "soft":
        col = "sexist_soft"
    else:
        raise ValueError()

    df_val = df_train.sample(frac=0.2, random_state=42)
    df_train = df_train.drop(df_val.index)

    tok_train = preprocessing_data(
        df_train[["text_clean", col]].rename(columns={col: "labels"}), tokenizer, max_tokens
    )
    tok_val = preprocessing_data(
        df_val[["text_clean", col]].rename(columns={col: "labels"}), tokenizer, max_tokens
    )
    if df_test is not None:
        tok_test = preprocessing_data(df_test[["text_clean"]], tokenizer, max_tokens)
    else:
        tok_test = None

    return tok_train, tok_val, tok_test


training_args = TrainingArguments(
    output_dir="baseline_es",
    learning_rate=5e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    # eval_strategy="epoch",
    # save_strategy="epoch",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=10,
    save_steps=10,
    load_best_model_at_end=True,
    save_total_limit=3,
)


def train(
    df_train,
    df_test=None,
    tokenizer=None,
    model=None,
    labels="hard",
    max_tokens=512,
    resume_from_checkpoint=False,
    train=True,
):
    set_deterministic()
    tok_train, tok_val, tok_test = get_data(
        tokenizer, df_train, df_test, labels=labels, max_tokens=max_tokens
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )
    if train:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    best_checkpoint = trainer.state.best_model_checkpoint

    if df_test is not None:
        predictions = trainer.predict(tok_test)
        results = df_test.copy()
        if labels == "soft":
            results["prediction"] = np.clip(predictions[0], 0, 1)
        elif labels == "hard":
            results["prediction"] = np.argmax(predictions[0], axis=1)
        del predictions
    else:
        results = None

    del trainer
    return results, best_checkpoint


def predict(tokenizer=None, model=None, labels="hard"):
    results = train(tokenizer, model, labels, train=False)
    return results
