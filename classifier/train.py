
import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
OUTPUT_DIR = PROJECT_ROOT/ "classifier" / "models" / "sentiment_model_v0"
DATA_PATH = PROJECT_ROOT/ "classifier" / "data" / "sentimentos.csv"

print(f"Arquivo CSV: {DATA_PATH}")
print(f"Arquivo existe? {DATA_PATH.exists()}")

df = pd.read_csv(DATA_PATH).dropna(
    subset=["texto", "sentimento"]
)



LABELS = {
    "alegria" : 0,
    "tristeza" : 1,
    "medo" : 2,
    "raiva" : 3,
    "surpresa" : 4,
    "nojo" : 5
}


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, MODEL_NAME):
        self.texts = dataframe["texto"].tolist()
        self.labels = [
            LABELS[label]
            for label in dataframe["sentimento"]
        ]

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item



# 5. Métricas de avaliação
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions)
    }


def treino():
    
    # 1. Carregar e validar os dados
    df = pd.read_csv(DATA_PATH).dropna(
        subset=["texto", "sentimento"]
    )
    print("Colunas encontradas:", df.columns.tolist())
    print(df.head())

    df["texto"] = df["texto"].astype(str).str.strip()
    df["sentimento"] = df["sentimento"].str.strip().str.lower()
    df = df[df["texto"] != ""]

    if not set(df["sentimento"]).issubset(LABELS):
        raise ValueError("Existem rótulos de sentimento inválidos.")

    if df["sentimento"].nunique() != len(LABELS):
        raise ValueError("O conjunto deve conter as três classes.")

    if df["sentimento"].value_counts().min() < 2:
        raise ValueError(
            "São necessários pelo menos dois exemplos por classe."
        )




    # 2. Separar treino e validação
    print("\nQuantidade de exemplos por sentimento:")
    print(df["sentimento"].value_counts())

    print("\nTotal de registros:", len(df))

    train_df, val_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["sentimento"],
    )


    # 3. Tokenizar os textos
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    train_dataset = SentimentDataset(train_df,MODEL_NAME)
    val_dataset = SentimentDataset(val_df,MODEL_NAME)


    # 4. Carregar BERT com cabeça de classificação
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={v: k for k, v in LABELS.items()},
        label2id=LABELS,
    )



    # 6. Configurar treinamento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        save_total_limit=1,
        seed=42,
    )


    # 7. Treinar e avaliar
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("Métricas:", metrics)

    predictions = trainer.predict(val_dataset)
    print(
        classification_report(
            predictions.label_ids,
            np.argmax(predictions.predictions, axis=-1),
            labels=list(LABELS.values()),
            target_names=list(LABELS.keys()),
            zero_division=0,
        )
    )

    # 8. Salvar modelo e tokenizador
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Modelo salvo em: {OUTPUT_DIR}")
    return OUTPUT_DIR

if __name__ == "__main__":
    treino()