import re

import os
import shutil
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from sqlalchemy import text, bindparam

from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)




OUTPUT_DIR = PROJECT_ROOT / "classifier" / "models"

EMOCOES = [
    "alegria", "tristeza", "medo",
    "raiva", "surpresa", "nojo"
]







def gerar_metricas_bert(caminho_modelo: str, caminho_csv: str) -> dict:
    """Avalia um modelo BERT salvo no formato Hugging Face."""

    caminho_modelo = Path(caminho_modelo)
    caminho_csv = Path(caminho_csv)

    df = pd.read_csv(caminho_csv, encoding="utf-8-sig")
    df = df[["texto", "sentimento"]].dropna()

    df["sentimento"] = df["sentimento"].str.lower().str.strip()
    df["texto"] = df["texto"].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(caminho_modelo))
    modelo = AutoModelForSequenceClassification.from_pretrained(
        str(caminho_modelo)
    ).to(device)

    # Confirme que esta ordem corresponde à usada no treinamento.
    id2label = modelo.config.id2label
    mapa = {}

    for i in range(modelo.config.num_labels):
        label = id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))
        label = str(label).lower().strip()

        if label.startswith("label_"):
            label = label.split("_")[-1]

        if label.isdigit():
            label = EMOCOES[int(label)]

        if label not in EMOCOES:
            raise ValueError(f"Classe não reconhecida: {label}")

        mapa[i] = label

    modelo.eval()
    previsoes = []

    with torch.no_grad():
        for texto in df["texto"]:
            tokens = tokenizer(
                texto,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            logits = modelo(**tokens).logits
            indice = logits.argmax(dim=1).item()
            previsoes.append(mapa[indice])

    y_true = df["sentimento"].tolist()
    y_pred = previsoes

    metricas = {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao_macro": precision_score(
            y_true, y_pred, labels=EMOCOES,
            average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=EMOCOES,
            average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, labels=EMOCOES,
            average="macro", zero_division=0
        ),
        "relatorio": classification_report(
            y_true, y_pred, labels=EMOCOES,
            zero_division=0, output_dict=True
        ),
        "matriz_confusao": confusion_matrix(
            y_true, y_pred, labels=EMOCOES
        ),
    }

    for nome, valor in metricas.items():
        if isinstance(valor, float):
            print(f"{nome}: {valor:.4f}")

    print("\nRelatório por classe:")
    print(classification_report(
        y_true, y_pred, labels=EMOCOES, zero_division=0
    ))

    print("Matriz de confusão:")
    print(pd.DataFrame(
        metricas["matriz_confusao"],
        index=EMOCOES,
        columns=EMOCOES,
    ))

    return metricas



def listar_versoes():
    """
    Retorna as versões válidas em ordem numérica.
    Ignora diretórios temporários de treinamento.
    """
    padrao = re.compile(r"sentiment_model_v(\d+)$")
    versoes = []

    for caminho in OUTPUT_DIR.iterdir():
        if not caminho.is_dir():
            continue

        match = padrao.fullmatch(caminho.name)

        if match:
            versoes.append(
                (int(match.group(1)), caminho)
            )

    return sorted(versoes, key=lambda item: item[0])


if __name__ == "__main__":
    versoes = listar_versoes()
    print("Versões encontradas:")
    for numero, caminho in versoes:
        print(f"  - {numero}: {caminho}")
        gerar_metricas_bert(caminho, PROJECT_ROOT / "classifier" / "data" / "validacao_sentimentos.csv")