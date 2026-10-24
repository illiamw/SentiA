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


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)




OUTPUT_DIR = PROJECT_ROOT / "classifier" / "models"
VALIDACAO_PATH = (
    PROJECT_ROOT / "classifier" / "data"
    / "validacao_sentimentos.csv"
)

BATCH_MINIMO = 5
MAX_VERSOES = 5

ROTULOS = [
    "alegria",
    "tristeza",
    "medo",
    "raiva",
    "surpresa",
    "nojo",
]

LABEL2ID = {
    nome: i for i, nome in enumerate(ROTULOS)
}

ID2LABEL = {
    i: nome for nome, i in LABEL2ID.items()
}

# Evita que duas instâncias deste script treinem ao mesmo tempo
POSTGRES_LOCK_ID = 87421031

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANTE:
# Importe o engine da configuração que seu projeto já utiliza.
# Exemplo, se ele estiver em app.py:
# from app import engine
#
# Não crie um segundo engine com credenciais diferentes.


# ============================================================
# MÉTRICAS
# ============================================================

def metricas(resultado):
    logits, labels = resultado

    previsoes = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, previsoes),
        "f1_macro": f1_score(
            labels,
            previsoes,
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def preparar_dataframe(df):
    df = df.dropna(
        subset=["texto", "sentimento"]
    ).copy()

    df["texto"] = df["texto"].astype(str).str.strip()

    df["sentimento"] = (
        df["sentimento"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[df["texto"] != ""].copy()

    invalidos = set(df["sentimento"]) - set(ROTULOS)

    if invalidos:
        raise ValueError(
            f"Sentimentos inválidos encontrados: {invalidos}"
        )

    df["label"] = df["sentimento"].map(LABEL2ID)

    return df


def preparar_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(
        df[["texto", "label"]].reset_index(drop=True)
    )

    def tokenizar(batch):
        return tokenizer(
            batch["texto"],
            truncation=True,
            max_length=256,
        )

    return dataset.map(
        tokenizar,
        batched=True,
        remove_columns=["texto"],
    )


# ============================================================
# VERSIONAMENTO DOS MODELOS
# ============================================================

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


def obter_ultima_versao():
    versoes = listar_versoes()

    if not versoes:
        raise FileNotFoundError(
            "Nenhum modelo treinado encontrado em "
            f"{OUTPUT_DIR}. Treine o modelo inicial "
            "antes de executar o treinamento incremental."
        )

    return versoes[-1][1]


def limpar_versoes_antigas():
    """
    Mantém apenas as MAX_VERSOES versões mais recentes.
    """
    versoes = listar_versoes()

    antigas = versoes[:-MAX_VERSOES]

    for numero, caminho in antigas:
        logger.info(
            "Removendo modelo antigo: %s", caminho
        )

        shutil.rmtree(caminho)


# ============================================================
# TREINAMENTO INCREMENTAL
# ============================================================

def treinar_incremental(engine):
    """
    Treina quando houver pelo menos BATCH_MINIMO registros
    com traino = FALSE.

    Os registros são marcados como utilizados somente após
    o salvamento bem-sucedido do modelo.
    """
    print("Iniciando treinamento incremental...")

    # Mantém a conexão aberta para conservar o lock de sessão.
    with engine.connect() as connection:

        # connection.execute(
        #     text(
        #         "SELECT pg_advisory_lock(:lock_id)"
        #     ),
        #     {"lock_id": POSTGRES_LOCK_ID},
        # )
        # connection.commit()
        print("Buscando registros pendentes...")
        try:
            # ------------------------------------------------
            # 1. Buscar registros pendentes
            # ------------------------------------------------
            
            df_novos = pd.read_sql(
                text("""
                    SELECT id, texto, sentimento
                    FROM sentimentos
                    WHERE traino = FALSE
                    ORDER BY id
                """),
                connection,
            )

            connection.commit()

            quantidade = len(df_novos)

            logger.info(
                "Registros pendentes: %s", quantidade
            )

            if quantidade < BATCH_MINIMO:
                logger.info(
                    "Treinamento não iniciado. "
                    "São necessários pelo menos %s registros.",
                    BATCH_MINIMO,
                )
                return

            ids_treinados = df_novos["id"].tolist()

            # ------------------------------------------------
            # 2. Validar novos registros
            # ------------------------------------------------
            print("Validando registros pendentes...")
            df_novos = preparar_dataframe(df_novos)

            if len(df_novos) != quantidade:
                raise ValueError(
                    "Existem registros inválidos ou vazios. "
                    "Nenhum registro será marcado como treinado."
                )

            # ------------------------------------------------
            # 3. Carregar validação independente
            # ------------------------------------------------

            if not VALIDACAO_PATH.exists():
                raise FileNotFoundError(
                    "Arquivo de validação não encontrado: "
                    f"{VALIDACAO_PATH}"
                )

            df_validacao = preparar_dataframe(
                pd.read_csv(VALIDACAO_PATH)
            )

            if df_validacao.empty:
                raise ValueError(
                    "O conjunto de validação está vazio."
                )

            if set(df_validacao["label"]) != set(
                range(len(ROTULOS))
            ):
                raise ValueError(
                    "O conjunto de validação precisa conter "
                    "exemplos das seis emoções."
                )

            # ------------------------------------------------
            # 4. Carregar modelo anterior
            # ------------------------------------------------
            print("Carregando modelo anterior...")
            modelo_anterior = obter_ultima_versao()

            versoes = listar_versoes()
            proxima_versao = versoes[-1][0] + 1

            modelo_novo = (
                OUTPUT_DIR
                / f"sentiment_model_v{proxima_versao}"
            )

            diretorio_temporario = (
                OUTPUT_DIR
                / f".tmp_sentiment_model_v{proxima_versao}"
            )

            diretorio_treino = (
                OUTPUT_DIR
                / f".training_v{proxima_versao}"
            )

            logger.info(
                "Modelo de origem: %s", modelo_anterior
            )
            logger.info(
                "Nova versão: %s", modelo_novo
            )

            # Não sobrescrever modelos existentes.
            if modelo_novo.exists():
                raise FileExistsError(modelo_novo)

            shutil.rmtree(
                diretorio_temporario,
                ignore_errors=True,
            )
            shutil.rmtree(
                diretorio_treino,
                ignore_errors=True,
            )

            print("Iniciando treinamento incremental...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(modelo_anterior)
                )

                model = AutoModelForSequenceClassification.from_pretrained(
                    str(modelo_anterior)
                )

                # Confere se o modelo anterior utiliza o mesmo
                # mapeamento das seis emoções.
                config = model.config

                if config.num_labels != len(ROTULOS):
                    raise ValueError(
                        "O modelo anterior não possui seis classes."
                    )

                if config.label2id != LABEL2ID:
                    raise ValueError(
                        "O mapeamento de rótulos do modelo anterior "
                        "não corresponde a LABEL2ID."
                    )

                # ------------------------------------------------
                # 5. Preparar exclusivamente os dados novos
                # ------------------------------------------------

                dataset_treino = preparar_dataset(
                    df_novos,
                    tokenizer,
                )

                dataset_validacao = preparar_dataset(
                    df_validacao,
                    tokenizer,
                )

                # ------------------------------------------------
                # 6. Continuar o treinamento
                # ------------------------------------------------

                args = TrainingArguments(
                    output_dir=str(diretorio_treino),
                    num_train_epochs=2,
                    learning_rate=5e-6,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="f1_macro",
                    greater_is_better=True,
                    save_total_limit=1,
                    report_to="none",
                    seed=42,
                )

                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=dataset_treino,
                    eval_dataset=dataset_validacao,
                    processing_class=tokenizer,
                    compute_metrics=metricas,
                )

                logger.info(
                    "Iniciando treinamento incremental..."
                )

                trainer.train()

                # Avalia a melhor versão selecionada pelo Trainer.
                resultados = trainer.evaluate()

                logger.info(
                    "Métricas finais: %s", resultados
                )

                # ------------------------------------------------
                # 7. Salvar a nova versão
                # ------------------------------------------------

                trainer.save_model(
                    str(diretorio_temporario)
                )

                tokenizer.save_pretrained(
                    str(diretorio_temporario)
                )

                # Verificação básica dos arquivos salvos.
                arquivos_obrigatorios = [
                    "config.json",
                    "model.safetensors",
                ]

                if not all(
                    (diretorio_temporario / arquivo).exists()
                    for arquivo in arquivos_obrigatorios
                ):
                    # Algumas configurações podem salvar pesos
                    # em pytorch_model.bin em vez de safetensors.
                    if not (
                        diretorio_temporario
                        / "pytorch_model.bin"
                    ).exists():
                        raise RuntimeError(
                            "Os pesos do modelo não foram salvos "
                            "corretamente."
                        )
                print("Treinamento incremental concluído com sucesso.")
                # Renomeia o diretório temporário para publicar
                # a versão final no mesmo sistema de arquivos.
                diretorio_temporario.rename(modelo_novo)

                logger.info(
                    "Modelo salvo com sucesso: %s",
                    modelo_novo,
                )

                # ------------------------------------------------
                # 8. Marcar como treinados apenas os IDs usados
                # ------------------------------------------------
                print("Marcando registros como treinados...")
                with connection.begin():
                    resultado_update = connection.execute(
                        text("""
                            UPDATE sentimentos
                            SET traino = TRUE
                            WHERE id IN :ids
                              AND traino = FALSE
                        """).bindparams(
                            bindparam("ids", expanding=True)
                        ),
                        {"ids": ids_treinados},
                    )

                logger.info(
                    "Registros marcados como treinados: %s",
                    resultado_update.rowcount,
                )

                # ------------------------------------------------
                # 9. Manter somente as cinco últimas versões
                # ------------------------------------------------

                limpar_versoes_antigas()

                logger.info(
                    "Treinamento incremental concluído."
                )

            finally:
                # Remove checkpoints e arquivos temporários.
                shutil.rmtree(
                    diretorio_treino,
                    ignore_errors=True,
                )
                shutil.rmtree(
                    diretorio_temporario,
                    ignore_errors=True,
                )

        finally:
            print("Finalizando treinamento incremental.")
        #     # Libera o lock mesmo em caso de exceção.
        #     connection.execute(
        #         text(
        #             "SELECT pg_advisory_unlock(:lock_id)"
        #         ),
        #         {"lock_id": POSTGRES_LOCK_ID},
        #     )
        #     connection.commit()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    # Importe o engine real do seu projeto antes de executar.
    # Exemplo:
    # from app import engine
    POSTGRES_DB= os.environ["POSTGRES_DB"]
    POSTGRES_USER= os.environ["POSTGRES_USER"]
    POSTGRES_PASSWORD= os.environ["POSTGRES_PASSWORD"]

    DATABASE_URL=f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5433/{POSTGRES_DB}"

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    treinar_incremental(engine)