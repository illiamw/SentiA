import re
import json
from pathlib import Path
import datetime

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "classifier" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


MODEL_DIR = obter_ultima_versao()
print(f"Carregando modelo de sentimento da versão: {MODEL_DIR}")




EMOCOES = [
    "alegria",
    "tristeza",
    "medo",
    "raiva",
    "surpresa",
    "nojo",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(device)

model.eval()

# Limite de tokens suportado pelo modelo, incluindo tokens especiais.
MAX_TOKENS = min(
    512,
    model.config.max_position_embeddings,
)

# Reserva espaço para [CLS], [SEP] e outros tokens especiais.
TOKENS_ESPECIAIS = tokenizer.num_special_tokens_to_add(pair=False)
TAMANHO_TRECHO = MAX_TOKENS - TOKENS_ESPECIAIS



def obter_rotulos() -> list[str]:
    """Obtém os nomes das classes na ordem das saídas do modelo."""
    id2label = model.config.id2label
    rotulos = []

    for i in range(model.config.num_labels):
        rotulo = id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))
        rotulos.append(str(rotulo).lower().strip())

    # Se o modelo tiver sido salvo com rótulos genéricos,
    # assume a ordem abaixo. Confirme se corresponde ao treinamento.
    if all(r.startswith("label_") for r in rotulos):
        if len(rotulos) != len(EMOCOES):
            raise ValueError(
                "O número de classes do modelo não corresponde "
                "às seis emoções esperadas."
            )
        return EMOCOES.copy()

    return rotulos


ROTULOS = obter_rotulos()
print("Rótulos do modelo:", ROTULOS)


def dividir_em_trechos(
    texto: str,
    tamanho: int = TAMANHO_TRECHO,
    sobreposicao: int = 40,
) -> list[list[int]]:
    """
    Divide o texto em trechos de IDs de tokens.

    A sobreposição ajuda a preservar contexto entre trechos.
    """
    if tamanho <= 0:
        raise ValueError("O tamanho do trecho deve ser positivo.")

    if sobreposicao < 0 or sobreposicao >= tamanho:
        raise ValueError(
            "A sobreposição deve ser >= 0 e menor que o tamanho."
        )

    token_ids = tokenizer.encode(
        texto,
        add_special_tokens=False,
        truncation=False,
    )

    if not token_ids:
        return []

    passo = tamanho - sobreposicao

    return [
        token_ids[inicio:inicio + tamanho]
        for inicio in range(0, len(token_ids), passo)
    ]


def analisar_sentimento(texto: str) -> dict:
    """Classifica um texto curto, respeitando o limite do modelo."""
    if not texto or not texto.strip():
        raise ValueError("O texto não pode ser vazio.")

    trechos = dividir_em_trechos(texto, sobreposicao=0)

    return classificar_trechos(trechos)[0]



def classificar_trechos(
    trechos: list[list[int]],
    batch_size: int = 8,
) -> list[dict]:
    """Classifica trechos em lotes usando o tokenizer diretamente."""
    resultados = []

    for inicio in range(0, len(trechos), batch_size):
        lote_ids = trechos[inicio:inicio + batch_size]

        # Converte os IDs em texto e deixa o tokenizer cuidar
        # dos tokens especiais, padding e máscaras de atenção.
        textos = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in lote_ids
        ]
        print(f"Classificando lote de {len(textos)} trechos...")

        inputs = tokenizer(
            textos,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )

        inputs = {
            chave: tensor.to(device)
            for chave, tensor in inputs.items()
        }

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilidades = torch.softmax(
                logits, dim=-1
            ).cpu()

        for texto, probs in zip(textos, probabilidades):
            indice = int(probs.argmax().item())

            resultados.append({
                "texto": texto,
                "sentimento": ROTULOS[indice],
                "probabilidades": {
                    ROTULOS[i]: round(float(p), 4)
                    for i, p in enumerate(probs)
                },
            })

    return resultados




def analisar_transcricao(
    caminho_arquivo: str | Path,
    batch_size: int = 8,
    sobreposicao: int = 40,
) -> dict:
    """
    Analisa uma transcrição TXT inteira.

    Retorna:
      - sentimento predominante;
      - probabilidades médias por emoção;
      - distribuição percentual;
      - resultados de cada trecho.
    """
    caminho = Path(caminho_arquivo)
    print(f"Analisando transcrição: {caminho}")

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    texto = caminho.read_text(encoding="utf-8-sig")
    print(f"Tamanho do texto: {len(texto)} caracteres")

    if not texto.strip():
        raise ValueError("O arquivo de transcrição está vazio.")

    trechos = dividir_em_trechos(
        texto,
        sobreposicao=sobreposicao,
    )
    print(f"Total de trechos gerados: {len(trechos)}")
    print(f"Tamanho de cada trecho (tokens): {TAMANHO_TRECHO}")

    if not trechos:
        raise ValueError("Não foi possível gerar trechos.")

    resultados = classificar_trechos(
        trechos,
        batch_size=batch_size,
    )
    print(f"Total de trechos classificados: {len(resultados)}")
    print(f"Exemplo de resultado do primeiro trecho: {resultados[0]}")

    # Média das probabilidades previstas em cada trecho.
    medias = {
        emocao: sum(
            r["probabilidades"][emocao]
            for r in resultados
        ) / len(resultados)
        for emocao in ROTULOS
    }

    # Percentual de trechos cuja classe mais provável foi cada emoção.
    contagens = pd.Series(
        [r["sentimento"] for r in resultados]
    ).value_counts()

    distribuicao = {
        emocao: round(
            100 * contagens.get(emocao, 0) / len(resultados),
            2,
        )
        for emocao in ROTULOS
    }

    sentimento_geral = max(medias, key=medias.get)

    return {
        "arquivo": caminho.name,
        "total_trechos": len(resultados),
        "sentimento_geral": sentimento_geral,
        "probabilidades_medias": {
            k: round(v, 4) for k, v in medias.items()
        },
        "distribuicao_percentual": distribuicao,
        "trechos": resultados,
    }


def preditor(PATH_TRANSCRICAO: str | Path, PATH_SAIDA: str | Path = None) -> any:
    """
    Retorna uma função que pode ser usada como preditor de sentimento
    em um pipeline de processamento de texto.
    """
    resultado = analisar_transcricao(
            caminho_arquivo=PATH_TRANSCRICAO,
            batch_size=8,
            sobreposicao=40,
        )
    
    print(json.dumps(
        resultado,
        ensure_ascii=False,
        indent=2,
    ))

    # Salva os resultados para consulta posterior.
    caminho_saida = (
        PATH_SAIDA
    )

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    caminho_saida.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nResultado salvo em: {caminho_saida}")


if __name__ == "__main__":
    
    NOME_PACIENTE = "paciente_002"
    DATA_HORA_TRANSCRICAO = "2024-06-15_14-30-00"

    preditor(
        PATH_TRANSCRICAO=PROJECT_ROOT  / "source" / f"transcricao_{NOME_PACIENTE}_{DATA_HORA_TRANSCRICAO}.txt",
        PATH_SAIDA=PROJECT_ROOT  / "source" / f"resultado_transcricao_{NOME_PACIENTE}_{DATA_HORA_TRANSCRICAO}.json"
    )