
from pathlib import Path

from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def transcrever_audio(
    AUDIO: str | Path,
    PATH_TRANSCRICAO: str | Path,
) -> str:
    """
    Transcreve um arquivo de áudio para texto em português.

    Args:
        AUDIO: Caminho do arquivo de áudio.
        PATH_TRANSCRICAO: Caminho do arquivo TXT de saída.

    Returns:
        Texto completo da transcrição.
    """
    caminho_audio = Path(AUDIO)
    caminho_saida = Path(PATH_TRANSCRICAO)

    if not caminho_audio.exists():
        raise FileNotFoundError(
            f"Arquivo de áudio não encontrado: {caminho_audio}"
        )

    caminho_saida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Modelo leve, executado localmente pela CPU.
    modelo = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )

    segmentos, info = modelo.transcribe(
        str(caminho_audio),
        language="pt",
        vad_filter=True,
    )

    texto = " ".join(
        segmento.text.strip()
        for segmento in segmentos
    ).strip()

    caminho_saida.write_text(
        texto,
        encoding="utf-8",
    )

    print(f"Idioma detectado: {info.language}")
    print(f"Transcrição salva em: {caminho_saida}")

    return texto



if __name__ == "__main__":

    NOME_PACIENTE = "paciente_002"
    DATA_HORA_TRANSCRICAO = "2024-06-15_14-30-00"
    AUDIO = PROJECT_ROOT / "source" / f"audio_{NOME_PACIENTE}_{DATA_HORA_TRANSCRICAO}.mp4"
    PATH_TRANSCRICAO = PROJECT_ROOT /  "source" / f"transcricao_{NOME_PACIENTE}_{DATA_HORA_TRANSCRICAO}.txt"

    transcrever_audio(AUDIO, PATH_TRANSCRICAO)