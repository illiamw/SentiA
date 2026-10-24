
from pathlib import Path

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

EMOCOES = [
    "alegria", "tristeza", "medo",
    "raiva", "surpresa", "nojo"
]



def gerar_minuta(
    caminho_transcricao: str | Path,
    caminho_saida: str | Path | None = None,
) -> str:
    """
    Gera uma minuta da sessão utilizando o Ollama local.

    A minuta contém:
    - Resumo da sessão
    - Principais pontos da sessão
    - Próximos passos para acompanhamento do paciente

    Args:
        caminho_transcricao: caminho do arquivo TXT da transcrição.
        caminho_saida: caminho opcional para salvar a minuta em TXT.

    Returns:
        Texto completo da minuta gerada pelo modelo.
    """

    caminho_transcricao = Path(caminho_transcricao).resolve()

    if not caminho_transcricao.is_file():
        raise FileNotFoundError(
            f"Transcrição não encontrada: {caminho_transcricao}"
        )

    transcricao = caminho_transcricao.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not transcricao:
        raise ValueError("O arquivo de transcrição está vazio.")

    prompt_resumo= f"""


Gere o resumo da transcrição em um paragrafo

TRANSCRIÇÃO DA SESSÃO:
<transcricao>
{transcricao}
</transcricao>
"""
    prompt_topicos= f"""
    
    
    Gere a lista de tópicos simples da sessão
    
    TRANSCRIÇÃO DA SESSÃO:
    <transcricao>
    {transcricao}
    </transcricao>
    """

    prompt_proximos= f"""
        
        
        Gere a lista de próximos tópicos para futura sessão
        
        TRANSCRIÇÃO DA SESSÃO:
        <transcricao>
        {transcricao}
        </transcricao>
        """

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

    payload_resumo = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você auxilia na organização de documentação clínica. "
                    "Seja fiel às evidências da transcrição e não invente "
                    "informações."
                ),
            },
            {
                "role": "user",
                "content": prompt_resumo,
            },
        ],
        "stream": False,
            "options": {
                "temperature": 0.2,
                "repeat_penalty": 1.15,
                "repeat_last_n": 128,
                "num_predict": 1024,
            },
    }

    payload_topicos = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Você auxilia na organização de documentação clínica. "
                        "Seja fiel às evidências da transcrição e não invente "
                        "informações."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt_topicos,
                },
            ],
            "stream": False,
                "options": {
                    "temperature": 0.2,
                    "repeat_penalty": 1.15,
                    "repeat_last_n": 128,
                    "num_predict": 1024,
                },
        }

    payload_proximos = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Você auxilia na organização de documentação clínica. "
                        "Seja fiel às evidências da transcrição e não invente "
                        "informações."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt_proximos,
                },
            ],
            "stream": False,
                "options": {
                    "temperature": 0.2,
                    "repeat_penalty": 1.15,
                    "repeat_last_n": 128,
                    "num_predict": 1024,
                },
        }


    try:
        with httpx.Client(timeout=300.0) as client:
            resposta_resumo = client.post(url, json=payload_resumo)
            resposta_resumo.raise_for_status()
            resultado_resumo = resposta_resumo.json()

        with httpx.Client(timeout=300.0) as client:
            resposta_topicos  = client.post(url, json=payload_topicos)
            resposta_topicos.raise_for_status()
            resultado_topicos = resposta_topicos.json()

        with httpx.Client(timeout=300.0) as client:
            resposta_proximos = client.post(url, json=payload_proximos)
            resposta_proximos.raise_for_status()
            resultado_proximos = resposta_proximos.json()

    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Não foi possível conectar ao Ollama em {url}. "
            "Verifique se o serviço está em execução e se "
            "OLLAMA_BASE_URL está configurada corretamente."
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama retornou HTTP {exc.response.status_code}: "
            f"{exc.response.text}"
        ) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Erro de comunicação com o Ollama: {exc}"
        ) from exc

    minuta_resumo = resultado_resumo.get("message", {}).get("content", "").strip()

    if not minuta_resumo:
        raise RuntimeError("O Ollama retornou uma minuta vazia.")

    minuta_topicos = resultado_topicos.get("message", {}).get("content", "").strip()
    
    if not minuta_topicos:
        raise RuntimeError("O Ollama retornou uma minuta vazia.")


    minuta_proximos = resultado_proximos.get("message", {}).get("content", "").strip()
    
    if not minuta_proximos:
        raise RuntimeError("O Ollama retornou uma minuta vazia.")
    

    if caminho_saida is not None:
        caminho_saida = Path(caminho_saida).resolve()
        caminho_saida.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        caminho_saida.write_text(
            minuta_resumo + "\n" + minuta_topicos + "\n" + minuta_proximos,
            encoding="utf-8"
        )

    return minuta_resumo , minuta_topicos , minuta_proximos


def gerar_relatorio_emocoes(
    caminho_json: str | Path,
    caminho_pdf: str | Path | None = None,
    caminho_transcricao: str | Path | None = None,
    paciente: str | None = None,
) -> Path:
    """Gera um PDF com gráfico radar e transcrição integral."""

    caminho_json = Path(caminho_json).resolve()

    with caminho_json.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    sentimento = str(
        dados.get("sentimento_geral", "não identificado")
    ).strip()

    probabilidades = dados["probabilidades_medias"]

    valores = []
    for emocao in EMOCOES:
        valor = float(probabilidades[emocao])

        if not math.isfinite(valor) or not 0 <= valor <= 1:
            raise ValueError(
                f"Probabilidade inválida para {emocao}: {valor}"
            )

        valores.append(valor)

    # Localiza a transcrição.
    if caminho_transcricao is not None:
        txt_path = Path(caminho_transcricao).resolve()
    else:
        txt_path = Path(dados["arquivo"])

        if not txt_path.is_absolute():
            txt_path = caminho_json.parent / txt_path

        txt_path = txt_path.resolve()

    minuta_resumo , minuta_topicos , minuta_proximos = gerar_minuta(
        caminho_transcricao=txt_path
    )

    if not txt_path.is_file():
        raise FileNotFoundError(
            f"Transcrição não encontrada: {txt_path}"
        )

    # Define o caminho de saída.
    if caminho_pdf is None:
        caminho_pdf = caminho_json.with_name(
            f"{caminho_json.stem}_relatorio.pdf"
        )
    else:
        caminho_pdf = Path(caminho_pdf).resolve()

    caminho_pdf.parent.mkdir(parents=True, exist_ok=True)

    caminho_grafico = caminho_pdf.with_name(
        f"{caminho_pdf.stem}_grafico.png"
    )

    # Gráfico radar com seis eixos.
    angulos = np.linspace(
        0, 2 * np.pi, len(EMOCOES), endpoint=False
    )

    angulos_fechados = np.append(angulos, angulos[0])
    valores_fechados = valores + [valores[0]]

    fig, ax = plt.subplots(
        figsize=(8, 7),
        subplot_kw={"polar": True}
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angulos)
    ax.set_xticklabels(
        [e.capitalize() for e in EMOCOES],
        fontsize=11
    )

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["20%", "40%", "60%", "80%", "100%"],
        fontsize=8
    )

    ax.plot(
        angulos_fechados,
        valores_fechados,
        linewidth=2,
        marker="o"
    )
    ax.fill(angulos_fechados, valores_fechados, alpha=0.2)
    ax.set_title(
        "Probabilidades médias por emoção",
        pad=25,
        fontsize=14
    )
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        caminho_grafico,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Configuração do PDF.
    estilos = getSampleStyleSheet()

    titulo = ParagraphStyle(
        "TituloRelatorio",
        parent=estilos["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#243B53"),
        spaceAfter=16
    )

    estilo_transcricao = ParagraphStyle(
        "Transcricao",
        parent=estilos["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=6,
        splitLongWords=True
    )

    documento = SimpleDocTemplate(
        str(caminho_pdf),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Relatório SentIA"
    )

    elementos = [
        Paragraph("Relatório de Análise Emocional", titulo),
        Paragraph(
                    f"<b>Paciente:</b> "
                    f"{paciente if paciente else 'não informado'}",
                    estilos["Heading2"]
                ),
        Paragraph(
            f"<b>Sentimento geral:</b> {escape(sentimento.capitalize())}",
            estilos["Heading2"]
        ),
        
        Spacer(1, 0.4 * cm),
        Image(
            str(caminho_grafico),
            width=15 * cm,
            height=12.5 * cm
        ),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "Probabilidades médias",
            estilos["Heading2"]
        ),
    ]

    for emocao, valor in zip(EMOCOES, valores):
        elementos.append(
            Paragraph(
                f"<b>{emocao.capitalize()}:</b> {valor:.2%}",
                estilos["BodyText"]
            )
        )

    # Segunda página: transcrição completa.
    elementos.extend([
        PageBreak(),
        Paragraph("Minuta da sessão", titulo),
        Paragraph(
                    f"<b>Resumo:</b>",
                    estilos["Heading2"]
                ),
        Paragraph(
            f"{minuta_resumo}",
            estilos["BodyText"]
        ),
        Paragraph(
                    f"<b>Tópicos:</b>",
                    estilos["Heading2"]
                ),
        Paragraph(
            f"{minuta_topicos}",
            estilos["BodyText"]
        ),
        Paragraph(
                    f"<b>Próximos Passos:</b>",
                    estilos["Heading2"]
                ),
        Paragraph(
            f"{minuta_proximos}",
            estilos["BodyText"]
        ),
        Spacer(1, 0.3 * cm)
    ])

    # Terceira página: transcrição completa.
    elementos.extend([
        PageBreak(),
        Paragraph("Transcrição integral da sessão", titulo),
        Paragraph(
            f"<b>Arquivo:</b> {escape(txt_path.name)}",
            estilos["BodyText"]
        ),
        Spacer(1, 0.3 * cm)
    ])

    texto = txt_path.read_text(encoding="utf-8-sig")

    for linha in texto.splitlines():
        elementos.append(
            Paragraph(
                escape(linha) if linha.strip() else " ",
                estilo_transcricao
            )
        )

    documento.build(elementos)

    # Remove a imagem temporária.
    caminho_grafico.unlink(missing_ok=True)

    return caminho_pdf




if __name__ == "__main__":

    NOME_PACIENTE = "William"
    DATA_HORA_TRANSCRICAO = "20230915_123456"
    PATH_CLASSIFIRR = PROJECT_ROOT /  "source" / f"resultado_transcricao_{NOME_PACIENTE}_{DATA_HORA_TRANSCRICAO}.json"
    PATH_SUMMARIZATION = PROJECT_ROOT /  "source" / f"sumarizacao_{NOME_PACIENTE}_{DATA_HORA_TRANSCRICAO}.pdf"


    pdf = gerar_relatorio_emocoes(caminho_json=PATH_CLASSIFIRR, caminho_pdf=PATH_SUMMARIZATION, paciente=NOME_PACIENTE)
    print(f"Relatório gerado em: {pdf}")