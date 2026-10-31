
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from fastapi.responses import FileResponse
import time

from fastapi.middleware.cors import CORSMiddleware





PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
SOURCE_DIR = PROJECT_ROOT / "source"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "classifier" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VALIDACAO_CSV = (
    PROJECT_ROOT
    / "classifier"
    / "data"
    / "validacao_sentimentos.csv"
)
print(f"MODELS_DIR: {MODELS_DIR}")


from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool

import datetime
from datetime import datetime
import os


from classifier.eval import gerar_metricas_bert
from classifier.treinar_incremental import treinar_incremental 
from classifier.train import treino 
from classifier.predict import preditor
from transcritor.main import transcrever_audio
from summary.main import gerar_relatorio_emocoes


import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Literal

# Controle dos processamentos em execução
processamentos = {}



logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentia API",
    description="API do assistente Sentia",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.environ["DATABASE_URL"]
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://ollama:11434"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


engine = create_engine(DATABASE_URL, pool_pre_ping=True)




logger = logging.getLogger(__name__)

EXTENSOES_PERMITIDAS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".webm", ".mpeg"
}

MAX_AUDIO_BYTES = 100 * 1024 * 1024  # 100 MB


@app.get("/models", tags=["Modelos"])
def listar_modelos():
    """
    Lista as versões dos modelos disponíveis e retorna suas métricas,
    exceto a matriz de confusão.
    """
    try:
        modelos_dirs = sorted(
            pasta
            for pasta in MODELS_DIR.iterdir()
            if pasta.is_dir() and pasta.name.startswith("sentiment_model_v")
        )

        modelos = []
        print(f"Modelos encontrados: {len(modelos_dirs)}")

        for pasta in modelos_dirs:
            modelo_info = {
                "nome": pasta.name,
                "caminho": str(pasta.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "metricas": None,
            }

            try:
                metricas = gerar_metricas_bert(
                    pasta,
                    VALIDACAO_CSV,
                )

                # Remove a matriz de confusão, mantendo as demais métricas.
                metricas.pop("matriz_confusao", None)

                modelo_info["metricas"] = metricas

            except Exception as erro:
                modelo_info["erro_metricas"] = str(erro)
                print(
                    f"Erro ao calcular métricas de {pasta.name}: {erro}"
                )

            modelos.append(modelo_info)

        return {
            "total": len(modelos),
            "diretorio": str(MODELS_DIR),
            "modelos": modelos,
        }

    except OSError as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao listar os modelos: {erro}",
        ) from erro

    
class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    print("Root endpoint accessed, Version 0.1.4")
    return {"app": "Sentia", "status": "ok", "version": "0.1.4"}


@app.get("/health")
def health():
    print("Health endpoint accessed, Version 0.1.4")
    return {"app": "Sentia", "status": "ok", "version": "0.1.4"}


@app.get("/health/database")
def database_health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"database": "ok"}
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados indisponível",
        )


@app.get("/health/llm")
async def llm_health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{OLLAMA_BASE_URL}/api/tags"
            )
            response.raise_for_status()
        return {"ollama": "ok"}
    except (httpx.HTTPError, ValueError):
        raise HTTPException(
            status_code=503,
            detail="Serviço LLM indisponível",
        )





@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": request.message,
                        }
                    ],
                    "stream": False,
                },
            )

            response.raise_for_status()
            data = response.json()

        return {
            "model": OLLAMA_MODEL,
            "response": data["message"]["content"],
        }

    

    except httpx.HTTPStatusError as exc:
        print("Status Ollama:", exc.response.status_code)
        print("Resposta Ollama:", exc.response.text)

        raise HTTPException(
            status_code=502,
            detail={
                "ollama_status": exc.response.status_code,
                "ollama_error": exc.response.text,
            },
        ) from exc

    except httpx.TimeoutException as exc:
        logger.error("Timeout ao consultar Ollama: %s", str(exc))
        raise HTTPException(
            status_code=504,
            detail="Tempo limite excedido ao consultar o modelo",
        ) from exc

    except httpx.RequestError as exc:
        logger.error("Erro de conexão com Ollama: %s", str(exc))
        raise HTTPException(
            status_code=502,
            detail="Não foi possível conectar ao Ollama",
        ) from exc

    except (KeyError, ValueError) as exc:
        logger.exception("Resposta inesperada do Ollama")
        raise HTTPException(
            status_code=502,
            detail="Resposta inválida do modelo de linguagem",
        ) from exc


@app.get("/treino_inicial")
def treino_inicial():
    try:
        dir_output = treino()
        return {"status": "concluido!", "dir_output": dir_output}
    except Exception as e:
        logger.error("Erro ao executar o treinamento inicial: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail="Erro ao executar o treinamento inicial",
        ) from e
    
    
########################################
#Registro de sentimentos no banco de dados para treinamento incremental do modelo de linguagem
#
######################################
# Contrato de entrada da API
class SentimentoRequest(BaseModel):
    texto: str = Field(min_length=1)
    sentimento: Literal[
        "alegria",
        "tristeza",
        "medo",
        "raiva",
        "surpresa",
        "nojo",
    ]


@app.on_event("startup")
def criar_tabela_sentimentos():
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS sentimentos (
                id SERIAL PRIMARY KEY,
                texto TEXT NOT NULL,
                sentimento VARCHAR(20) NOT NULL
                    CHECK (sentimento IN (
                        'alegria',
                        'tristeza',
                        'medo',
                        'raiva',
                        'surpresa',
                        'nojo'
                    )),
                traino BOOLEAN NOT NULL DEFAULT FALSE,
                criado_em TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))


# Endpoint para registrar um texto e seu sentimento
@app.post("/sentimentos", status_code=201)
def criar_sentimento(request: SentimentoRequest):
    try:
        with engine.begin() as connection:
            result = connection.execute(
                text("""
                    INSERT INTO sentimentos (texto, sentimento, traino)
                    VALUES (:texto, :sentimento, FALSE)
                    RETURNING id, texto, sentimento, traino, criado_em
                """),
                {
                    "texto": request.texto.strip(),
                    "sentimento": request.sentimento,
                },
            )

            registro = result.mappings().one()

        ### Treinamento incremental do modelo de linguagem
        treinar_incremental(engine)

        return dict(registro)

    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao salvar o sentimento no banco de dados: {exc}",
        ) from exc



def executar_pipeline(
    nome_paciente: str,
    audio_path: Path,
    timestamp: str,
) -> dict:
    """Executa a transcrição, classificação e geração do relatório."""

    inicio_pipeline = time.perf_counter()

    def registrar_tempo(etapa: str, inicio: float) -> float:
        """Registra o tempo decorrido de uma etapa."""
        decorrido = time.perf_counter() - inicio
        acumulado = time.perf_counter() - inicio_pipeline

        print(
            f"[TEMPO] {etapa}: {decorrido:.2f}s "
            f"| Acumulado: {acumulado:.2f}s"
        )
        return decorrido

    path_transcricao = (
        SOURCE_DIR / f"transcricao_{nome_paciente}_{timestamp}.txt"
    )

    path_classificar = (
        SOURCE_DIR / f"resultado_transcricao_{nome_paciente}_{timestamp}.json"
    )

    path_pdf = (
        SOURCE_DIR / f"sumarizacao_{nome_paciente}_{timestamp}.pdf"
    )

    print(f"\n{'='*60}")
    print(f"Iniciando pipeline para: {nome_paciente}")
    print(f"Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"{'='*60}")

    tempos = {}

    # 1. Transcrição
    print("\n[1/3] Gerando transcrição...")
    inicio = time.perf_counter()

    transcrever_audio(audio_path, path_transcricao)

    tempos["transcricao"] = registrar_tempo("Transcrição", inicio)
    print(f"Transcrição salva em: {path_transcricao}")

    # 2. Classificação
    print("\n[2/3] Classificando sentimentos...")
    inicio = time.perf_counter()

    preditor(path_transcricao, path_classificar)

    tempos["classificacao"] = registrar_tempo("Classificação", inicio)
    print(f"Classificação salva em: {path_classificar}")

    # 3. Geração do relatório
    print("\n[3/3] Gerando relatório em PDF...")
    inicio = time.perf_counter()

    gerar_relatorio_emocoes(
        caminho_json=path_classificar,
        caminho_pdf=path_pdf,
        paciente=nome_paciente,
    )

    tempos["relatorio_pdf"] = registrar_tempo("Relatório PDF", inicio)
    print(f"Relatório PDF salvo em: {path_pdf}")

    # Finalização
    tempo_total = time.perf_counter() - inicio_pipeline

    tempos["total"] = tempo_total

    print(f"\n{'='*60}")
    print(f"PIPELINE CONCLUÍDO: {nome_paciente}")
    print(f"{'='*60}")

    for etapa, duracao in tempos.items():
        print(f"{etapa:<20}: {duracao:>10.2f}s")

    print(
        f"\nTérmino: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )
    print(f"{'='*60}\n")

    return {
        "data_hora": timestamp,
        "paciente": nome_paciente,
        "audio": str(audio_path),
        "transcricao": str(path_transcricao),
        "classificacao": str(path_classificar),
        "relatorio_pdf": str(path_pdf),
        "tempos_execucao": tempos,
        "status": "concluido",
    }

def executar_pipeline_background(
    nome_paciente: str,
    audio_path: Path,
    timestamp: str,
    id_sessao: str,
):
    """Executa o pipeline em segundo plano."""

    processamentos[id_sessao]["status"] = "processando"

    try:
        resultado = executar_pipeline(
            nome_paciente=nome_paciente,
            audio_path=audio_path,
            timestamp=timestamp,
        )

        resultado["trechos"] = extrair_trechos(
            resultado["classificacao"]
        )

        processamentos[id_sessao].update({
            "status": "concluido",
            "resultado": resultado,
            "finalizado_em": datetime.now().isoformat(),
        })

        logger.info(
            "Pipeline concluído para a sessão %s",
            id_sessao,
        )

    except Exception as exc:
        processamentos[id_sessao].update({
            "status": "erro",
            "erro": str(exc),
            "finalizado_em": datetime.now().isoformat(),
        })

        logger.exception(
            "Erro durante o processamento da sessão %s",
            id_sessao,
        )


def extrair_trechos(caminho_json: str | Path) -> dict:
    """
    Lê um JSON e retorna somente a chave 'trechos'.

    Args:
        caminho_json: Caminho do arquivo JSON.

    Returns:
        Dicionário contendo apenas os trechos.
    """
    caminho_json = Path(caminho_json)

    with caminho_json.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    return {
        "trechos": dados.get("trechos", [])
    }



@app.post("/relatorio/{nome_paciente}", status_code=202)
async def gerar_relatorio(
    nome_paciente: str,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
):
    """
    Recebe um áudio, salva em source/ e agenda
    o processamento do SentiA em segundo plano.
    Retorna imediatamente o identificador da sessão.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_pdf = f"sumarizacao_{nome_paciente}_{timestamp}.pdf"
    id_sessao = f"{nome_paciente}_{timestamp}"

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", nome_paciente):
        raise HTTPException(
            status_code=400,
            detail="Nome do paciente inválido.",
        )

    extensao = Path(audio.filename or "").suffix.lower()

    if extensao not in EXTENSOES_PERMITIDAS:
        raise HTTPException(
            status_code=400,
            detail={
                "erro": "Formato de áudio não permitido.",
                "formatos_aceitos": sorted(EXTENSOES_PERMITIDAS),
            },
        )

    audio_path = (
        SOURCE_DIR
        / f"audio_{nome_paciente}_{timestamp}{extensao}"
    )

    tamanho = 0

    try:
        # Salva o áudio antes de agendar o processamento.
        with audio_path.open("wb") as destino:

            while True:
                bloco = await audio.read(1024 * 1024)

                if not bloco:
                    break

                tamanho += len(bloco)

                if tamanho > MAX_AUDIO_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="O áudio excede o limite de 100 MB.",
                    )

                destino.write(bloco)

        if tamanho == 0:
            raise HTTPException(
                status_code=400,
                detail="O arquivo de áudio está vazio.",
            )

        # Registra a sessão.
        processamentos[id_sessao] = {
            "id_sessao": id_sessao,
            "paciente": nome_paciente,
            "timestamp": timestamp,
            "status": "aguardando",
            "criado_em": datetime.now().isoformat(),
            "resultado": None,
            "erro": None,
        }

        # Agenda a execução em segundo plano.
        background_tasks.add_task(
            executar_pipeline_background,
            nome_paciente,
            audio_path,
            timestamp,
            id_sessao,
        )

        # Retorna imediatamente.
        return {
            "mensagem": "Processamento iniciado com sucesso.",
            "id_sessao": id_sessao,
            "paciente": nome_paciente,
            "status": "aguardando",
            "audio": str(audio_path),
            "arquivo_pdf": nome_pdf,
            "endpoint_status": f"/relatorio/{id_sessao}/status",
        }

    except HTTPException:
        audio_path.unlink(missing_ok=True)
        raise

    except Exception as exc:
        audio_path.unlink(missing_ok=True)

        logger.exception("Erro ao iniciar processamento do SentiA.")

        raise HTTPException(
            status_code=500,
            detail="Não foi possível iniciar o processamento.",
        ) from exc

    finally:
        await audio.close()



@app.get("/relatorio/{id_sessao}/status")
def consultar_status_relatorio(id_sessao: str):
    """
    Consulta o status do processamento, verifica os arquivos
    gerados e retorna a quantidade de etapas concluídas.
    """

    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,100}",
        id_sessao,
    ):
        raise HTTPException(
            status_code=400,
            detail="Identificador de sessão inválido.",
        )

    if id_sessao not in processamentos:
        raise HTTPException(
            status_code=404,
            detail="Sessão não encontrada.",
        )

    dados = processamentos[id_sessao]

    nome_paciente = dados["paciente"]
    timestamp = dados["timestamp"]

    # Caminhos dos arquivos esperados.
    path_transcricao = (
        SOURCE_DIR
        / f"transcricao_{nome_paciente}_{timestamp}.txt"
    )

    path_classificar = (
        SOURCE_DIR
        / f"resultado_transcricao_{nome_paciente}_{timestamp}.json"
    )

    path_pdf = (
        SOURCE_DIR
        / f"sumarizacao_{nome_paciente}_{timestamp}.pdf"
    )

    arquivos = {
        "transcricao": path_transcricao,
        "classificacao": path_classificar,
        "relatorio_pdf": path_pdf,
    }

    verificacao = {
        nome: {
            "criado": caminho.is_file(),
            "caminho": str(caminho),
            "tamanho_bytes": (
                caminho.stat().st_size
                if caminho.is_file()
                else 0
            ),
        }
        for nome, caminho in arquivos.items()
    }

    # Quantidade de etapas concluídas.
    etapas_concluidas = sum(
        1 for arquivo in verificacao.values()
        if arquivo["criado"]
    )

    total_etapas = len(arquivos)

    todos_criados = etapas_concluidas == total_etapas

    status = dados["status"]

    return {
        "id_sessao": id_sessao,
        "paciente": nome_paciente,
        "status": status,

        "progresso": {
            "etapas_concluidas": etapas_concluidas,
            "total_etapas": total_etapas,
            "percentual": round(
                (etapas_concluidas / total_etapas) * 100,
                2,
            ),
        },

        "arquivos_criados": todos_criados,
        "arquivos": verificacao,
        "erro": dados.get("erro"),
        "criado_em": dados["criado_em"],
        "finalizado_em": dados.get("finalizado_em"),
        "resultado": (
            dados.get("resultado")
            if status == "concluido"
            else None
        ),
    }




@app.get("/relatorios/{id_sessao}")
def baixar_relatorio(id_sessao: str):
    pasta_relatorios = Path("/app/source").resolve()
    arquivo = (pasta_relatorios / f"sumarizacao_{id_sessao}.pdf").resolve()

    # Impede acesso a arquivos fora da pasta de relatórios
    if arquivo.parent != pasta_relatorios:
        raise HTTPException(
            status_code=400,
            detail="Nome de arquivo inválido",
        )

    if not arquivo.is_file() or arquivo.suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=404,
            detail="Relatório PDF não encontrado",
        )

    return FileResponse(
        path=arquivo,
        media_type="application/pdf",
        filename=arquivo.name,
    )




if __name__ == "__main__":
    treino_inicial()
    gerar_relatorio("William", SOURCE_DIR / "audio_William_20230915_123456.mp3")
