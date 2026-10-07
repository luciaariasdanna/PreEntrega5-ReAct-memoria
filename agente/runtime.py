"""Compilación del grafo con persistencia (Fase 3) y helper para hacer preguntas."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from agente.config import Configuracion
from agente.grafo import EntradaAgente, EstadoAgente, construir_grafo
from agente.traza import Turno, construir_turno

type Agente = CompiledStateGraph[EstadoAgente, None, EntradaAgente, EstadoAgente]


@asynccontextmanager
async def abrir_agente(cfg: Configuracion) -> AsyncIterator[tuple[Agente, AsyncSqliteSaver]]:
    """Compila el grafo con un checkpointer SQLite en disco.

    Usamos AsyncSqliteSaver (la variante async de SqliteSaver, del mismo paquete
    langgraph-checkpoint-sqlite) porque todo el grafo corre con asyncio: el
    SqliteSaver síncrono no soporta `ainvoke`. Cada paso del grafo queda guardado
    en el archivo, indexado por `thread_id`.
    """
    # max_retries/timeout acotados: si la API está saturada (503) fallamos en
    # segundos con un error claro en vez de quedar reintentando en silencio.
    llm = ChatGoogleGenerativeAI(model=cfg.modelo, temperature=0, max_retries=3, timeout=60)
    async with AsyncSqliteSaver.from_conn_string(str(cfg.ruta_checkpoints)) as checkpointer:
        await checkpointer.setup()  # crea las tablas si el archivo es nuevo
        agente = construir_grafo(llm, cfg.max_mensajes_contexto).compile(checkpointer=checkpointer)
        yield agente, checkpointer


def config_de_ejecucion(thread_id: str, recursion_limit: int) -> RunnableConfig:
    # recursion_limit = techo de super-pasos del grafo por invocación. Cada vuelta
    # modelo -> herramientas consume 2, así que 10 permite ~4 llamadas a herramientas
    # antes de cortar: evita bucles infinitos y costos inesperados en la API.
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


async def preguntar(agente: Agente, thread_id: str, texto: str, recursion_limit: int) -> Turno:
    config = config_de_ejecucion(thread_id, recursion_limit)
    previo = await agente.aget_state(config)
    n_previos = len(previo.values.get("messages", []))

    try:
        # llamadas_a_herramientas=0: el reducer operator.add lo suma al acumulado guardado
        await agente.ainvoke(EntradaAgente(messages=[HumanMessage(texto)], llamadas_a_herramientas=0), config)
    except GraphRecursionError:
        # Red de seguridad: el nodo modelo ya cierra hacia END usando remaining_steps,
        # así que esto solo ocurriría si cambiara el diseño del grafo.
        estado = await agente.aget_state(config)
        turno = construir_turno(thread_id, estado.values["messages"][n_previos:])
        turno["respuesta"] = (
            f"[Se alcanzó recursion_limit={recursion_limit} sin respuesta final; se cortó el ciclo.]"
        )
        return turno

    estado = await agente.aget_state(config)
    return construir_turno(thread_id, estado.values["messages"][n_previos:])
