r"""Estado y grafo del agente ReAct (Fase 2).

    START -> modelo --(tools_condition: hay tool_calls)--> herramientas
               ^   \                                           |
               |    `--(tools_condition: sin tool_calls)--> END |
               `-----------------------------------------------'

El LLM decide en cada vuelta si llama a una herramienta o si ya puede responder:
no hay ningún if/else nuestro que elija la herramienta.

Todo camino termina en END: o el modelo responde sin tool_calls, o se le acaban
los pasos (`remaining_steps`) y el nodo fuerza una respuesta final sin tool_calls.
Así el grafo nunca se corta a la mitad por `GraphRecursionError`.
"""

import operator
from collections.abc import Sequence
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, tools_condition

from agente.herramientas import HERRAMIENTAS

PROMPT_SISTEMA = """Sos un asistente de atención al cliente con acceso a la base de datos de pedidos.

Reglas:
- Nunca inventes datos: toda cifra, fecha o estado tiene que salir de una herramienta.
- Podés encadenar varias herramientas para responder una sola pregunta.
- Si una herramienta devuelve "error", leé el mensaje: si trae una sugerencia que
  claramente corresponde a lo que pidió el usuario, reintentá con ella; si no,
  explicale al usuario qué pasó y pedile que aclare el dato.
- Usá el contexto de la conversación: si ya conocés el cliente_id de alguien, reutilizalo.
- Respondé en español, breve, con montos en formato $14.500."""


class EntradaAgente(MessagesState):
    """Lo que se le pasa al grafo en cada invocación.

    Hereda `messages` de MessagesState (reducer `add_messages`: agrega mensajes
    nuevos en lugar de reemplazar la lista) y suma un contador propio con el
    reducer `operator.add`, que acumula entre vueltas y entre turnos del thread."""

    llamadas_a_herramientas: Annotated[int, operator.add]


class EstadoAgente(EntradaAgente):
    """Estado interno completo. `remaining_steps` es un valor gestionado por
    LangGraph (no se guarda ni se pasa en la entrada): indica cuántos pasos
    quedan antes de llegar a `recursion_limit`."""

    remaining_steps: RemainingSteps


# Una vuelta más necesita 2 pasos: ejecutar la herramienta y volver al modelo.
PASOS_POR_VUELTA = 2


def recortar_historial(mensajes: Sequence[BaseMessage], max_mensajes: int) -> list[BaseMessage]:
    """Evita el "estado sucio": el historial completo queda guardado en el
    checkpointer, pero al LLM solo le mandamos los últimos `max_mensajes`.
    `start_on="human"` garantiza que el recorte no deje un ToolMessage huérfano
    (sin el AIMessage que lo pidió), algo que la API del modelo rechazaría."""
    return trim_messages(
        mensajes,
        max_tokens=max_mensajes,
        token_counter=len,  # cuenta mensajes, no tokens
        strategy="last",
        start_on="human",
    )


def construir_grafo(llm: BaseChatModel, max_mensajes_contexto: int) -> StateGraph[EstadoAgente, None, EntradaAgente, EstadoAgente]:
    llm_con_herramientas = llm.bind_tools(HERRAMIENTAS)

    async def nodo_modelo(state: EstadoAgente) -> dict[str, Any]:
        historial = recortar_historial(state["messages"], max_mensajes_contexto)
        respuesta = await llm_con_herramientas.ainvoke([SystemMessage(PROMPT_SISTEMA), *historial])

        if getattr(respuesta, "tool_calls", None) and state["remaining_steps"] < PASOS_POR_VUELTA:
            # No hay pasos para otra vuelta: respondemos sin tool_calls para que
            # tools_condition nos lleve a END en lugar de chocar contra el límite.
            respuesta = AIMessage(
                "No pude completar la consulta dentro del límite de pasos permitido. "
                "¿Podés reformular la pregunta o darme más detalles?"
            )
        return {
            "messages": [respuesta],
            "llamadas_a_herramientas": len(getattr(respuesta, "tool_calls", [])),
        }

    grafo = StateGraph(EstadoAgente, input_schema=EntradaAgente)
    grafo.add_node("modelo", nodo_modelo)
    # handle_tool_errors=True: si una herramienta lanza una excepción (p. ej. el LLM
    # manda un argumento con tipo inválido), el error vuelve al modelo como
    # ToolMessage en lugar de cortar la ejecución; así puede corregirse y reintentar.
    grafo.add_node("herramientas", ToolNode(HERRAMIENTAS, handle_tool_errors=True))

    grafo.add_edge(START, "modelo")
    grafo.add_conditional_edges("modelo", tools_condition, {"tools": "herramientas", END: END})
    grafo.add_edge("herramientas", "modelo")  # el ciclo ReAct
    return grafo
