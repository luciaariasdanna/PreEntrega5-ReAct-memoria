"""Serialización de la traza ReAct a JSON y a un log legible."""

from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


class Turno(TypedDict):
    thread_id: str
    pregunta: str
    respuesta: str
    llamadas_a_herramientas: int
    pasos: list[dict[str, Any]]


def extraer_texto(mensaje: BaseMessage) -> str:
    """Gemini a veces devuelve `content` como lista de bloques en lugar de str."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    return "".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in contenido
    )


def serializar_mensajes(mensajes: list[BaseMessage]) -> list[dict[str, Any]]:
    pasos: list[dict[str, Any]] = []
    for m in mensajes:
        paso: dict[str, Any] = {"tipo": type(m).__name__, "contenido": extraer_texto(m)}
        if isinstance(m, AIMessage) and m.tool_calls:
            paso["tool_calls"] = [{"nombre": tc["name"], "argumentos": tc["args"]} for tc in m.tool_calls]
        if isinstance(m, ToolMessage):
            paso["herramienta"] = m.name
            paso["status"] = m.status
        pasos.append(paso)
    return pasos


def construir_turno(thread_id: str, mensajes_nuevos: list[BaseMessage]) -> Turno:
    pregunta = next((extraer_texto(m) for m in mensajes_nuevos if isinstance(m, HumanMessage)), "")
    return Turno(
        thread_id=thread_id,
        pregunta=pregunta,
        respuesta=extraer_texto(mensajes_nuevos[-1]),
        llamadas_a_herramientas=sum(isinstance(m, ToolMessage) for m in mensajes_nuevos),
        pasos=serializar_mensajes(mensajes_nuevos),
    )


def turno_a_log(turno: Turno) -> str:
    lineas = [f"[thread_id={turno['thread_id']}]"]
    for paso in turno["pasos"]:
        match paso["tipo"]:
            case "HumanMessage":
                lineas.append(f'Usuario: "{paso["contenido"]}"')
            case "AIMessage" if "tool_calls" in paso:
                if paso["contenido"]:
                    lineas.append(f"  -> El agente razona: {paso['contenido']}")
                for tc in paso["tool_calls"]:
                    args = ", ".join(f"{k}={v!r}" for k, v in tc["argumentos"].items())
                    lineas.append(f"  -> El agente decide usar la herramienta: {tc['nombre']}({args})")
            case "ToolMessage":
                lineas.append(f"  -> La herramienta devuelve: {paso['contenido']}")
            case "AIMessage":
                lineas.append("  -> El agente razona: no necesita más herramientas -> responde.")
                lineas.append(f'Respuesta: "{paso["contenido"]}"')
    if not turno["pasos"] or "tool_calls" in turno["pasos"][-1] or turno["pasos"][-1]["tipo"] != "AIMessage":
        lineas.append(f"Respuesta: {turno['respuesta']}")  # p. ej. corte por recursion_limit
    lineas.append(f"(herramientas invocadas en este turno: {turno['llamadas_a_herramientas']})")
    return "\n".join(lineas)
