"""Contrato de herramientas (Fase 1).

El LLM decide qué herramienta usar leyendo SOLO el nombre, los parámetros
tipados y el docstring de cada función. Por eso los docstrings dicen
explícitamente cuándo usarla, qué devuelve y qué hacer si falla.

Las herramientas son `async` y simulan la latencia de una base de datos real
con `asyncio.sleep`, para que todo el flujo sea asíncrono de punta a punta.
"""

import asyncio
import difflib
import json

from langchain_core.tools import BaseTool, tool

from agente.datos import CLIENTES_DB, PEDIDOS_DB

LATENCIA_DB_SEGUNDOS = 0.2


def _a_json(datos: object) -> str:
    return json.dumps(datos, ensure_ascii=False)


@tool
async def buscar_cliente_por_nombre(nombre: str) -> str:
    """Obtiene el cliente_id (identificador numérico interno) de un cliente a partir de su nombre completo.

    Usá esta herramienta SIEMPRE que el usuario mencione a un cliente por su nombre
    y todavía no conozcas su cliente_id. Es el paso previo obligatorio para consultar
    sus pedidos con `buscar_pedidos` u `obtener_ultimo_pedido`.

    Args:
        nombre: nombre y apellido del cliente, por ejemplo "Ana García".

    Returns:
        JSON con `cliente_id` si hay coincidencia exacta. Si no la hay, devuelve un
        JSON con `error` y, cuando existan, `sugerencias` con nombres parecidos:
        si alguna sugerencia es claramente el cliente que pidió el usuario (por
        ejemplo, una diferencia de tildes), volvé a llamar a esta herramienta con
        ese nombre exacto; si no hay sugerencias, pedile al usuario que aclare el nombre.
    """
    await asyncio.sleep(LATENCIA_DB_SEGUNDOS)
    indice = {n.casefold(): n for n in CLIENTES_DB}
    clave = " ".join(nombre.split()).casefold()

    if clave in indice:
        nombre_registrado = indice[clave]
        return _a_json({"nombre": nombre_registrado, "cliente_id": CLIENTES_DB[nombre_registrado]})

    parecidos = difflib.get_close_matches(clave, list(indice), n=3, cutoff=0.6)
    return _a_json(
        {
            "error": f"No existe ningún cliente registrado con el nombre '{nombre}'.",
            "sugerencias": [indice[p] for p in parecidos],
        }
    )


@tool
async def buscar_pedidos(cliente_id: int) -> str:
    """Devuelve la CANTIDAD de pedidos y el MONTO TOTAL gastado (en pesos) por un cliente.

    Requiere el cliente_id numérico, NO el nombre. Si solo tenés el nombre,
    primero usá `buscar_cliente_por_nombre` para obtener el cliente_id.
    Usala para preguntas como "¿cuántos pedidos hizo?" o "¿cuánto gastó en total?".

    Args:
        cliente_id: identificador numérico del cliente, por ejemplo 102.

    Returns:
        JSON con `cliente_id`, `pedidos` (cantidad) y `total` (suma de montos), o un
        JSON con `error` si el cliente_id no existe.
    """
    await asyncio.sleep(LATENCIA_DB_SEGUNDOS)
    if cliente_id not in PEDIDOS_DB:
        return _a_json({"error": f"No existe ningún cliente con cliente_id={cliente_id}."})
    pedidos = PEDIDOS_DB[cliente_id]
    return _a_json(
        {"cliente_id": cliente_id, "pedidos": len(pedidos), "total": sum(p["monto"] for p in pedidos)}
    )


@tool
async def obtener_ultimo_pedido(cliente_id: int) -> str:
    """Devuelve el detalle del pedido MÁS RECIENTE de un cliente: número, fecha, monto y estado.

    Requiere el cliente_id numérico, NO el nombre. Si ya obtuviste el cliente_id
    antes en esta conversación, reutilizalo en lugar de volver a buscarlo.
    Usala para preguntas como "¿cuál fue su último pedido?", "¿cuándo compró por
    última vez?" o "¿en qué estado está su pedido más reciente?".

    Args:
        cliente_id: identificador numérico del cliente, por ejemplo 102.

    Returns:
        JSON con `pedido_id`, `fecha` (AAAA-MM-DD), `monto` y `estado`, o un JSON con
        `error` si el cliente_id no existe o no tiene pedidos.
    """
    await asyncio.sleep(LATENCIA_DB_SEGUNDOS)
    pedidos = PEDIDOS_DB.get(cliente_id)
    if not pedidos:
        return _a_json({"error": f"No hay pedidos registrados para cliente_id={cliente_id}."})
    ultimo = max(pedidos, key=lambda p: p["fecha"])
    return _a_json({"cliente_id": cliente_id, **ultimo})


HERRAMIENTAS: list[BaseTool] = [buscar_cliente_por_nombre, buscar_pedidos, obtener_ultimo_pedido]
