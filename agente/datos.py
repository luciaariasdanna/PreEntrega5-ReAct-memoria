"""Base de datos simulada (en memoria).

Hay dos "tablas" separadas a propósito: el usuario pregunta por el *nombre*
del cliente, pero los pedidos se indexan por *cliente_id*. Eso obliga al agente
a razonar en varios pasos (nombre -> id -> pedidos) sin que le digamos el orden.
"""

from typing import TypedDict


class Pedido(TypedDict):
    pedido_id: int
    fecha: str  # ISO 8601 (YYYY-MM-DD)
    monto: int  # en pesos
    estado: str


# Tabla "clientes": nombre completo -> cliente_id
CLIENTES_DB: dict[str, int] = {
    "Ana García": 102,
    "Juan Pérez": 205,
    "María López": 310,
}

# Tabla "pedidos": cliente_id -> lista de pedidos (ordenados por fecha)
PEDIDOS_DB: dict[int, list[Pedido]] = {
    102: [
        {"pedido_id": 9001, "fecha": "2026-06-14", "monto": 4200, "estado": "entregado"},
        {"pedido_id": 9017, "fecha": "2026-08-02", "monto": 6100, "estado": "entregado"},
        {"pedido_id": 9045, "fecha": "2026-09-21", "monto": 4200, "estado": "en camino"},
    ],
    205: [
        {"pedido_id": 9010, "fecha": "2026-07-09", "monto": 3200, "estado": "entregado"},
    ],
    310: [
        {"pedido_id": 8950, "fecha": "2026-03-30", "monto": 5000, "estado": "entregado"},
        {"pedido_id": 8988, "fecha": "2026-05-11", "monto": 7300, "estado": "entregado"},
        {"pedido_id": 9002, "fecha": "2026-06-18", "monto": 2500, "estado": "devuelto"},
        {"pedido_id": 9033, "fecha": "2026-08-27", "monto": 9000, "estado": "entregado"},
        {"pedido_id": 9050, "fecha": "2026-09-30", "monto": 4000, "estado": "preparando"},
    ],
}
