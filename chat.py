"""Chat interactivo con memoria persistente.

    python chat.py --thread-id mi-sesion

Cerrá el programa (escribí "salir") y volvé a abrirlo con el mismo --thread-id:
el agente recuerda la conversación porque se guarda en checkpoints.sqlite.
"""

import argparse
import asyncio
import sys

from agente.config import cargar_configuracion
from agente.runtime import abrir_agente, config_de_ejecucion, preguntar
from agente.traza import extraer_texto, turno_a_log


async def main(thread_id: str, mostrar_traza: bool) -> None:
    cfg = cargar_configuracion()
    async with abrir_agente(cfg) as (agente, _):
        estado = await agente.aget_state(config_de_ejecucion(thread_id, cfg.recursion_limit))
        previos = estado.values.get("messages", [])
        print(f"thread_id={thread_id} | mensajes guardados: {len(previos)}")
        if previos:
            print(f"Último mensaje recordado: {extraer_texto(previos[-1])[:120]}")
        print('Escribí tu pregunta ("salir" para terminar).\n')

        while True:
            texto = (await asyncio.to_thread(input, "Vos: ")).strip()
            if texto.lower() in {"salir", "exit", "quit"}:
                break
            if not texto:
                continue
            turno = await preguntar(agente, thread_id, texto, cfg.recursion_limit)
            print(turno_a_log(turno) if mostrar_traza else f"Agente: {turno['respuesta']}", end="\n\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thread-id", default="sesion-1", help="identificador de la sesión a recordar")
    parser.add_argument("--sin-traza", action="store_true", help="mostrar solo la respuesta final")
    args = parser.parse_args()
    asyncio.run(main(args.thread_id, mostrar_traza=not args.sin_traza))
