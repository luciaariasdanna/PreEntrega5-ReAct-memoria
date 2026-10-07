"""Prueba de ejecución: corre los escenarios de la consigna y guarda la traza ReAct.

    python demo.py

Genera trazas/traza_ejecucion.json y trazas/traza_ejecucion.log.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from agente.config import Configuracion, cargar_configuracion
from agente.runtime import abrir_agente, preguntar
from agente.traza import Turno, turno_a_log

CARPETA_TRAZAS = Path("trazas")

# Cada escenario es un thread_id distinto con una lista de "sesiones"; entre
# sesiones se cierra y se vuelve a abrir la conexión a SQLite, para probar que la
# memoria sale del archivo en disco y no de una variable en RAM.
ESCENARIOS: dict[str, dict[str, Any]] = {
    "1_multi_paso_y_memoria": {
        "descripcion": (
            "Multi-paso: el agente necesita 2 herramientas (nombre -> id -> pedidos). "
            "Después, con el mismo thread_id y tras reabrir el checkpointer, '¿y el último?' "
            "solo tiene sentido si recuerda de qué cliente se hablaba."
        ),
        "thread_id": "demo-multi-paso",
        "sesiones": [
            ["¿Cuántos pedidos tuvo Ana García y cuál fue el total?"],
            ["¿Y el último?", "¿Y Juan Pérez?"],
        ],
    },
    "2_ciclo_de_retorno_reintento": {
        "descripcion": (
            "La herramienta devuelve error por una tilde faltante ('Garcia') con una sugerencia; "
            "el agente hace un segundo intento con el nombre corregido."
        ),
        "thread_id": "demo-reintento",
        "sesiones": [["¿Cuánto gastó en total Ana Garcia?"]],
    },
    "3_ciclo_de_retorno_aclaracion": {
        "descripcion": (
            "El cliente no existe y no hay sugerencias: el agente pide aclaración en lugar de inventar. "
            "El usuario aclara en el turno siguiente y el agente completa la tarea."
        ),
        "thread_id": "demo-aclaracion",
        "sesiones": [["¿Cuántos pedidos tuvo Roberto Sánchez?", "Perdón, quise decir María López."]],
    },
}


async def correr_escenario(cfg: Configuracion, escenario: dict[str, Any]) -> list[Turno]:
    thread_id: str = escenario["thread_id"]
    turnos: list[Turno] = []

    for n_sesion, preguntas in enumerate(escenario["sesiones"], start=1):
        async with abrir_agente(cfg) as (agente, checkpointer):
            if n_sesion == 1:
                await checkpointer.adelete_thread(thread_id)  # demo reproducible
            else:
                print(f"\n   ... checkpointer reabierto desde {cfg.ruta_checkpoints} (mismo thread_id) ...")
            for pregunta in preguntas:
                turno = await preguntar(agente, thread_id, pregunta, cfg.recursion_limit)
                print("\n" + turno_a_log(turno))
                turnos.append(turno)
    return turnos


async def main() -> None:
    cfg = cargar_configuracion()
    traza: dict[str, Any] = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modelo": cfg.modelo,
        "recursion_limit": cfg.recursion_limit,
        "escenarios": {},
    }
    logs: list[str] = []

    for nombre, escenario in ESCENARIOS.items():
        print(f"\n{'=' * 80}\n{nombre}\n{escenario['descripcion']}\n{'=' * 80}")
        turnos = await correr_escenario(cfg, escenario)
        traza["escenarios"][nombre] = {"descripcion": escenario["descripcion"], "turnos": turnos}
        logs.append(f"### {nombre}\n# {escenario['descripcion']}\n\n" + "\n\n".join(map(turno_a_log, turnos)))

    CARPETA_TRAZAS.mkdir(exist_ok=True)
    (CARPETA_TRAZAS / "traza_ejecucion.json").write_text(
        json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (CARPETA_TRAZAS / "traza_ejecucion.log").write_text("\n\n\n".join(logs) + "\n", encoding="utf-8")
    print(f"\nTraza guardada en {CARPETA_TRAZAS}/traza_ejecucion.json y .log")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # consola de Windows
    asyncio.run(main())
