"""Configuración leída de variables de entorno (archivo .env, nunca commiteado)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Configuracion:
    modelo: str
    ruta_checkpoints: Path
    recursion_limit: int
    max_mensajes_contexto: int


def cargar_configuracion() -> Configuracion:
    load_dotenv()
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Copiá .env.example a .env y pegá tu key "
            "(gratis en https://aistudio.google.com/apikey)."
        )
    return Configuracion(
        modelo=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
        ruta_checkpoints=Path(os.getenv("CHECKPOINTS_DB", "checkpoints.sqlite")),
        recursion_limit=int(os.getenv("RECURSION_LIMIT", "10")),
        max_mensajes_contexto=int(os.getenv("MAX_MENSAJES_CONTEXTO", "20")),
    )
