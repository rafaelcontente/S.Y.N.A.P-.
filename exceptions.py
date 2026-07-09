"""Excepções próprias do módulo Núcleo Cognitivo de Dupla Via."""

from __future__ import annotations


class NucleoError(Exception):
    """Erro base do módulo nucleo."""


class NucleoValidationError(NucleoError):
    """Erro de validação de input (parâmetros de geração inválidos)."""


class InsufficientDataError(NucleoError):
    """Erro de dados insuficientes para treinar o autoencoder de forma estável."""
