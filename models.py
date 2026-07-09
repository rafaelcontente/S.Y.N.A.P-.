"""Estruturas de dados e tipos do módulo Núcleo Cognitivo de Dupla Via.

Define a linha de base do autoencoder, os anti-exemplos destinados ao
Hipocampo (Módulo 2), e o relatório/saída de cada lote — o pacote que
fecha o ciclo neural-simbólico com o Módulo 3 dentro do mesmo lote.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class NucleoRejectionReason(str, Enum):
    """Motivo de rejeição de uma linha quimera candidata no Núcleo Cognitivo."""

    MAHALANOBIS = "distancia_mahalanobis"
    PLAUSIBILIDADE = "score_plausibilidade"
    ASP = "rejeicao_logica_asp"
    ARTEFACTO = "artefacto_autoencoder"
    TENTATIVAS_ESGOTADAS = "tentativas_esgotadas"


class AutoencoderBaseline(BaseModel):
    """Linha de base da perda de reconstrução do autoencoder, fixada na semente.

    Attributes:
        media_perda: Perda de reconstrução média observada na semente.
        desvio_perda: Desvio padrão da perda de reconstrução na semente
            (com piso mínimo, ver `MIN_LOSS_STD_FLOOR`).
        limiar: `media_perda + k * desvio_perda` — acima disto, uma
            linha é classificada como ESTRANHA.
    """

    model_config = ConfigDict(frozen=True)

    media_perda: float
    desvio_perda: float
    limiar: float


class AntiExample(BaseModel):
    """Um anti-exemplo — artefacto detectado, destinado ao Hipocampo (Módulo 2).

    Representa uma combinação de valores que, apesar de aprovada pelos
    filtros estatísticos e pelo ASP, foi identificada como um artefacto
    irreal pelo autoencoder.
    """

    model_config = ConfigDict(frozen=True)

    valores: dict[str, Any]
    perda_reconstrucao: float
    limiar: float


class NucleoBatchReport(BaseModel):
    """Relatório estatístico de um lote processado pelo Núcleo Cognitivo.

    Attributes:
        linhas_solicitadas: Número de linhas pedidas.
        linhas_geradas: Número de linhas efetivamente aprovadas (nos 4 filtros).
        tentativas_totais: Número total de linhas quimera candidatas testadas.
        rejeicoes_por_motivo: Contagem de rejeições por :class:`NucleoRejectionReason`.
        taxa_rejeicao: Fração de candidatas rejeitadas sobre o total testado.
        artefactos_detectados: Nº de candidatas rejeitadas especificamente
            por estranheza do autoencoder.
        perda_reconstrucao_media_aprovadas: Perda de reconstrução média
            entre as linhas finalmente aprovadas.
        reciclagens_autoencoder: Nº de vezes que o autoencoder foi
            atualizado (`partial_fit`) durante este lote.
    """

    model_config = ConfigDict(frozen=True)

    linhas_solicitadas: int
    linhas_geradas: int
    tentativas_totais: int
    rejeicoes_por_motivo: dict[str, int] = Field(default_factory=dict)
    taxa_rejeicao: float
    artefactos_detectados: int
    perda_reconstrucao_media_aprovadas: float
    reciclagens_autoencoder: int


class NucleoOutput(BaseModel):
    """Saída completa de um lote do Módulo 5.

    Attributes:
        lote: DataFrame apenas com as novas linhas aprovadas neste lote.
        relatorio: Estatísticas de filtragem do lote.
        anti_exemplos: Artefactos detectados, prontos a registar no
            Hipocampo (Módulo 2) como conhecimento negativo.
        memoria_final: A memória vetorial no final do lote (semente +
            aprovadas), para encadear "ciclos" sucessivos preservando a
            aprendizagem de pesos de seleção entre chamadas.
        pesos_finais: Pesos de seleção das fontes no final do lote,
            no mesmo alinhamento de `memoria_final` — preservam o
            efeito de qualquer atenção negativa aplicada, para reutilizar
            num ciclo seguinte.
        warnings: Alertas gerados durante o processamento.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    lote: pd.DataFrame
    relatorio: NucleoBatchReport
    anti_exemplos: list[AntiExample] = Field(default_factory=list)
    memoria_final: pd.DataFrame
    pesos_finais: list[float] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
