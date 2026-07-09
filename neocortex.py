"""Neocórtex Perceptual e Ancoragem Estatística (Contrato de Realismo).

Ponto de entrada do sistema S.Y.N.A.P. Recebe o esquema do utilizador,
as regras de negócio e uma âncora de realismo (CSV real, parâmetros
manuais, ou nenhuma) e produz um :class:`~neocortex.models.StatisticalContract`
imutável que serve de "tábua de lei" para os módulos seguintes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats

from .exceptions import (
    InsufficientSampleError,
    NeocortexValidationError,
)
from .models import (
    BusinessRule,
    CategoricalStatistics,
    ColumnStatistics,
    ColumnType,
    ContinuousStatistics,
    ContractSource,
    CorrelationPair,
    DistributionType,
    RuleOperator,
    SchemaDefinition,
    StatisticalContract,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

MIN_SAMPLE_SIZE_HARD: int = 50
"""Abaixo deste número de linhas, o CSV de ancoragem é rejeitado."""

MIN_SAMPLE_SIZE_RECOMMENDED: int = 100
"""Abaixo deste número de linhas (mas >= HARD), gera-se um aviso de
baixa confiança estatística no contrato."""

DEFAULT_KL_TOLERANCE: float = 0.05
"""Divergência de Kullback-Leibler máxima aceite por omissão."""

DEFAULT_STD_FRACTION_OF_RANGE: float = 6.0
"""Fração do intervalo [minimo, maximo] usada para estimar o desvio
padrão por omissão em Modo Sintético Puro (intervalo / 6 ~= 3-sigma)."""

STANDARD_NORMAL_MEDIA: float = 0.0
STANDARD_NORMAL_DESVIO: float = 1.0
"""Parâmetros da normal-padrão usados quando não há domínio nem âncora."""

_RULE_PATTERN = re.compile(
    r"^\s*(?P<coluna>\w+)\s*(?P<operador>>=|<=|==|!=|>|<)\s*"
    r"(?P<valor>-?\d+(?:\.\d+)?)\s*$"
)


# --------------------------------------------------------------------------
# Carregamento e validação do esquema
# --------------------------------------------------------------------------


def load_schema(schema_path: Path) -> SchemaDefinition:
    """Carrega e valida o esquema do utilizador a partir de YAML ou JSON.

    Args:
        schema_path: Caminho para o ficheiro `.yaml`, `.yml` ou `.json`.

    Returns:
        O esquema validado como :class:`SchemaDefinition`.

    Raises:
        NeocortexValidationError: Se o ficheiro não existir, tiver uma
            extensão não suportada, contiver YAML/JSON inválido, ou o
            esquema falhar a validação estrutural.
    """
    if not schema_path.exists():
        raise NeocortexValidationError(f"ficheiro de esquema não encontrado: {schema_path}")

    suffix = schema_path.suffix.lower()
    try:
        raw_text = schema_path.read_text(encoding="utf-8")
        if suffix in (".yaml", ".yml"):
            raw_data: dict[str, Any] = yaml.safe_load(raw_text)
        elif suffix == ".json":
            raw_data = json.loads(raw_text)
        else:
            raise NeocortexValidationError(
                f"extensão de esquema não suportada: '{suffix}' "
                "(esperado .yaml, .yml ou .json)"
            )
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise NeocortexValidationError(f"esquema malformado em {schema_path}: {exc}") from exc

    if raw_data is None:
        raise NeocortexValidationError(f"esquema vazio em {schema_path}")

    try:
        return SchemaDefinition.model_validate(raw_data)
    except Exception as exc:  # noqa: BLE001 - re-lançado com contexto próprio
        raise NeocortexValidationError(f"esquema inválido: {exc}") from exc


# --------------------------------------------------------------------------
# Regras de negócio
# --------------------------------------------------------------------------


def parse_business_rules(
    rules_text: list[str], schema: SchemaDefinition
) -> list[BusinessRule]:
    """Parseia regras de negócio em texto para :class:`BusinessRule`.

    Args:
        rules_text: Lista de regras no formato "coluna operador valor",
            ex.: "rendimento > 0".
        schema: Esquema já validado, usado para confirmar que a coluna
            referida em cada regra existe.

    Returns:
        Lista de regras parseadas.

    Raises:
        NeocortexValidationError: Se uma regra não seguir o formato
            esperado ou referir uma coluna inexistente no esquema.
    """
    parsed_rules: list[BusinessRule] = []
    for texto in rules_text:
        match = _RULE_PATTERN.match(texto)
        if match is None:
            raise NeocortexValidationError(
                f"regra de negócio malformada: '{texto}' "
                "(formato esperado: 'coluna operador valor')"
            )
        coluna = match.group("coluna")
        if coluna not in schema.colunas:
            raise NeocortexValidationError(
                f"regra '{texto}' refere coluna inexistente no esquema: '{coluna}'"
            )
        parsed_rules.append(
            BusinessRule(
                coluna=coluna,
                operador=RuleOperator(match.group("operador")),
                valor=float(match.group("valor")),
                texto_original=texto,
            )
        )
    return parsed_rules


def _validar_regras_contra_dados(
    df: pd.DataFrame, rules: list[BusinessRule]
) -> list[str]:
    """Verifica (sem falhar) se os dados reais respeitam as regras de negócio.

    Violações não interrompem a construção do contrato — geram apenas
    um aviso, pois os dados reais são um facto consumado; a regra serve
    para orientar a geração sintética futura, não para invalidar o passado.
    """
    warnings: list[str] = []
    operator_funcs = {
        RuleOperator.GT: lambda s, v: s > v,
        RuleOperator.LT: lambda s, v: s < v,
        RuleOperator.GE: lambda s, v: s >= v,
        RuleOperator.LE: lambda s, v: s <= v,
        RuleOperator.EQ: lambda s, v: s == v,
        RuleOperator.NE: lambda s, v: s != v,
    }
    for rule in rules:
        if rule.coluna not in df.columns:
            continue
        mask_valido = operator_funcs[rule.operador](df[rule.coluna], rule.valor)
        n_violacoes = int((~mask_valido).sum())
        if n_violacoes > 0:
            pct = 100.0 * n_violacoes / len(df)
            warnings.append(
                f"regra '{rule.texto_original}' violada em {n_violacoes} "
                f"linha(s) do CSV de ancoragem ({pct:.1f}%)"
            )
    return warnings


# --------------------------------------------------------------------------
# Estatísticas a partir de CSV real
# --------------------------------------------------------------------------


def _load_anchor_csv(csv_path: Path) -> pd.DataFrame:
    """Carrega o CSV de ancoragem e valida o tamanho mínimo da amostra.

    Raises:
        NeocortexValidationError: Se o ficheiro não existir ou não puder
            ser parseado como CSV.
        InsufficientSampleError: Se o número de linhas for inferior a
            :data:`MIN_SAMPLE_SIZE_HARD`.
    """
    if not csv_path.exists():
        raise NeocortexValidationError(f"ficheiro CSV de ancoragem não encontrado: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise NeocortexValidationError(f"CSV de ancoragem malformado: {exc}") from exc

    if len(df) < MIN_SAMPLE_SIZE_HARD:
        raise InsufficientSampleError(
            f"amostra insuficiente: {len(df)} linha(s) fornecidas, "
            f"mínimo rígido é {MIN_SAMPLE_SIZE_HARD}. "
            "Recomenda-se fornecer parâmetros manuais em alternativa."
        )
    return df


def _compute_continuous_stats_from_series(series: pd.Series) -> ContinuousStatistics:
    """Calcula estatísticas de referência para uma coluna numérica real."""
    media = float(series.mean())
    desvio = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    q1, q2, q3 = (float(v) for v in series.quantile([0.25, 0.5, 0.75]))
    return ContinuousStatistics(
        tipo=DistributionType.NORMAL,
        media=media,
        desvio=desvio,
        minimo=float(series.min()),
        maximo=float(series.max()),
        quartis=(q1, q2, q3),
    )


def _detect_categorical_order(
    df: pd.DataFrame, cat_col: str, categories: list[str], numeric_columns: list[str]
) -> tuple[str, list[str]] | None:
    """Deteta a coluna numérica mais associada a `cat_col` e a ordem das
    categorias por média crescente dessa coluna.

    Usa um score do tipo eta-quadrado (variância entre grupos / variância
    total) para escolher, entre as colunas numéricas, a que melhor separa
    as categorias — a mesma lógica usada para decidir a ordem em que o
    Módulo 2 deve mapear percentis da numérica para categorias (ver
    `hipocampo._sample_categorical_child`). Sem esta ordem, o mapeamento
    por percentil usaria a ordem arbitrária de `frequencias` (a ordem de
    `value_counts`), desligada de qualquer relação real com a numérica —
    um bug de fidelidade detectado durante a validação do sistema.

    Returns:
        Tuplo (nome_da_coluna_driver, categorias_ordenadas), ou `None`
        se nenhuma numérica mostrar associação (score) relevante.
    """
    melhor_col, melhor_score = None, 0.05  # piso mínimo de eta² para considerar associação real
    for num_col in numeric_columns:
        total_var = df[num_col].var()
        if not total_var or pd.isna(total_var):
            continue
        medias = df.groupby(cat_col)[num_col].mean()
        contagens = df.groupby(cat_col)[num_col].count()
        media_global = df[num_col].mean()
        var_entre_grupos = ((medias - media_global) ** 2 * contagens).sum() / len(df)
        score = float(var_entre_grupos / total_var)
        if score > melhor_score:
            melhor_score, melhor_col = score, num_col

    if melhor_col is None:
        return None

    medias_por_categoria = df.groupby(cat_col)[melhor_col].mean()
    ordem = [c for c in medias_por_categoria.sort_values().index if c in categories]
    # inclui categorias sem ocorrências no CSV (média indefinida) no fim, por omissão
    ordem += [c for c in categories if c not in ordem]
    return melhor_col, ordem


def _compute_categorical_stats_from_series(
    series: pd.Series,
    schema_categorias: list[str] | None,
    df: pd.DataFrame | None = None,
    numeric_columns: list[str] | None = None,
) -> tuple[CategoricalStatistics, list[str]]:
    """Calcula frequências empíricas para uma coluna categórica real.

    Se `df` e `numeric_columns` forem fornecidos, tenta também detectar
    a coluna numérica mais associada e a ordem das categorias por essa
    associação (ver :func:`_detect_categorical_order`) — necessário
    para o Módulo 2 mapear percentis para categorias na direção certa.
    """
    warnings: list[str] = []
    frequencias = (series.value_counts(normalize=True)).to_dict()
    frequencias = {str(k): float(v) for k, v in frequencias.items()}

    if schema_categorias is not None:
        categorias_inesperadas = set(frequencias) - set(schema_categorias)
        if categorias_inesperadas:
            warnings.append(
                f"coluna '{series.name}' contém categorias não declaradas "
                f"no esquema: {sorted(categorias_inesperadas)}"
            )

    driver, ordem = None, None
    if df is not None and numeric_columns:
        deteccao = _detect_categorical_order(df, str(series.name), list(frequencias), numeric_columns)
        if deteccao is not None:
            driver, ordem = deteccao

    return (
        CategoricalStatistics(frequencias=frequencias, ordem_categorias=ordem, driver_numerico=driver),
        warnings,
    )


def _compute_correlations(
    df: pd.DataFrame, numeric_columns: list[str]
) -> tuple[list[CorrelationPair], list[str]]:
    """Calcula a matriz de correlação de Pearson entre colunas numéricas.

    Pares envolvendo colunas de variância nula (desvio = 0) produzem
    correlação indefinida (NaN) e são omitidos do contrato, com um
    aviso associado.
    """
    warnings: list[str] = []
    pairs: list[CorrelationPair] = []
    if len(numeric_columns) < 2:
        return pairs, warnings

    matriz = df[numeric_columns].corr(method="pearson")
    for i, col_a in enumerate(numeric_columns):
        for col_b in numeric_columns[i + 1 :]:
            valor = matriz.loc[col_a, col_b]
            if pd.isna(valor):
                warnings.append(
                    f"correlação entre '{col_a}' e '{col_b}' indefinida "
                    "(variância nula numa das colunas) — omitida do contrato"
                )
                continue
            pairs.append(
                CorrelationPair(coluna_a=col_a, coluna_b=col_b, valor=float(valor))
            )
    return pairs, warnings


def _build_stats_from_csv(
    df: pd.DataFrame, schema: SchemaDefinition
) -> tuple[dict[str, ColumnStatistics], list[CorrelationPair], list[str]]:
    """Constrói as estatísticas de referência de todas as colunas a partir do CSV.

    Em duas passagens: primeiro todas as numéricas (para que
    `numeric_columns` esteja completo), depois as categóricas — que
    precisam de conhecer todas as numéricas disponíveis para detectar
    a sua coluna-driver (ver :func:`_detect_categorical_order`).
    """
    warnings: list[str] = []
    column_stats: dict[str, ColumnStatistics] = {}
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for nome_coluna, col_schema in schema.colunas.items():
        if nome_coluna not in df.columns:
            warnings.append(
                f"coluna '{nome_coluna}' declarada no esquema mas ausente do CSV "
                "de ancoragem — ignorada"
            )
            continue
        if col_schema.tipo == ColumnType.NUMERIC:
            column_stats[nome_coluna] = _compute_continuous_stats_from_series(df[nome_coluna])
            numeric_columns.append(nome_coluna)
        else:
            categorical_columns.append(nome_coluna)

    for nome_coluna in categorical_columns:
        col_schema = schema.colunas[nome_coluna]
        cat_stats, cat_warnings = _compute_categorical_stats_from_series(
            df[nome_coluna], col_schema.categorias, df=df, numeric_columns=numeric_columns
        )
        column_stats[nome_coluna] = cat_stats
        warnings.extend(cat_warnings)

    correlations, corr_warnings = _compute_correlations(df, numeric_columns)
    warnings.extend(corr_warnings)
    return column_stats, correlations, warnings


# --------------------------------------------------------------------------
# Estatísticas a partir de parâmetros manuais
# --------------------------------------------------------------------------


def _validar_parametros_manuais_contra_esquema(
    manual_params: dict[str, dict[str, Any]], schema: SchemaDefinition
) -> None:
    """Valida que os parâmetros manuais não contradizem o esquema.

    Raises:
        NeocortexValidationError: Se uma coluna manual não existir no
            esquema, se o tipo de distribuição não corresponder ao tipo
            de coluna, se o desvio padrão não for positivo, ou se a
            média cair fora do domínio [minimo, maximo] declarado.
    """
    for nome_coluna, params in manual_params.items():
        if nome_coluna not in schema.colunas:
            raise NeocortexValidationError(
                f"parâmetro manual para coluna inexistente no esquema: '{nome_coluna}'"
            )
        col_schema = schema.colunas[nome_coluna]
        distribuicao = params.get("distribuicao", "normal")

        if col_schema.tipo == ColumnType.NUMERIC:
            if distribuicao not in ("normal", "uniforme"):
                raise NeocortexValidationError(
                    f"coluna numérica '{nome_coluna}' não pode usar "
                    f"distribuição '{distribuicao}'"
                )
            media = params.get("media")
            desvio = params.get("desvio")
            if media is None or desvio is None:
                raise NeocortexValidationError(
                    f"parâmetros manuais de '{nome_coluna}' requerem "
                    "'media' e 'desvio'"
                )
            if desvio <= 0:
                raise NeocortexValidationError(
                    f"desvio padrão manual de '{nome_coluna}' deve ser > 0 "
                    f"(recebido: {desvio})"
                )
            if col_schema.minimo is not None and media < col_schema.minimo:
                raise NeocortexValidationError(
                    f"parâmetro contraditório: média manual de '{nome_coluna}' "
                    f"({media}) é inferior ao domínio mínimo declarado "
                    f"no esquema ({col_schema.minimo})"
                )
            if col_schema.maximo is not None and media > col_schema.maximo:
                raise NeocortexValidationError(
                    f"parâmetro contraditório: média manual de '{nome_coluna}' "
                    f"({media}) é superior ao domínio máximo declarado "
                    f"no esquema ({col_schema.maximo})"
                )
        else:
            if distribuicao != "categorica":
                raise NeocortexValidationError(
                    f"coluna categórica '{nome_coluna}' requer "
                    "distribuicao='categorica'"
                )
            frequencias = params.get("frequencias")
            if not frequencias:
                raise NeocortexValidationError(
                    f"parâmetros manuais de '{nome_coluna}' requerem 'frequencias'"
                )
            if col_schema.categorias is not None:
                categorias_desconhecidas = set(frequencias) - set(col_schema.categorias)
                if categorias_desconhecidas:
                    raise NeocortexValidationError(
                        f"parâmetro contraditório: '{nome_coluna}' declara "
                        f"frequências para categorias fora do esquema: "
                        f"{sorted(categorias_desconhecidas)}"
                    )


def _build_stats_from_manual_params(
    manual_params: dict[str, dict[str, Any]], schema: SchemaDefinition
) -> dict[str, ColumnStatistics]:
    """Constrói estatísticas de referência a partir de parâmetros manuais.

    Sem dados reais disponíveis, os quartis de colunas contínuas são
    estimados analiticamente a partir da distribuição declarada (normal
    ou uniforme), e o domínio [minimo, maximo] usa o declarado no
    esquema quando presente, ou uma estimativa a média ± 3 desvios.
    """
    column_stats: dict[str, ColumnStatistics] = {}

    for nome_coluna, col_schema in schema.colunas.items():
        params = manual_params.get(nome_coluna)
        if params is None:
            continue

        if col_schema.tipo == ColumnType.NUMERIC:
            media = float(params["media"])
            desvio = float(params["desvio"])
            distribuicao = params.get("distribuicao", "normal")
            minimo = col_schema.minimo if col_schema.minimo is not None else media - 3 * desvio
            maximo = col_schema.maximo if col_schema.maximo is not None else media + 3 * desvio

            if distribuicao == "normal":
                q1, q2, q3 = scipy_stats.norm.ppf([0.25, 0.5, 0.75], loc=media, scale=desvio)
            else:
                q1, q2, q3 = np.percentile([minimo, maximo], [25, 50, 75])

            column_stats[nome_coluna] = ContinuousStatistics(
                tipo=DistributionType(distribuicao),
                media=media,
                desvio=desvio,
                minimo=float(minimo),
                maximo=float(maximo),
                quartis=(float(q1), float(q2), float(q3)),
            )
        else:
            frequencias = {str(k): float(v) for k, v in params["frequencias"].items()}
            column_stats[nome_coluna] = CategoricalStatistics(frequencias=frequencias)

    return column_stats


# --------------------------------------------------------------------------
# Modo Sintético Puro (sem âncora)
# --------------------------------------------------------------------------


def _build_synthetic_defaults(
    schema: SchemaDefinition,
) -> tuple[dict[str, ColumnStatistics], list[str]]:
    """Constrói distribuições padrão quando o utilizador não fornece âncora.

    Colunas categóricas recebem frequências uniformes sobre as
    categorias declaradas. Colunas numéricas recebem uma distribuição
    normal centrada no domínio declarado, ou a normal-padrão N(0, 1)
    se nenhum domínio for declarado.
    """
    warnings: list[str] = [
        "MODO SINTÉTICO PURO: nenhuma âncora de realismo foi fornecida "
        "(nem CSV real, nem parâmetros manuais). O contrato assume "
        "distribuições padrão (normal para contínuas, uniforme para "
        "categóricas) que podem não refletir a realidade do domínio. "
        "Recomenda-se fortemente fornecer uma âncora real."
    ]

    column_stats: dict[str, ColumnStatistics] = {}
    for nome_coluna, col_schema in schema.colunas.items():
        if col_schema.tipo == ColumnType.CATEGORICAL:
            if not col_schema.categorias:
                raise NeocortexValidationError(
                    f"coluna categórica '{nome_coluna}' sem 'categorias' "
                    "declaradas no esquema não pode receber distribuição "
                    "padrão em Modo Sintético Puro"
                )
            n = len(col_schema.categorias)
            column_stats[nome_coluna] = CategoricalStatistics(
                frequencias={cat: 1.0 / n for cat in col_schema.categorias}
            )
        else:
            if col_schema.minimo is not None and col_schema.maximo is not None:
                media = (col_schema.minimo + col_schema.maximo) / 2
                desvio = (col_schema.maximo - col_schema.minimo) / DEFAULT_STD_FRACTION_OF_RANGE
                minimo, maximo = col_schema.minimo, col_schema.maximo
            else:
                media, desvio = STANDARD_NORMAL_MEDIA, STANDARD_NORMAL_DESVIO
                minimo, maximo = media - 3 * desvio, media + 3 * desvio
                warnings.append(
                    f"coluna '{nome_coluna}' sem domínio declarado — "
                    f"assumida normal-padrão N({media}, {desvio})"
                )
            q1, q2, q3 = scipy_stats.norm.ppf([0.25, 0.5, 0.75], loc=media, scale=desvio)
            column_stats[nome_coluna] = ContinuousStatistics(
                tipo=DistributionType.NORMAL,
                media=media,
                desvio=desvio,
                minimo=minimo,
                maximo=maximo,
                quartis=(float(q1), float(q2), float(q3)),
            )
    return column_stats, warnings


# --------------------------------------------------------------------------
# Orquestrador principal
# --------------------------------------------------------------------------


def build_contract(
    schema_path: Path,
    rules: list[str] | None = None,
    anchor_csv: Path | None = None,
    manual_params: dict[str, dict[str, Any]] | None = None,
    tolerancia_kl: float = DEFAULT_KL_TOLERANCE,
) -> StatisticalContract:
    """Constrói o Contrato Estatístico Interno (Módulo 1 do S.Y.N.A.P.).

    Args:
        schema_path: Caminho para o ficheiro de esquema (YAML ou JSON).
        rules: Regras de negócio em texto, ex.: `["rendimento > 0"]`.
        anchor_csv: Caminho opcional para um CSV com dados reais
            (mínimo `MIN_SAMPLE_SIZE_HARD` linhas). Mutuamente exclusivo
            com `manual_params`.
        manual_params: Parâmetros estatísticos manuais opcionais, por
            coluna. Mutuamente exclusivo com `anchor_csv`.
        tolerancia_kl: Divergência de Kullback-Leibler máxima aceite.

    Returns:
        O :class:`StatisticalContract` imutável, pronto a ser consumido
        pelos Módulos 3, 6 e 7.

    Raises:
        NeocortexValidationError: Se o esquema, as regras ou os
            parâmetros manuais forem inválidos ou contraditórios.
        InsufficientSampleError: Se o CSV de ancoragem tiver menos de
            `MIN_SAMPLE_SIZE_HARD` linhas.
    """
    if anchor_csv is not None and manual_params is not None:
        raise NeocortexValidationError(
            "forneça apenas uma âncora: 'anchor_csv' OU 'manual_params', não ambas"
        )

    schema = load_schema(schema_path)
    # As regras de negócio vivem por omissão no próprio esquema (campo
    # `regras`); o parâmetro `rules` permite sobrepor/estender esse conjunto
    # sem precisar de reescrever o ficheiro de esquema.
    regras_efetivas = rules if rules is not None else schema.regras
    parsed_rules = parse_business_rules(regras_efetivas, schema)

    warnings: list[str] = []
    sample_size: int | None = None

    if anchor_csv is not None:
        df = _load_anchor_csv(anchor_csv)
        sample_size = len(df)
        if sample_size < MIN_SAMPLE_SIZE_RECOMMENDED:
            warnings.append(
                f"amostra pequena ({sample_size} linhas): confiança estatística "
                f"reduzida. Recomenda-se >= {MIN_SAMPLE_SIZE_RECOMMENDED} linhas."
            )
        column_stats, correlations, csv_warnings = _build_stats_from_csv(df, schema)
        warnings.extend(csv_warnings)
        warnings.extend(_validar_regras_contra_dados(df, parsed_rules))
        source = ContractSource.REAL_CSV
        is_synthetic_pure = False

    elif manual_params is not None:
        _validar_parametros_manuais_contra_esquema(manual_params, schema)
        column_stats = _build_stats_from_manual_params(manual_params, schema)
        correlations = []
        source = ContractSource.MANUAL_PARAMS
        is_synthetic_pure = False

    else:
        column_stats, synth_warnings = _build_synthetic_defaults(schema)
        correlations = []
        warnings.extend(synth_warnings)
        source = ContractSource.SYNTHETIC_PURE
        is_synthetic_pure = True
        logger.warning("Contrato construído em Modo Sintético Puro — sem âncora real.")

    contract = StatisticalContract(
        column_stats=column_stats,
        correlations=correlations,
        tolerancia_kl=tolerancia_kl,
        source=source,
        is_synthetic_pure=is_synthetic_pure,
        sample_size=sample_size,
        rules=parsed_rules,
        warnings=warnings,
    )
    logger.info(
        "Contrato Estatístico construído: source=%s, colunas=%d, "
        "correlações=%d, avisos=%d",
        source.value,
        len(column_stats),
        len(correlations),
        len(warnings),
    )
    return contract
