"""Hipocampo e Motor de Génese Causal (Semente + Validação Estrutural).

Módulo 2 do S.Y.N.A.P. Recebe o Contrato Estatístico (Módulo 1) e uma
DAG causal opcional. Valida/ajusta a DAG por teste de d-separação
(correlação parcial + Fisher-Z), aprende uma DAG do zero por um
algoritmo PC simplificado quando nenhuma é fornecida, e gera a semente
inicial (5.000-10.000 linhas) por amostragem topológica com um modelo
linear-Gaussiano calibrado para reproduzir as correlações do Contrato.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from synap.neocortex.models import (
    BusinessRule,
    CategoricalStatistics,
    ContinuousStatistics,
    RuleOperator,
    StatisticalContract,
)

from .exceptions import CyclicGraphError, HipocampoValidationError
from .models import (
    CausalDAG,
    CausalEdge,
    DAGAdjustment,
    HipocampoOutput,
    SeedGenerationReport,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------

ALPHA_D_SEPARACAO: float = 0.01
"""Nível de significância do teste de Fisher-Z para d-separação."""

MIN_SEED_ROWS: int = 5_000
MAX_SEED_ROWS: int = 10_000

NOISE_VARIANCE_FLOOR: float = 1e-6
"""Piso de variância residual, evita log(0) e ruído nulo no BIC/geração."""

MAX_CONDITIONING_SET_SIZE: int = 2
"""Tamanho máximo do conjunto de condicionamento testado na fase de
esqueleto do PC-algorithm simplificado. Mantém o custo computacional
controlado para DAGs de dimensão prática (dezenas de nós)."""

CATEGORICAL_PARENT_EFFECT_SIZE: float = 0.3
"""Peso heurístico do efeito de um pai categórico sobre um filho
numérico, usado apenas quando não existe nenhum pai numérico
disponível para calibrar a relação a partir de correlações do Contrato."""

RESAMPLE_ATTEMPTS_FOR_RULES: int = 5
"""Número de tentativas de reamostragem de ruído antes de recorrer a
corte (clipping) para satisfazer uma regra de negócio."""


# --------------------------------------------------------------------------
# Matriz de correlação e correlação parcial
# --------------------------------------------------------------------------


def _numeric_columns(contract: StatisticalContract) -> list[str]:
    """Devolve os nomes das colunas contínuas (numéricas) do Contrato, por ordem de declaração."""
    return [
        nome
        for nome, stat in contract.column_stats.items()
        if isinstance(stat, ContinuousStatistics)
    ]


def _categorical_columns(contract: StatisticalContract) -> list[str]:
    """Devolve os nomes das colunas categóricas do Contrato, por ordem de declaração."""
    return [
        nome
        for nome, stat in contract.column_stats.items()
        if isinstance(stat, CategoricalStatistics)
    ]


def build_correlation_matrix(contract: StatisticalContract) -> pd.DataFrame:
    """Reconstrói a matriz de correlação completa a partir do Contrato.

    Pares de colunas numéricas sem correlação registada no Contrato
    (ex.: omitidos por variância nula) são preenchidos com 0.0.

    Args:
        contract: O Contrato Estatístico do Módulo 1.

    Returns:
        Matriz de correlação simétrica, indexada pelas colunas numéricas.
    """
    cols = _numeric_columns(contract)
    matriz = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for par in contract.correlations:
        if par.coluna_a in cols and par.coluna_b in cols:
            matriz.loc[par.coluna_a, par.coluna_b] = par.valor
            matriz.loc[par.coluna_b, par.coluna_a] = par.valor
    return matriz


def partial_correlation(
    correlation_matrix: pd.DataFrame, x: str, y: str, z: list[str]
) -> float:
    """Calcula a correlação parcial entre `x` e `y`, condicionada em `z`.

    Usa a inversa (pseudo-inversa, para robustez a colinearidade) da
    submatriz de correlação {x, y} ∪ z: rho_xy.z = -P_xy / sqrt(P_xx * P_yy),
    onde P é a matriz de precisão (inversa da correlação).

    Args:
        correlation_matrix: Matriz de correlação completa.
        x: Primeira variável.
        y: Segunda variável.
        z: Conjunto de variáveis de condicionamento (pode ser vazio).

    Returns:
        Correlação parcial, no intervalo [-1, 1].
    """
    if not z:
        return float(np.clip(correlation_matrix.loc[x, y], -1.0, 1.0))

    variaveis = [x, y, *z]
    submatriz = correlation_matrix.loc[variaveis, variaveis].to_numpy()
    precisao = np.linalg.pinv(submatriz)
    denom = np.sqrt(precisao[0, 0] * precisao[1, 1])
    if denom <= 0:
        return 0.0
    rho = -precisao[0, 1] / denom
    return float(np.clip(rho, -1.0, 1.0))


def fisher_z_p_value(r: float, n: int, cond_set_size: int) -> float:
    """Calcula o p-valor do teste de Fisher-Z para independência condicional.

    Args:
        r: Correlação (parcial) observada.
        n: Tamanho da amostra que sustenta a correlação.
        cond_set_size: Número de variáveis no conjunto de condicionamento.

    Returns:
        P-valor bicaudal para H0: a correlação (parcial) é zero. Devolve
        1.0 (independência perfeita) se os graus de liberdade forem
        insuficientes (n - cond_set_size - 3 <= 0).
    """
    graus_liberdade = n - cond_set_size - 3
    if graus_liberdade <= 0:
        return 1.0
    r_seguro = float(np.clip(r, -0.999999, 0.999999))
    z = 0.5 * np.log((1 + r_seguro) / (1 - r_seguro)) * np.sqrt(graus_liberdade)
    return float(2 * (1 - scipy_stats.norm.cdf(abs(z))))


# --------------------------------------------------------------------------
# Construção e validação de DAG fornecida pelo utilizador
# --------------------------------------------------------------------------


def build_dag(
    nodes: list[str], edges: list[tuple[str, str]] | None = None
) -> CausalDAG:
    """Constrói e valida uma :class:`CausalDAG`.

    Args:
        nodes: Todos os nós (colunas) que devem existir na DAG.
        edges: Arestas dirigidas `(origem, destino)`. `None` equivale a
            nenhuma aresta (grafo totalmente desconectado).

    Returns:
        A DAG validada (aciclicidade e existência de nós garantidas).

    Raises:
        CyclicGraphError: Se as arestas fornecidas formarem um ciclo.
        HipocampoValidationError: Se uma aresta referir um nó não
            presente em `nodes`.
    """
    arestas = [CausalEdge(origem=o, destino=d) for o, d in (edges or [])]
    try:
        return CausalDAG(nos=nodes, arestas=arestas)
    except Exception as exc:  # noqa: BLE001 - reclassificado abaixo com contexto
        mensagem = str(exc)
        if "ciclo" in mensagem:
            raise CyclicGraphError(f"DAG cíclica fornecida: {mensagem}") from exc
        raise HipocampoValidationError(f"DAG inválida: {mensagem}") from exc


def _validate_dag_against_contract(dag: CausalDAG, contract: StatisticalContract) -> None:
    """Garante que todos os nós da DAG existem no Contrato Estatístico."""
    colunas_contrato = set(contract.column_stats)
    nos_desconhecidos = set(dag.nos) - colunas_contrato
    if nos_desconhecidos:
        raise HipocampoValidationError(
            f"DAG refere nó(s) ausente(s) do Contrato Estatístico: "
            f"{sorted(nos_desconhecidos)}"
        )


# --------------------------------------------------------------------------
# BIC — critério de informação Bayesiano para regressão linear-Gaussiana
# --------------------------------------------------------------------------


def _r_squared(correlation_matrix: pd.DataFrame, child: str, parents: list[str]) -> float:
    """Calcula o R² da regressão linear de `child` sobre `parents` a partir da matriz de correlação."""
    if not parents:
        return 0.0
    r_xx = correlation_matrix.loc[parents, parents].to_numpy()
    r_xy = correlation_matrix.loc[parents, child].to_numpy()
    r2 = float(r_xy @ np.linalg.pinv(r_xx) @ r_xy)
    return float(np.clip(r2, 0.0, 1.0 - NOISE_VARIANCE_FLOOR))


def _node_bic(
    correlation_matrix: pd.DataFrame, child: str, parents: list[str], n: int
) -> float:
    """Calcula o BIC do modelo linear-Gaussiano de `child` dados os `parents`.

    Nós sem pais (raízes) não têm modelo de regressão associado e
    contribuem 0.0 — o mesmo em qualquer variante da DAG comparada,
    pelo que não afeta comparações de melhoria relativa.
    """
    if not parents:
        return 0.0
    r2 = _r_squared(correlation_matrix, child, parents)
    variancia_residual = max(1.0 - r2, NOISE_VARIANCE_FLOOR)
    k = len(parents) + 1  # +1 para o intercepto
    return n * np.log(variancia_residual) + k * np.log(n)


def compute_dag_bic(dag: CausalDAG, contract: StatisticalContract) -> float:
    """Calcula o BIC total da DAG, somando o BIC de cada nó numérico com pais.

    Raises:
        HipocampoValidationError: Se o Contrato não tiver `sample_size`
            (o BIC depende do tamanho da amostra).
    """
    if contract.sample_size is None:
        raise HipocampoValidationError(
            "cálculo de BIC requer 'sample_size' no Contrato Estatístico "
            "(indisponível em contratos de parâmetros manuais ou sintéticos puros)"
        )
    correlation_matrix = build_correlation_matrix(contract)
    numeric_cols = set(_numeric_columns(contract))
    total = 0.0
    for node in dag.nos:
        if node not in numeric_cols:
            continue
        parents = [p for p in dag.parents(node) if p in numeric_cols]
        total += _node_bic(correlation_matrix, node, parents, contract.sample_size)
    return total


# --------------------------------------------------------------------------
# Deteção e adição automática de arestas em falta (d-separação)
# --------------------------------------------------------------------------


def _infer_edge_direction(
    dag: CausalDAG, x: str, y: str, causal_order: list[str] | None
) -> tuple[str, str, bool]:
    """Infere a direção (origem, destino) para uma nova aresta entre `x` e `y`.

    Ordem de preferência:
    1. Se já existe um caminho dirigido x->..->y (ou y->..->x) na DAG
       atual, usa essa ordem (o exemplo canónico: Educação->Rendimento
       ->Risco implica Educação antes de Risco).
    2. Se uma `causal_order` explícita for fornecida, usa a posição
       relativa de x e y nessa lista.
    3. Caso contrário, usa a ordem de declaração dos nós na DAG,
       assinalando `direcao_assumida=True`.
    """
    if dag.has_directed_path(x, y):
        return x, y, False
    if dag.has_directed_path(y, x):
        return y, x, False
    if causal_order is not None and x in causal_order and y in causal_order:
        return (x, y, False) if causal_order.index(x) < causal_order.index(y) else (y, x, False)
    return (x, y, True) if dag.nos.index(x) < dag.nos.index(y) else (y, x, True)


def detect_missing_edges(
    dag: CausalDAG,
    contract: StatisticalContract,
    causal_order: list[str] | None = None,
    alpha: float = ALPHA_D_SEPARACAO,
) -> list[DAGAdjustment]:
    """Testa todos os pares não-adjacentes da DAG para arestas em falta.

    Para cada par de colunas numéricas não ligadas diretamente na DAG,
    calcula a correlação parcial condicionada na união dos pais de
    ambos os nós. Se essa correlação for estatisticamente significativa
    (p < alpha), reporta-a como uma aresta em falta.

    Args:
        dag: A DAG atual.
        contract: O Contrato Estatístico (fornece a matriz de correlação e `sample_size`).
        causal_order: Ordem causal opcional para desambiguar a direção
            de arestas entre nós sem relação de ancestralidade prévia.
        alpha: Nível de significância do teste.

    Returns:
        Lista de :class:`DAGAdjustment` (ainda não aplicados à DAG).
    """
    if contract.sample_size is None:
        return []

    correlation_matrix = build_correlation_matrix(contract)
    numeric_cols = _numeric_columns(contract)
    candidatos: list[DAGAdjustment] = []

    for x, y in itertools.combinations(numeric_cols, 2):
        if dag.is_adjacent(x, y):
            continue
        conditioning_set = sorted(set(dag.parents(x)) | set(dag.parents(y)) - {x, y})
        conditioning_set = [c for c in conditioning_set if c in numeric_cols]

        rho = partial_correlation(correlation_matrix, x, y, conditioning_set)
        p_valor = fisher_z_p_value(rho, contract.sample_size, len(conditioning_set))
        if p_valor >= alpha:
            continue

        origem, destino, assumida = _infer_edge_direction(dag, x, y, causal_order)
        parents_destino_antes = dag.parents(destino)
        bic_antes = _node_bic(correlation_matrix, destino, parents_destino_antes, contract.sample_size)
        parents_destino_depois = [*parents_destino_antes, origem]
        bic_depois = _node_bic(correlation_matrix, destino, parents_destino_depois, contract.sample_size)
        melhoria = (
            (bic_antes - bic_depois) / abs(bic_antes) if bic_antes != 0 else float(bic_antes != bic_depois)
        )

        candidatos.append(
            DAGAdjustment(
                origem=origem,
                destino=destino,
                correlacao_parcial=rho,
                p_valor=p_valor,
                bic_antes=bic_antes,
                bic_depois=bic_depois,
                melhoria_percentual=melhoria,
                direcao_assumida=assumida,
            )
        )
    return candidatos


def add_missing_edges(
    dag: CausalDAG,
    contract: StatisticalContract,
    causal_order: list[str] | None = None,
    alpha: float = ALPHA_D_SEPARACAO,
) -> tuple[CausalDAG, list[DAGAdjustment], list[str]]:
    """Deteta e aplica arestas em falta na DAG, uma de cada vez.

    Depois de cada aresta aplicada, o processo é repetido sobre a DAG
    atualizada (uma vez que os conjuntos de pais mudam), até não
    existirem mais candidatos ou até uma aresta candidata introduzir
    um ciclo (nesse caso é ignorada e reportada como aviso).

    Returns:
        Tuplo (dag_ajustada, ajustes_aplicados, avisos).
    """
    warnings: list[str] = []
    ajustes_aplicados: list[DAGAdjustment] = []
    dag_atual = dag

    if contract.sample_size is None:
        warnings.append(
            "amostra ausente no Contrato Estatístico — verificação de "
            "d-separação ignorada (BIC e testes de significância requerem "
            "'sample_size')"
        )
        return dag_atual, ajustes_aplicados, warnings

    max_iteracoes = len(dag.nos) ** 2  # limite defensivo contra ciclos de deteção
    for _ in range(max_iteracoes):
        candidatos = detect_missing_edges(dag_atual, contract, causal_order, alpha)
        if not candidatos:
            break
        melhor = min(candidatos, key=lambda c: c.p_valor)
        try:
            dag_atual = dag_atual.with_edge(melhor.origem, melhor.destino)
        except Exception:  # noqa: BLE001 - aresta cíclica, ignorar e avisar
            warnings.append(
                f"aresta candidata '{melhor.origem} -> {melhor.destino}' "
                "introduziria um ciclo — ignorada"
            )
            break
        ajustes_aplicados.append(melhor)
        warnings.append(
            f"aresta em falta detetada e adicionada: '{melhor.origem} -> "
            f"{melhor.destino}' (correlação parcial={melhor.correlacao_parcial:.3f}, "
            f"p={melhor.p_valor:.2e}, melhoria BIC={melhor.melhoria_percentual:.1%})"
        )
        if melhor.direcao_assumida:
            warnings.append(
                f"direção de '{melhor.origem} -> {melhor.destino}' foi assumida "
                "pela ordem de declaração das colunas — sem relação de "
                "ancestralidade prévia nem 'causal_order' para desambiguar"
            )
    return dag_atual, ajustes_aplicados, warnings


# --------------------------------------------------------------------------
# Aprendizagem de DAG do zero (PC-algorithm simplificado)
# --------------------------------------------------------------------------


def learn_dag_from_contract(
    contract: StatisticalContract,
    causal_order: list[str] | None = None,
    alpha: float = ALPHA_D_SEPARACAO,
    max_conditioning_size: int = MAX_CONDITIONING_SET_SIZE,
) -> tuple[CausalDAG, list[str]]:
    """Aprende uma DAG aproximada a partir da matriz de correlação do Contrato.

    Implementação simplificada do PC-algorithm, limitada a colunas
    numéricas (colunas categóricas entram como nós isolados, já que o
    Contrato não regista correlações entre tipos):

    1. Fase de esqueleto: parte de um grafo completo não-dirigido e
       remove arestas cuja correlação parcial (condicionada em
       subconjuntos dos vizinhos, até `max_conditioning_size`) não é
       estatisticamente significativa.
    2. Orientação de v-estruturas: tripletos não-blindados x-z-y onde
       `z` não pertence ao conjunto separador de (x, y) são orientados
       como colisores x->z<-y.
    3. Arestas remanescentes (não orientadas pelas v-estruturas) são
       orientadas pela `causal_order`, se fornecida, ou pela ordem de
       declaração das colunas no Contrato — sem aplicar as regras de
       Meek na íntegra. Esta é uma simplificação assumida e documentada,
       adequada para DAGs esparsas de dimensão prática.

    Returns:
        Tuplo (dag_aprendida, avisos).
    """
    warnings: list[str] = []
    numeric_cols = _numeric_columns(contract)
    categorical_cols = _categorical_columns(contract)
    n = contract.sample_size

    if n is None:
        warnings.append(
            "amostra ausente no Contrato Estatístico — aprendizagem de DAG "
            "impossível sem 'sample_size'; devolvida DAG sem arestas"
        )
        return CausalDAG(nos=[*numeric_cols, *categorical_cols], arestas=[]), warnings

    correlation_matrix = build_correlation_matrix(contract)

    # --- Fase 1: esqueleto ---
    skeleton: set[frozenset[str]] = {
        frozenset((a, b)) for a, b in itertools.combinations(numeric_cols, 2)
    }
    separating_sets: dict[frozenset[str], list[str]] = {}

    for tamanho_condicionamento in range(max_conditioning_size + 1):
        for par in list(skeleton):
            x, y = tuple(par)
            vizinhos_x = [
                v
                for edge in skeleton
                if x in edge
                for v in edge
                if v != x and v != y
            ]
            candidatos_z = [
                list(subset)
                for subset in itertools.combinations(sorted(set(vizinhos_x)), tamanho_condicionamento)
            ]
            for z in candidatos_z:
                rho = partial_correlation(correlation_matrix, x, y, z)
                p_valor = fisher_z_p_value(rho, n, len(z))
                if p_valor >= alpha:
                    skeleton.discard(par)
                    separating_sets[par] = z
                    break

    # --- Fase 2: orientação de v-estruturas ---
    orientadas: set[tuple[str, str]] = set()
    for z in numeric_cols:
        vizinhos_z = [v for v in numeric_cols if v != z and frozenset((v, z)) in skeleton]
        for x, y in itertools.combinations(vizinhos_z, 2):
            if frozenset((x, y)) in skeleton:
                continue  # x e y são adjacentes, não é um tripleto não-blindado
            separador = separating_sets.get(frozenset((x, y)), [])
            if z not in separador:
                orientadas.add((x, z))
                orientadas.add((y, z))

    # --- Fase 3: orientação das arestas remanescentes ---
    ordem_fallback = causal_order if causal_order is not None else numeric_cols
    arestas_finais: list[tuple[str, str]] = list(orientadas)
    for par in skeleton:
        x, y = tuple(par)
        if (x, y) in orientadas or (y, x) in orientadas:
            continue
        if x in ordem_fallback and y in ordem_fallback:
            origem, destino = (x, y) if ordem_fallback.index(x) < ordem_fallback.index(y) else (y, x)
        else:
            origem, destino = (x, y) if numeric_cols.index(x) < numeric_cols.index(y) else (y, x)
        arestas_finais.append((origem, destino))

    if categorical_cols:
        warnings.append(
            f"coluna(s) categórica(s) {categorical_cols} adicionada(s) como "
            "nó(s) isolado(s) — o Contrato não regista correlações entre "
            "colunas categóricas e numéricas, pelo que não podem ser "
            "causalmente ordenadas por este algoritmo"
        )

    try:
        dag = CausalDAG(
            nos=[*numeric_cols, *categorical_cols],
            arestas=[CausalEdge(origem=o, destino=d) for o, d in arestas_finais],
        )
    except Exception as exc:  # noqa: BLE001 - ciclo residual da orientação de fallback
        warnings.append(
            f"orientação automática produziu um ciclo ({exc}) — a DAG "
            "aprendida foi reduzida ao esqueleto sem as arestas conflituosas"
        )
        dag = _remove_cycle_causing_edges(numeric_cols, categorical_cols, arestas_finais)

    warnings.append(
        f"DAG aprendida automaticamente via PC-algorithm simplificado "
        f"(conjunto de condicionamento até {max_conditioning_size} variáveis); "
        "revise as arestas sem relação de ancestralidade prévia."
    )
    return dag, warnings


def _remove_cycle_causing_edges(
    numeric_cols: list[str], categorical_cols: list[str], arestas: list[tuple[str, str]]
) -> CausalDAG:
    """Adiciona arestas incrementalmente, ignorando as que introduzem ciclos."""
    dag = CausalDAG(nos=[*numeric_cols, *categorical_cols], arestas=[])
    for origem, destino in arestas:
        try:
            dag = dag.with_edge(origem, destino)
        except Exception:  # noqa: BLE001 - aresta conflituosa, omitida deliberadamente
            continue
    return dag


# --------------------------------------------------------------------------
# Aplicação de regras de negócio à semente gerada
# --------------------------------------------------------------------------

_OPERATOR_FUNCS = {
    RuleOperator.GT: lambda s, v: s > v,
    RuleOperator.LT: lambda s, v: s < v,
    RuleOperator.GE: lambda s, v: s >= v,
    RuleOperator.LE: lambda s, v: s <= v,
    RuleOperator.EQ: lambda s, v: s == v,
    RuleOperator.NE: lambda s, v: s != v,
}


def _enforce_rule(
    values: np.ndarray, rule: BusinessRule, regenerate: Any
) -> tuple[np.ndarray, int]:
    """Corrige violações de uma regra de negócio numa coluna gerada.

    Tenta primeiro reamostrar apenas as células violadoras (chamando
    `regenerate(n)` para obter `n` novos valores); se ainda houver
    violações após `RESAMPLE_ATTEMPTS_FOR_RULES` tentativas, corta
    (clip) os valores remanescentes para o limite da regra.

    Returns:
        Tuplo (valores_corrigidos, numero_de_celulas_cortadas).
    """
    mask_valido = _OPERATOR_FUNCS[rule.operador](values, rule.valor)
    for _ in range(RESAMPLE_ATTEMPTS_FOR_RULES):
        if mask_valido.all():
            break
        n_invalidas = int((~mask_valido).sum())
        values = values.copy()
        values[~mask_valido] = regenerate(n_invalidas)
        mask_valido = _OPERATOR_FUNCS[rule.operador](values, rule.valor)

    n_cortadas = int((~mask_valido).sum())
    if n_cortadas > 0:
        epsilon = 1e-9
        if rule.operador in (RuleOperator.GT, RuleOperator.GE):
            values = np.where(mask_valido, values, rule.valor + epsilon)
        elif rule.operador in (RuleOperator.LT, RuleOperator.LE):
            values = np.where(mask_valido, values, rule.valor - epsilon)
        else:
            values = np.where(mask_valido, values, rule.valor)
    return values, n_cortadas


# --------------------------------------------------------------------------
# Geração da semente
# --------------------------------------------------------------------------


def _sample_root_continuous(
    stat: ContinuousStatistics, n_rows: int, rng: np.random.Generator
) -> np.ndarray:
    """Amostra uma coluna raiz contínua a partir da sua distribuição declarada."""
    if stat.tipo.value == "uniforme":
        return rng.uniform(stat.minimo, stat.maximo, n_rows)
    return rng.normal(stat.media, stat.desvio, n_rows)


def _sample_root_categorical(
    stat: CategoricalStatistics, n_rows: int, rng: np.random.Generator
) -> np.ndarray:
    """Amostra uma coluna raiz categórica a partir das frequências declaradas."""
    categorias = list(stat.frequencias)
    probs = np.array([stat.frequencias[c] for c in categorias])
    probs = probs / probs.sum()
    return rng.choice(categorias, size=n_rows, p=probs)


def _sample_continuous_child(
    node: str,
    parents: list[str],
    stat: ContinuousStatistics,
    contract: StatisticalContract,
    correlation_matrix: pd.DataFrame,
    data: dict[str, np.ndarray],
    n_rows: int,
    rng: np.random.Generator,
    warnings: list[str],
) -> np.ndarray:
    """Amostra uma coluna filha contínua via modelo linear-Gaussiano.

    Os coeficientes são resolvidos algebricamente (regressão múltipla
    a partir da matriz de correlação) para reproduzir, em expectativa,
    as correlações-alvo do Contrato entre `node` e os seus pais numéricos.
    """
    numeric_parents = [p for p in parents if isinstance(contract.column_stats[p], ContinuousStatistics)]
    categorical_parents = [p for p in parents if p not in numeric_parents]

    contribuicao = np.zeros(n_rows)
    variancia_explicada = 0.0

    if numeric_parents:
        parents_no_contrato = [p for p in numeric_parents if p in correlation_matrix.columns]
        if node in correlation_matrix.columns and parents_no_contrato:
            r_xx = correlation_matrix.loc[parents_no_contrato, parents_no_contrato].to_numpy()
            r_xy = np.array(
                [
                    correlation_matrix.loc[p, node] if node in correlation_matrix.columns else 0.0
                    for p in parents_no_contrato
                ]
            )
            beta_std = np.linalg.pinv(r_xx) @ r_xy
            variancia_explicada = float(np.clip(r_xy @ beta_std, 0.0, 1.0 - NOISE_VARIANCE_FLOOR))
            for peso, pai in zip(beta_std, parents_no_contrato):
                pai_stat = contract.column_stats[pai]
                z_pai = (data[pai] - pai_stat.media) / pai_stat.desvio
                contribuicao += peso * z_pai
        else:
            warnings.append(
                f"coluna '{node}' sem correlação-alvo definida no Contrato para "
                f"os pais numéricos {parents_no_contrato} — assumida associação nula"
            )

    if categorical_parents and not numeric_parents:
        warnings.append(
            f"coluna '{node}' depende apenas de pai(s) categórico(s) "
            f"{categorical_parents}; o Contrato não define correlação "
            "cruzada entre tipos, pelo que foi usado um efeito heurístico "
            f"fixo ({CATEGORICAL_PARENT_EFFECT_SIZE})"
        )
        for pai in categorical_parents:
            categorias = list(contract.column_stats[pai].frequencias)
            n_cat = len(categorias)
            rank = {cat: i for i, cat in enumerate(categorias)}
            offsets = np.array([rank[v] for v in data[pai]], dtype=float)
            if n_cat > 1:
                offsets = (offsets / (n_cat - 1)) * 2 - 1  # normalizado para [-1, 1]
            contribuicao += CATEGORICAL_PARENT_EFFECT_SIZE * offsets
        variancia_explicada = max(variancia_explicada, CATEGORICAL_PARENT_EFFECT_SIZE**2)

    variancia_residual = max(1.0 - variancia_explicada, NOISE_VARIANCE_FLOOR)
    ruido = rng.normal(0.0, np.sqrt(variancia_residual), n_rows)
    return stat.media + stat.desvio * (contribuicao + ruido)


def _sample_categorical_child(
    node: str,
    parents: list[str],
    stat: CategoricalStatistics,
    contract: StatisticalContract,
    data: dict[str, np.ndarray],
    n_rows: int,
    rng: np.random.Generator,
    warnings: list[str],
) -> np.ndarray:
    """Amostra uma coluna filha categórica, acoplada ao(s) pai(s) numérico(s) por percentil."""
    numeric_parents = [p for p in parents if isinstance(contract.column_stats[p], ContinuousStatistics)]
    # A ordem importa: o mapeamento por percentil abaixo assume que
    # 'categorias' está ordenada por associação crescente com os pais
    # numéricos. Sem `ordem_categorias` (detectada no Módulo 1 a partir
    # do CSV real), cairíamos de volta na ordem arbitrária de
    # `frequencias` (a ordem de `value_counts`) — que não tem nenhuma
    # relação necessária com a numérica, e produziria um acoplamento
    # internamente consistente mas na direção ERRADA face à realidade.
    categorias = stat.ordem_categorias if stat.ordem_categorias else list(stat.frequencias)
    probs = np.array([stat.frequencias[c] for c in categorias])
    probs = probs / probs.sum()

    if not numeric_parents:
        if parents:
            warnings.append(
                f"coluna categórica '{node}' depende apenas de pai(s) "
                f"categórico(s) {parents} — acoplamento categórico-categórico "
                "não suportado; usada amostragem marginal"
            )
        return rng.choice(categorias, size=n_rows, p=probs)

    if not stat.ordem_categorias:
        warnings.append(
            f"coluna '{node}' sem ordem de categorias detectada no Contrato — "
            "o acoplamento com os pais numéricos usa a ordem de 'frequencias', "
            "que pode não refletir a direção real da associação (ver Módulo 1)"
        )

    # Combina os pais numéricos (padronizados) numa única pontuação e usa
    # o seu percentil para escolher a categoria via CDF inversa acumulada.
    pontuacao = np.zeros(n_rows)
    for pai in numeric_parents:
        pai_stat = contract.column_stats[pai]
        pontuacao += (data[pai] - pai_stat.media) / pai_stat.desvio
    percentil = scipy_stats.rankdata(pontuacao) / (n_rows + 1)

    cumulativo = np.cumsum(probs)
    indices = np.searchsorted(cumulativo, percentil)
    indices = np.clip(indices, 0, len(categorias) - 1)
    return np.array(categorias)[indices]


def generate_seed(
    dag: CausalDAG,
    contract: StatisticalContract,
    n_rows: int = MIN_SEED_ROWS,
    rng_seed: int | None = None,
) -> tuple[pd.DataFrame, SeedGenerationReport, list[str]]:
    """Gera a semente inicial por amostragem topológica sobre a DAG.

    Args:
        dag: A DAG final (validada e ajustada).
        contract: O Contrato Estatístico.
        n_rows: Número de linhas a gerar (entre `MIN_SEED_ROWS` e `MAX_SEED_ROWS`).
        rng_seed: Semente do gerador aleatório, para reprodutibilidade.

    Returns:
        Tuplo (dataframe_semente, relatorio_qualidade, avisos).

    Raises:
        HipocampoValidationError: Se `n_rows` estiver fora do intervalo permitido.
    """
    if not (MIN_SEED_ROWS <= n_rows <= MAX_SEED_ROWS):
        raise HipocampoValidationError(
            f"n_rows={n_rows} fora do intervalo permitido "
            f"[{MIN_SEED_ROWS}, {MAX_SEED_ROWS}]"
        )

    warnings: list[str] = []
    rng = np.random.default_rng(rng_seed)
    correlation_matrix = build_correlation_matrix(contract)
    data: dict[str, np.ndarray] = {}

    for node in dag.topological_order():
        stat = contract.column_stats[node]
        parents = dag.parents(node)

        if isinstance(stat, ContinuousStatistics):
            if not parents:
                data[node] = _sample_root_continuous(stat, n_rows, rng)
            else:
                data[node] = _sample_continuous_child(
                    node, parents, stat, contract, correlation_matrix, data, n_rows, rng, warnings
                )
        else:
            if not parents:
                data[node] = _sample_root_categorical(stat, n_rows, rng)
            else:
                data[node] = _sample_categorical_child(
                    node, parents, stat, contract, data, n_rows, rng, warnings
                )

    regras_aplicadas = 0
    total_cortadas = 0
    for rule in contract.rules:
        if rule.coluna not in data:
            continue
        regras_aplicadas += 1

        def regenerate(n: int, _rule: BusinessRule = rule) -> np.ndarray:
            stat_regra = contract.column_stats[_rule.coluna]
            if isinstance(stat_regra, ContinuousStatistics):
                return rng.normal(stat_regra.media, stat_regra.desvio, n)
            return rng.choice(list(stat_regra.frequencias), size=n)

        data[rule.coluna], n_cortadas = _enforce_rule(data[rule.coluna], rule, regenerate)
        total_cortadas += n_cortadas
        if n_cortadas > 0:
            warnings.append(
                f"regra '{rule.texto_original}' exigiu corte (clip) em "
                f"{n_cortadas} linha(s) após reamostragem"
            )

    df = pd.DataFrame(data)[list(contract.column_stats)]

    erros: list[float] = []
    for par in contract.correlations:
        if par.coluna_a in df.columns and par.coluna_b in df.columns:
            empirica = float(df[par.coluna_a].corr(df[par.coluna_b]))
            erros.append(abs(empirica - par.valor))

    erro_medio = float(np.mean(erros)) if erros else 0.0
    erro_maximo = float(np.max(erros)) if erros else 0.0
    if erro_medio > 0.05:
        warnings.append(
            f"erro médio de correlação da semente ({erro_medio:.1%}) acima "
            "do limiar recomendado de 5%"
        )

    relatorio = SeedGenerationReport(
        n_linhas=n_rows,
        correlacoes_avaliadas=len(erros),
        erro_medio_absoluto=erro_medio,
        erro_maximo_absoluto=erro_maximo,
        regras_aplicadas=regras_aplicadas,
        linhas_corrigidas_por_regras=total_cortadas,
    )
    return df, relatorio, warnings


# --------------------------------------------------------------------------
# Orquestrador principal
# --------------------------------------------------------------------------


def run_hipocampo(
    contract: StatisticalContract,
    user_dag_edges: list[tuple[str, str]] | None = None,
    causal_order: list[str] | None = None,
    n_rows: int = MIN_SEED_ROWS,
    rng_seed: int | None = None,
) -> HipocampoOutput:
    """Constrói o Hipocampo (Módulo 2 do S.Y.N.A.P.): DAG validada + semente.

    Args:
        contract: O Contrato Estatístico produzido pelo Módulo 1.
        user_dag_edges: Arestas dirigidas `(origem, destino)` fornecidas
            pelo utilizador. Se `None`, a DAG é aprendida a partir do Contrato.
        causal_order: Ordem causal opcional (lista de nomes de colunas)
            usada para desambiguar a direção de arestas adicionadas
            automaticamente sem relação de ancestralidade prévia.
        n_rows: Número de linhas da semente a gerar.
        rng_seed: Semente do gerador aleatório, para reprodutibilidade.

    Returns:
        :class:`HipocampoOutput` com a DAG final, a semente gerada, os
        ajustes estruturais aplicados e o relatório de qualidade.
    """
    warnings: list[str] = []
    todas_colunas = list(contract.column_stats)

    if user_dag_edges is not None:
        dag = build_dag(todas_colunas, user_dag_edges)
        _validate_dag_against_contract(dag, contract)
    else:
        dag, warnings_aprendizagem = learn_dag_from_contract(contract, causal_order)
        warnings.extend(warnings_aprendizagem)

    dag_ajustada, ajustes, warnings_ajuste = add_missing_edges(dag, contract, causal_order)
    warnings.extend(warnings_ajuste)

    seed_df, relatorio, warnings_geracao = generate_seed(dag_ajustada, contract, n_rows, rng_seed)
    warnings.extend(warnings_geracao)

    logger.info(
        "Hipocampo construído: nos=%d, arestas=%d, ajustes=%d, linhas=%d, "
        "erro_medio_correlacao=%.4f",
        len(dag_ajustada.nos),
        len(dag_ajustada.arestas),
        len(ajustes),
        relatorio.n_linhas,
        relatorio.erro_medio_absoluto,
    )

    return HipocampoOutput(
        dag=dag_ajustada,
        seed=seed_df,
        ajustes=ajustes,
        relatorio=relatorio,
        warnings=warnings,
    )
