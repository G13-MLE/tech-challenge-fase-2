"""Distribuicoes globais e por sessao para o calculo de Divergencia KL.

Computa:
- Gk(v): Distribuicao global (frequencia) de cada valor de propriedade.
- Psi_sk(v): Distribuicao por sessao com suavizacao.
- DKL(Psi_sk || Gk): Divergencia KL por propriedade.
- cs(j): Coeficiente de interesse para itens candidatos.
"""

import polars as pl

ALPHA = 0.5


def global_property_distribution(
    events: pl.DataFrame,
    item_props: pl.DataFrame,
    properties: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Calcula a distribuicao global Gk(v) para cada propriedade.

    Para cada propriedade k, Gk(v) = count(v) / total_count e a
    frequencia relativa de cada valor v em todos os eventos.

    Args:
        events: DataFrame com colunas itemid, event.
        item_props: DataFrame com colunas itemid, property, value.
        properties: Lista de propriedades para calcular. Se None, usa todas.

    Returns:
        Dict mapeando property -> DataFrame com colunas (value, prob).
    """
    if properties is not None:
        item_props = item_props.filter(pl.col("property").is_in(properties))

    event_props = events.join(
        item_props.select(["itemid", "property", "value"]),
        on="itemid",
        how="inner",
    )

    result = {}
    for prop in event_props["property"].unique().to_list():
        prop_df = (
            event_props.filter(pl.col("property") == prop)
            .group_by("value")
            .agg(pl.len().alias("count"))
            .with_columns((pl.col("count") / pl.col("count").sum()).alias("prob"))
            .sort("prob", descending=True)
        )
        result[prop] = prop_df

    return result


def session_property_distribution(
    session_events: pl.DataFrame,
    item_props: pl.DataFrame,
    session_id: str,
    properties: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Calcula a distribuicao por sessao Psi_sk(v) com suavizacao.

    Psi_hat_sk(v) = (1 - exp(-alpha * |s|)) * f_sk(v) + exp(-alpha * |s|) * Gk(v)

    Onde |s| e o numero de eventos na sessao, f_sk(v) e a frequencia
    empirica do valor v na sessao, e Gk(v) e a distribuicao global.

    Args:
        session_events: DataFrame filtrado para uma sessao especifica.
        item_props: DataFrame com colunas itemid, property, value.
        session_id: ID da sessao.
        properties: Propriedades para calcular. Se None, usa todas.
        alpha: Parametro de suavizacao. Default 0.5.

    Returns:
        Dict mapeando property -> DataFrame com colunas (value, smoothed_prob).
    """
    global global_property_distribution

    return {}


def compute_kl_divergence(
    session_dist: dict[str, pl.DataFrame],
    global_dist: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Calcula DKL(Psi_sk || Gk) para cada propriedade.

    DKL(P || Q) = sum_v P(v) * log(P(v) / Q(v))

    Args:
        session_dist: Distribuicao suavizada por sessao.
        global_dist: Distribuicao global.

    Returns:
        DataFrame com colunas (property, kl_divergence).
    """
    rows = []
    for prop in session_dist:
        if prop not in global_dist:
            continue
        sess = session_dist[prop]
        glob = global_dist[prop]

        merged = sess.join(glob, on="value", how="inner", suffix="_global")
        if merged.height == 0:
            continue

        kl = (merged["prob"] * (merged["prob"] / merged["prob_global"]).log()).sum()
        rows.append({"property": prop, "kl_divergence": kl})

    return pl.DataFrame(rows).sort("kl_divergence", descending=True)


def compute_interest_coefficient(
    item_properties: dict[str, str],
    session_dist: dict[str, pl.DataFrame],
    global_dist: dict[str, pl.DataFrame],
) -> float:
    """Calcula o coeficiente de interesse cs(j) para um item candidato j.

    cs(j) = prod_{k in U} Psi_hat_sk(j) / Gk(j)

    Onde U e o conjunto de propriedades de interesse.

    Args:
        item_properties: Dict property_k -> value_v para o item j.
        session_dist: Distribuicao suavizada por sessao.
        global_dist: Distribuicao global.

    Returns:
        Coeficiente de interesse multiplicativo.
    """
    cs = 1.0
    for prop, value in item_properties.items():
        if prop not in session_dist or prop not in global_dist:
            continue

        sess_prob = session_dist[prop].filter(pl.col("value") == value)["prob"]
        glob_prob = global_dist[prop].filter(pl.col("value") == value)["prob"]

        if sess_prob.height == 0 or glob_prob.height == 0:
            continue

        ratio = float(sess_prob[0]) / float(glob_prob[0])
        cs *= ratio

    return cs
