"""Construcao da super tabela de features com merge_asof (point-in-time).

Une events com item_properties (categoryid, available, propriedades numericas)
respeitando a temporalidade para evitar data leakage.
"""

import polars as pl

from data_pipeline.load import load_category_tree


def build_super_table(
    events: pl.DataFrame,
    item_props: pl.DataFrame,
    category_tree_path: str | None = None,
) -> pl.DataFrame:
    """Constroi a super tabela com features de item, categoria e historico.

    Args:
        events: DataFrame com colunas timestamp, visitorid, event, itemid,
            transactionid, datetime, session_id.
        item_props: DataFrame filtrado de item_properties com colunas
            timestamp, itemid, property, value, property_time,
            value_numeric.
        category_tree_path: Caminho para category_tree.csv. Se fornecido,
            adiciona category_parentid.

    Returns:
        DataFrame com todas as features adicionadas.
    """
    events = events.sort("timestamp")

    categoryid_pivot = _pivot_special_property(item_props, "categoryid")
    available_pivot = _pivot_special_property(item_props, "available")
    numeric_pivot = _pivot_numeric_properties(item_props)

    result = events

    if categoryid_pivot.height > 0:
        result = _merge_asof_property(result, categoryid_pivot, "categoryid_asof")
    else:
        result = result.with_columns(
            pl.lit(None).cast(pl.Int64).alias("categoryid_asof")
        )

    if available_pivot.height > 0:
        result = _merge_asof_property(result, available_pivot, "available_asof")
    else:
        result = result.with_columns(
            pl.lit(None).cast(pl.Int64).alias("available_asof")
        )

    if numeric_pivot.height > 0:
        for col_name in [
            c for c in numeric_pivot.columns if c not in {"itemid", "property_time"}
        ]:
            result = _merge_asof_numeric(result, numeric_pivot, col_name)

    if category_tree_path is not None:
        cat_tree = load_category_tree(category_tree_path)
        if "categoryid_asof" in result.columns:
            result = result.join(
                cat_tree, left_on="categoryid_asof", right_on="categoryid", how="left"
            )
            if "parentid" in result.columns:
                result = result.rename({"parentid": "category_parentid"})

    result = _add_historical_features(result)

    result = result.with_columns(
        (pl.col("event") == "transaction").cast(pl.Int8).alias("target_transaction"),
        ((pl.col("event") == "addtocart") | (pl.col("event") == "transaction"))
        .cast(pl.Int8)
        .alias("target_addtocart_or_transaction"),
        pl.col("datetime").dt.month().alias("event_month"),
        pl.col("datetime").dt.weekday().alias("event_dayofweek"),
        pl.col("datetime").dt.hour().alias("event_hour"),
    )

    return result


def _pivot_special_property(item_props: pl.DataFrame, prop: str) -> pl.DataFrame:
    """Faz pivot de uma propriedade especial (categoryid ou available)."""
    df = item_props.filter(pl.col("property") == prop)
    if df.height == 0:
        return df

    col_type = pl.Int64 if prop == "categoryid" else pl.Int64

    df = (
        df.select(["itemid", "property_time", "value"])
        .with_columns(pl.col("value").cast(col_type, strict=False).alias(prop))
        .drop("value")
        .sort(["itemid", "property_time"])
    )
    return df


def _pivot_numeric_properties(item_props: pl.DataFrame) -> pl.DataFrame:
    """Faz pivot de propriedades numericas que possuem value_numeric."""
    numeric_props = item_props.filter(
        pl.col("value_numeric").is_not_null()
        & (pl.col("property") != "categoryid")
        & (pl.col("property") != "available")
    )

    if numeric_props.height == 0:
        return numeric_props

    prop_names = numeric_props["property"].unique().to_list()

    pivoted = numeric_props.pivot(
        index=["itemid", "property_time"],
        on="property",
        values="value_numeric",
    ).sort(["itemid", "property_time"])

    return pivoted


def _merge_asof_property(
    events: pl.DataFrame, prop_df: pl.DataFrame, col_name: str
) -> pl.DataFrame:
    """Faz merge_asof entre events e prop_df usando timestamp."""
    events = events.sort("timestamp")

    if "property_time" not in prop_df.columns:
        return events.with_columns(pl.lit(None).cast(pl.Int64).alias(col_name))

    prop_df = prop_df.sort("property_time")

    return events.join_asof(
        prop_df,
        left_on="timestamp",
        right_on="property_time",
        by="itemid",
        strategy="backward",
    ).drop("property_time", strict=False)


def _merge_asof_numeric(
    events: pl.DataFrame, numeric_df: pl.DataFrame, col_name: str
) -> pl.DataFrame:
    """Faz merge_asof para uma coluna numerica especifica."""
    events = events.sort("timestamp")

    subset = numeric_df.select(["itemid", "property_time", col_name]).sort(
        "property_time"
    )

    return events.join_asof(
        subset,
        left_on="timestamp",
        right_on="property_time",
        by="itemid",
        strategy="backward",
    ).drop("property_time", strict=False)


def _add_historical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Adiciona features acumuladas: eventos anteriores e tempo desde ultimo evento."""
    result = df.sort(["visitorid", "timestamp"])

    result = result.with_columns(
        pl.col("event").cum_count().over("visitorid").alias("user_events_before"),
    )

    result = result.with_columns(
        (pl.col("timestamp").diff().cast(pl.Float64) / 60_000)
        .over("visitorid")
        .alias("user_minutes_since_previous"),
    )

    result = result.sort(["itemid", "timestamp"])

    result = result.with_columns(
        pl.col("event").cum_count().over("itemid").alias("item_events_before"),
    )

    result = result.with_columns(
        (pl.col("timestamp").diff().cast(pl.Float64) / 60_000)
        .over("itemid")
        .alias("item_minutes_since_previous"),
    )

    return result.sort(["visitorid", "timestamp"])
