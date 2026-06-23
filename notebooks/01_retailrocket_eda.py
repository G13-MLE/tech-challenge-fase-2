# /// script
# dependencies = [
#     "matplotlib>=3.10.9",
#     "marimo>=0.23.5",
#     "pandas>=2.3.3",
#     "polars>=1.41.2",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EDA RetailRocket — Análise Exploratória Unificada

    Este notebook consolida a análise exploratória do dataset RetailRocket em três eixos:

    1. **Perfil do dataset** — shapes, nulos, distribuição de eventos, esparsidade
    2. **Gramática de `value` e propriedades de item** — classificação dos 7 tipos de valor, perfil por propriedade
    3. **Fluxo de eventos e super tabela** — transições, sessões, features temporais, join plan

    Decisões técnicas são documentadas inline ao longo do notebook.
    """)
    return


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path

    plt.style.use("seaborn-v0_8-whitegrid")

    DATA_DIR = Path(os.environ.get("RETAILROCKET_DATA_DIR", Path.cwd().parent / "data" / "raw"))
    if not DATA_DIR.exists():
        DATA_DIR = Path.cwd() / "data" / "raw"

    events_path = DATA_DIR / "events.csv"
    category_tree_path = DATA_DIR / "category_tree.csv"
    item_prop_paths = [
        DATA_DIR / "item_properties_part1.csv",
        DATA_DIR / "item_properties_part2.csv",
    ]

    missing = [
        p.name for p in [events_path, category_tree_path] + item_prop_paths if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(f"CSV(s) nao encontrados: {', '.join(missing)}")

    events_path, category_tree_path, item_prop_paths
    return DATA_DIR, Path, events_path, category_tree_path, item_prop_paths, pl, pd, plt, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Perfil do Dataset
    """)
    return


@app.cell
def _(events_path, pl):
    events = pl.read_csv(events_path).with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
    )
    events
    return (events,)


@app.cell
def _(events, pl):
    null_counts = pl.DataFrame(
        {
            "column": events.columns,
            "null_count": [events[col].null_count() for col in events.columns],
            "null_pct": [
                f"{events[col].null_count() / events.height * 100:.2f}%" for col in events.columns
            ],
        }
    )
    null_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `transactionid` é nulo em ~99.2% das linhas — apenas eventos do tipo `transaction` preenchem esse campo.
    Isso é esperado e não é problema de qualidade.
    """)
    return


@app.cell
def _(events, pl):
    events.group_by("event", pl.col("transactionid").is_null()).agg(pl.len()).sort(
        "len", descending=True
    )
    return


@app.cell
def _(events, pl):
    events_with_date = events.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
    )
    events_with_date["event"].value_counts().with_columns(
        (pl.col("count") / pl.col("count").sum() * 100).round(1).alias("pct"),
    ).sort("count", descending=True)
    return (events_with_date,)


@app.cell
def _(events, mo, pl):
    _total = events.height
    _unique_visitors = events["visitorid"].n_unique()
    _unique_items = events["itemid"].n_unique()
    _unique_pairs = events.select("visitorid", "itemid").n_unique()
    _sparsity = 1 - _unique_pairs / (_unique_visitors * _unique_items)

    mo.md(f"""
    ### Esparsidade

    - **{_total:,} eventos** de **{_unique_visitors:,} visitors** sobre **{_unique_items:,} itens**
    - Pares úteis visitor-item: **{_unique_pairs:,}**
    - Esparsidade da matriz: **{_sparsity:.6%}**
    - Período: **{events["datetime"].min()}** a **{events["datetime"].max()}**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Sessões — Por que criar e como definir
    """)
    return


@app.cell
def _(events, mo, pl):
    _single = events.group_by("visitorid").agg(pl.len().alias("n")).filter(pl.col("n") == 1).height
    _unique = events["visitorid"].n_unique()
    _pct = _single / _unique * 100

    mo.md(f"""
    **{_pct:.1f}% dos visitors aparecem apenas 1 vez.** Sem agrupar eventos em sessões, o sistema de recomendação
    não consegue capturar padrões de navegação dentro de uma mesma visita.
    """)
    return


@app.cell
def _(events_with_date, pl):
    sorted_events = events_with_date.sort(["visitorid", "timestamp"])
    deltas = sorted_events.group_by("visitorid").agg(
        (pl.col("timestamp").diff().cast(pl.Float64) / 60000).alias("delta_min"),
    )
    exploded_deltas = deltas.explode("delta_min").filter(pl.col("delta_min").is_not_null())

    gap_dist = (
        exploded_deltas.with_columns(
            pl.col("delta_min")
            .cut(breaks=[5, 10, 30, 60, 120, 1440, 10080, 43200])
            .alias("gap_bin"),
        )
        .group_by("gap_bin")
        .agg(pl.len().alias("count"))
        .sort("gap_bin")
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).round(1).alias("pct"),
        )
    )
    gap_dist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ~61% dos gaps são < 5 min (navegação bursty). O salto significativo está em > 2h (~22%).
    Poucos gaps caem entre 30–60 min, então thresholds de 30 e 60 min produzem resultados similares.
    """)
    return


@app.cell
def _(events_with_date, pl):
    _sorted = events_with_date.sort(["visitorid", "timestamp"])
    _s30 = _sorted.group_by("visitorid").agg(
        (pl.col("timestamp").diff().cast(pl.Float64) / 60000 >= 30)
        .cast(pl.Int32)
        .sum()
        .over("visitorid")
        .alias("breaks"),
    )
    _s60 = _sorted.group_by("visitorid").agg(
        (pl.col("timestamp").diff().cast(pl.Float64) / 60000 >= 60)
        .cast(pl.Int32)
        .sum()
        .over("visitorid")
        .alias("breaks"),
    )
    _n30 = _s30.select((pl.col("breaks") + 1).sum()).item()
    _n60 = _s60.select((pl.col("breaks") + 1).sum()).item()

    pl.DataFrame(
        {
            "threshold": ["30 min (gap)", "60 min (gap)"],
            "estimated_sessions": [_n30, _n60],
            "total_events": [events_with_date.height, events_with_date.height],
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Heurística de sessão adotada

    Nova sessão quando:
    1. Gap ≥ 30 min entre eventos consecutivos do mesmo visitor, **OU**
    2. Evento anterior == `transaction` **E** evento atual != `transaction`

    Transações consecutivas ficam na mesma sessão (mesma jornada de compra).
    Só se inicia nova sessão quando o usuário volta a navegar após comprar.
    """)
    return


@app.cell
def _(events_with_date, pl):
    events_with_session = (
        events_with_date.sort(["visitorid", "timestamp"])
        .with_columns(
            pl.col("timestamp").diff().cast(pl.Float64).over("visitorid").alias("delta_ms"),
            pl.col("event").shift(1).over("visitorid").alias("prev_event"),
        )
        .with_columns(
            (
                pl.col("delta_ms").is_null()
                | (pl.col("delta_ms") / 60000 >= 30)
                | ((pl.col("prev_event") == "transaction") & (pl.col("event") != "transaction"))
            ).alias("is_new_session"),
        )
        .with_columns(
            session_per_visitor=pl.col("is_new_session").cast(pl.Int32).cum_sum().over("visitorid"),
        )
        .with_columns(
            session_id=pl.col("visitorid").cast(pl.Utf8)
            + "_"
            + pl.col("session_per_visitor").cast(pl.Utf8),
        )
        .drop(["delta_ms", "prev_event", "is_new_session", "session_per_visitor"])
    )
    events_with_session.head(10)
    return (events_with_session,)


@app.cell
def _(events_with_date, events_with_session, pl):
    gap_only = (
        events_with_date.sort(["visitorid", "timestamp"])
        .with_columns(
            session_per_visitor=(
                (pl.col("timestamp").diff().cast(pl.Float64) / 60000 >= 30)
                .over("visitorid")
                .fill_null(False)
                .cast(pl.Int32)
                .cum_sum()
                .over("visitorid")
            ),
        )
        .with_columns(
            session_id_gap_only=pl.col("visitorid").cast(pl.Utf8)
            + "_"
            + pl.col("session_per_visitor").cast(pl.Utf8),
        )
    )

    pl.DataFrame(
        {
            "heuristic": ["Gap 30min only", "Gap 30min + purchase ends session"],
            "total_sessions": [
                gap_only["session_id_gap_only"].n_unique(),
                events_with_session["session_id"].n_unique(),
            ],
            "diff_from_gap_only": [
                0,
                events_with_session["session_id"].n_unique()
                - gap_only["session_id_gap_only"].n_unique(),
            ],
        }
    )
    return


@app.cell
def _(events_with_session, pl):
    session_stats = events_with_session.group_by("session_id").agg(
        [
            pl.len().alias("n_events"),
            (pl.col("datetime").max() - pl.col("datetime").min()).alias("duration"),
            pl.col("event").unique().alias("event_types"),
        ]
    )
    session_stats
    return (session_stats,)


@app.cell
def _(mo, plt, session_stats):
    fig_events, ax_events = plt.subplots()
    ax_events.hist(session_stats["n_events"], bins=50, log=True)
    ax_events.set_xlabel("Events per session")
    ax_events.set_ylabel("Number of sessions (log)")
    ax_events.set_title("Events per session distribution")
    mo.mpl.interactive(fig_events)
    return


@app.cell
def _(pl, session_stats):
    session_stats.select(
        [
            pl.col("n_events").median().alias("median"),
            pl.col("n_events").quantile(0.9).alias("p90"),
            pl.col("n_events").quantile(0.99).alias("p99"),
            pl.col("n_events").max().alias("max"),
        ]
    )
    return


@app.cell
def _(mo, pl, plt, session_stats):
    session_dur = session_stats.with_columns(
        (pl.col("duration").cast(pl.Int64) / 60_000_000).alias("duration_min"),
    ).filter(pl.col("n_events") > 1)

    fig_dur, ax_dur = plt.subplots()
    ax_dur.hist(session_dur["duration_min"], bins=50, log=True)
    ax_dur.set_xlabel("Session duration (min)")
    ax_dur.set_ylabel("Number of sessions (log)")
    ax_dur.set_title("Session duration (sessions with >1 event)")
    mo.mpl.interactive(fig_dur)
    return (session_dur,)


@app.cell
def _(pl, session_dur):
    pl.DataFrame(
        {
            "metric": ["median", "p90", "p99", "max"],
            "duration_min": [
                round(session_dur["duration_min"].median(), 1),
                round(session_dur["duration_min"].quantile(0.9), 1),
                round(session_dur["duration_min"].quantile(0.99), 1),
                round(session_dur["duration_min"].max(), 1),
            ],
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Gramática de `value` e Propriedades de Item
    """)
    return


@app.cell
def _(category_tree_path, item_prop_paths, pd):
    category_tree = pd.read_csv(category_tree_path)
    item_props_parts = [pd.read_csv(p) for p in item_prop_paths]
    item_properties = pd.concat(item_props_parts, ignore_index=True)
    item_properties["property_time"] = pd.to_datetime(
        item_properties["timestamp"], unit="ms", utc=True
    )
    item_properties["property"] = item_properties["property"].astype("string")
    item_properties["value"] = item_properties["value"].astype("string")

    shape_summary = pd.DataFrame(
        {
            "table": ["events", "category_tree", "item_properties"],
            "rows": [2756101, len(category_tree), len(item_properties)],
            "cols": [5, 2, 4],
        }
    )
    shape_summary
    return (item_properties, category_tree)


@app.cell
def _(item_properties, pd):
    import re

    NUMERIC_TOKEN_PATTERN = r"\bn-?\d+\.\d{3}\b"
    PLAIN_HASH_TOKEN_PATTERN = r"(?<![\w.])\d+(?![\w.])"

    value_frame = (
        item_properties[["itemid", "property", "value", "property_time"]]
        .sample(
            min(500_000, len(item_properties)),
            random_state=42,
        )
        .copy()
    )
    value_frame["value_text"] = value_frame["value"].fillna("").astype(str)
    value_frame["token_count"] = (
        value_frame["value_text"].str.split().str.len().fillna(0).astype(int)
    )
    value_frame["encoded_numeric_tokens"] = value_frame["value_text"].str.count(
        NUMERIC_TOKEN_PATTERN
    )
    value_frame["plain_hash_tokens"] = value_frame["value_text"].str.count(PLAIN_HASH_TOKEN_PATTERN)

    value_frame["is_categoryid"] = value_frame["property"].eq("categoryid")
    value_frame["is_available"] = value_frame["property"].eq("available")
    value_frame["is_multitoken"] = value_frame["token_count"].gt(1)

    value_frame["value_grammar"] = "other_or_unexpected"
    value_frame.loc[value_frame["is_categoryid"], "value_grammar"] = "special_categoryid"
    value_frame.loc[value_frame["is_available"], "value_grammar"] = "special_available"
    _regular = ~(value_frame["is_categoryid"] | value_frame["is_available"])
    value_frame.loc[
        _regular & value_frame["encoded_numeric_tokens"].eq(value_frame["token_count"]),
        "value_grammar",
    ] = "encoded_numeric_only"
    value_frame.loc[
        _regular
        & value_frame["plain_hash_tokens"].eq(value_frame["token_count"])
        & value_frame["token_count"].eq(1),
        "value_grammar",
    ] = "single_hashed_text_token"
    value_frame.loc[
        _regular
        & value_frame["plain_hash_tokens"].eq(value_frame["token_count"])
        & value_frame["token_count"].gt(1),
        "value_grammar",
    ] = "multi_hashed_text_tokens"
    value_frame.loc[
        _regular
        & value_frame["encoded_numeric_tokens"].gt(0)
        & value_frame["plain_hash_tokens"].gt(0),
        "value_grammar",
    ] = "mixed_hashed_text_and_numeric"

    grammar_summary = (
        value_frame.groupby("value_grammar")
        .agg(
            rows=("value_text", "size"),
            properties=("property", "nunique"),
            items=("itemid", "nunique"),
            avg_tokens=("token_count", "mean"),
        )
        .sort_values("rows", ascending=False)
    )
    grammar_summary["pct"] = (grammar_summary["rows"] / len(value_frame) * 100).round(2)
    grammar_summary["avg_tokens"] = grammar_summary["avg_tokens"].round(2)
    grammar_summary
    return (NUMERIC_TOKEN_PATTERN, PLAIN_HASH_TOKEN_PATTERN, value_frame, grammar_summary)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Propriedades especiais

    - **`categoryid`**: identificador de categoria diretamente usável, ligável a `category_tree`
    - **`available`**: flag 0/1 de disponibilidade, muda ao longo do tempo
    - **Propriedades numéricas** (ex: `790`): valor com prefixo `n`, decodificável para float
    - **Propriedades hashed**: candidatos a bag-of-tokens ou embeddings
    """)
    return


@app.cell
def _(category_tree, item_properties, pd, pl):
    _special = item_properties[item_properties["property"].isin(["categoryid", "available"])].copy()
    _cats = _special[_special["property"].eq("categoryid")].copy()
    _cats["categoryid_num"] = pd.to_numeric(_cats["value"], errors="coerce").astype("Int64")
    _cat_match = _cats["categoryid_num"].isin(category_tree["categoryid"])

    _avail = _special[_special["property"].eq("available")]
    _avail_dist = (
        _avail["value"].value_counts(dropna=False).rename_axis("available").to_frame("count")
    )
    _avail_dist["pct"] = (_avail_dist["count"] / _avail_dist["count"].sum() * 100).round(2)

    pl.DataFrame(
        {
            "metric": [
                "categoryid rows",
                "unique categories",
                "pct matching category_tree",
                "available rows",
                "available=0 pct",
                "available=1 pct",
            ],
            "value": [
                len(_cats),
                _cats["categoryid_num"].nunique(),
                round(_cat_match.mean() * 100, 2),
                len(_avail),
                _avail_dist.loc["0", "pct"] if "0" in _avail_dist.index else 0,
                _avail_dist.loc["1", "pct"] if "1" in _avail_dist.index else 0,
            ],
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Fluxo de Eventos e Super Tabela
    """)
    return


@app.cell
def _(events_path, pd):
    events_pd = pd.read_csv(events_path)
    events_pd["datetime"] = pd.to_datetime(events_pd["timestamp"], unit="ms", utc=True)
    events_pd
    return (events_pd,)


@app.cell
def _(events_pd, pd):
    EVENT_ORDER = ["view", "addtocart", "transaction"]

    flow_events = events_pd.sort_values(["visitorid", "datetime", "itemid"]).copy()
    flow_events["next_event"] = flow_events.groupby("visitorid")["event"].shift(-1)
    flow_events["next_itemid"] = flow_events.groupby("visitorid")["itemid"].shift(-1)
    flow_events["minutes_to_next"] = (
        (flow_events.groupby("visitorid")["datetime"].shift(-1) - flow_events["datetime"])
        .dt.total_seconds()
        .div(60)
    )

    transition_matrix = pd.crosstab(flow_events["event"], flow_events["next_event"])
    transition_matrix = (
        transition_matrix.reindex(index=EVENT_ORDER, columns=EVENT_ORDER).fillna(0).astype(int)
    )
    transition_rates = (
        transition_matrix.div(transition_matrix.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)
        * 100
    ).round(2)

    transition_matrix, transition_rates
    return (EVENT_ORDER, flow_events, transition_matrix, transition_rates)


@app.cell
def _(flow_events):
    _visitor_outcomes = flow_events.groupby("visitorid")["event"].agg(
        n_events="size",
        has_view=lambda x: int((x == "view").any()),
        has_addtocart=lambda x: int((x == "addtocart").any()),
        has_transaction=lambda x: int((x == "transaction").any()),
    )
    _visitor_outcomes["signature"] = (
        "view="
        + _visitor_outcomes["has_view"].astype(str)
        + " addtocart="
        + _visitor_outcomes["has_addtocart"].astype(str)
        + " transaction="
        + _visitor_outcomes["has_transaction"].astype(str)
    )
    _visitor_outcomes["signature"].value_counts().to_frame("visitors")
    return


@app.cell
def _(flow_events, pd):
    flow_events["minutes_since_previous"] = (
        (flow_events["datetime"] - flow_events.groupby("visitorid")["datetime"].shift(1))
        .dt.total_seconds()
        .div(60)
    )

    flow_events.groupby("event")["minutes_since_previous"].describe(
        percentiles=[0.5, 0.75, 0.9, 0.99],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature readiness e super tabela

    ~55% dos eventos não têm histórico do usuário. ~40% não têm histórico do item.
    Apenas ~34% têm ambos. Isso reforça a necessidade de sessões e cold-start handling.

    O join plan usa `merge_asof` backward para garantir zero leakage temporal.
    """)
    return


@app.cell
def _(events_pd, pd):
    feature_events = events_pd.sort_values(["datetime", "visitorid", "itemid"]).copy()
    feature_events.insert(0, "event_id", range(1, len(feature_events) + 1))
    feature_events["user_events_before"] = feature_events.groupby("visitorid").cumcount()
    feature_events["item_events_before"] = feature_events.groupby("itemid").cumcount()
    feature_events["has_user_history"] = feature_events["user_events_before"] > 0
    feature_events["has_item_history"] = feature_events["item_events_before"] > 0

    pd.DataFrame(
        {
            "metric": [
                "total events",
                "pct_no_user_history",
                "pct_no_item_history",
                "pct_both_histories",
                "p95_user_events_before",
                "p95_item_events_before",
            ],
            "value": [
                len(feature_events),
                round((~feature_events["has_user_history"]).mean() * 100, 2),
                round((~feature_events["has_item_history"]).mean() * 100, 2),
                round(
                    (feature_events["has_user_history"] & feature_events["has_item_history"]).mean()
                    * 100,
                    2,
                ),
                feature_events["user_events_before"].quantile(0.95),
                feature_events["item_events_before"].quantile(0.95),
            ],
        }
    )
    return (feature_events,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. Perfis comportamentais e cenarios de recomendacao

    A PR #26 trouxe um ponto importante: alem de entender o dataset, a EDA precisa apontar
    qual experiencia de recomendacao queremos viabilizar. Como nao existem dados demograficos
    do visitor, os perfis precisam ser derivados do proprio historico.
    """)
    return


@app.cell
def _(events_pd, pd):
    transaction_events = events_pd[events_pd["event"].eq("transaction")].copy()

    top_transaction_items = (
        transaction_events.groupby("itemid")
        .size()
        .rename("transactions")
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    top_buyers = (
        transaction_events.groupby("visitorid")
        .size()
        .rename("transactions")
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    top_transaction_items, top_buyers
    return top_buyers, top_transaction_items, transaction_events


@app.cell
def _(events_pd, pd):
    visitor_events = pd.crosstab(events_pd["visitorid"], events_pd["event"])
    for _event in ["view", "addtocart", "transaction"]:
        if _event not in visitor_events.columns:
            visitor_events[_event] = 0

    visitor_events["total_events"] = visitor_events[["view", "addtocart", "transaction"]].sum(
        axis=1
    )
    visitor_events["unique_items"] = events_pd.groupby("visitorid")["itemid"].nunique()
    visitor_events["profile"] = "cold_or_single_view"
    visitor_events.loc[visitor_events["total_events"].ge(5), "profile"] = "recurring_browser"
    visitor_events.loc[visitor_events["addtocart"].gt(0), "profile"] = "cart_intent"
    visitor_events.loc[visitor_events["transaction"].gt(0), "profile"] = "buyer"

    profile_summary = (
        visitor_events.groupby("profile")
        .agg(
            visitors=("total_events", "size"),
            avg_events=("total_events", "mean"),
            avg_unique_items=("unique_items", "mean"),
            avg_views=("view", "mean"),
            avg_carts=("addtocart", "mean"),
            avg_transactions=("transaction", "mean"),
        )
        .sort_values("visitors", ascending=False)
        .round(2)
    )

    profile_summary
    return profile_summary, visitor_events


@app.cell
def _(pd, transaction_events):
    transactions_by_hour = (
        transaction_events.assign(event_hour=transaction_events["datetime"].dt.hour)
        .groupby("event_hour")
        .size()
        .rename("transactions")
        .reset_index()
    )

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    transactions_by_weekday = (
        transaction_events.assign(
            event_dayofweek=transaction_events["datetime"].dt.day_name(),
        )
        .groupby("event_dayofweek")
        .size()
        .rename("transactions")
        .reindex(weekday_order)
        .reset_index()
    )

    transactions_by_hour, transactions_by_weekday
    return transactions_by_hour, transactions_by_weekday


@app.cell
def _(pd):
    recommendation_scenarios = pd.DataFrame(
        [
            (
                "Home / top 5 para visitor",
                "visitorid + perfil comportamental",
                "historico, categorias preferidas, recencia, popularidade e fallback",
                "top 5 itens com maior probabilidade de interacao forte",
            ),
            (
                "Pagina de produto",
                "item atual + visitor quando existir",
                "coocorrencia item-item, categoria, disponibilidade e sinais da sessao",
                "itens similares/complementares ao item exibido",
            ),
            (
                "Cold start",
                "sem historico suficiente",
                "best sellers, trending por periodo e categorias mais fortes",
                "recomendacao segura ate acumular historico",
            ),
        ],
        columns=["cenario", "entrada", "sinais", "saida_esperada"],
    )

    recommendation_scenarios
    return (recommendation_scenarios,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. Decisoes tecnicas e proximos passos

    | Decisao | Detalhe |
    |---------|---------|
    | **Sessão** | gap ≥ 30min OU (prev == transaction E cur != transaction). Transacoes consecutivas ficam na mesma sessao. |
    | **Split** | Cronologico 70/15/15. Dentro de cada particao, h=[s1..sm-1] vs t=sm. |
    | **Feedback** | Implicito: view=1, addtocart=3, transaction=5 |
    | **Perfis** | Derivados do historico: cold/single view, recurring browser, cart intent e buyer |
    | **Cenarios** | Top 5 na home, recomendacao em pagina de produto e fallback cold-start |
    | **Features diretas** | `categoryid`, `available`, property `790` (numerico) |
    | **Features hashed** | Bag-of-tokens apos filtro por cobertura |
    | **Join temporal** | `merge_asof backward` — zero leakage |
    | **Cold start** | Sessoes de 1 evento incluidas |
    """)
    return


if __name__ == "__main__":
    app.run()
