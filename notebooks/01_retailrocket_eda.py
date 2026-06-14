# /// script
# dependencies = [
#     "altair==6.2.1",
#     "marimo",
#     "matplotlib==3.10.9",
#     "polars==1.41.2",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    alt.data_transformers.enable("default")
    return mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Overview Base Eventos
    """)
    return


@app.cell
def _(pl):
    events = pl.read_csv("data/raw/events.csv")
    events
    return (events,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Abaixo rodamos um describe do dataset, que basicamente não ajuda muito, pois apesar da amioria das colunas como timestamp, visitorid e itemid serem considerados colunas numéricas, elas na realidade são IDs e estatísticas descritivas não trazem muita informação, alem disso o timestamp deve ser tratado melhor.
    """)
    return


@app.cell
def _(events):
    events.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Tratando timestamp para Date
    """)
    return


@app.cell
def _(events, pl):
    events_with_date = events.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime")
    )
    events_with_date
    return (events_with_date,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3. Checagem de Nulos
    """)
    return


@app.cell
def _(events, events_with_date, pl):
    null_counts = pl.DataFrame({
        "column": events_with_date.columns,
        "null_count": [events_with_date[col].null_count() for col in events_with_date.columns],
        "null_pct": [f"{events_with_date[col].null_count() / len(events) * 100:.2f}%" for col in events_with_date.columns],
    })
    null_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tabela abaixo nos mostra que apenas observacoes com evento de transacao, apresentam transaction ID.
    """)
    return


@app.cell
def _(events_with_date, pl):
    (
      events_with_date
       .group_by("event", pl.col("transactionid").is_null())
       .agg(pl.len())
       .sort("len", descending= True)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Verificar quantas vezes um mesmo visitorid aparece na base
    """)
    return


@app.cell
def _(events):
    events["visitorid"].value_counts().sort(by="count", descending=True)
    return


@app.cell
def _(events, mo):
    import matplotlib.pyplot as plt

    _visitor_counts = events["visitorid"].value_counts()
    fig, ax = plt.subplots()
    ax.hist(_visitor_counts["count"], bins=50, log=True)
    ax.set_xlabel("Events per visitor")
    ax.set_ylabel("Number of visitors (log scale)")
    ax.set_title("Distribution of events per visitor")
    mo.mpl.interactive(fig)
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Na tabela abaixo e possivel perceber que ate 90% do visitors ID occorem tem ate 3 iteracoes, e 99% ate 13. Existem IDs extremamente outliers
    """)
    return


@app.cell
def _(events, pl):
    _vc = events["visitorid"].value_counts()["count"]
    _support = pl.DataFrame({
        "metric": ["median", "p90", "p99", "max"],
        "value": [_vc.median(), _vc.quantile(0.9), _vc.quantile(0.99), _vc.max()],
    })
    _support
    return


@app.cell
def _(events_with_date, mo, pl):
    _total_events = events_with_date.height
    _unique_visitors = events_with_date["visitorid"].n_unique()
    _single_event_visitors = events_with_date.group_by("visitorid").agg(pl.len().alias("n")).filter(pl.col("n") == 1).height
    _pct_single = _single_event_visitors / _unique_visitors * 100

    mo.md(f"""
    # 5. Por que criar sessões?

    O dataset tem **{_total_events:,} eventos** de **{_unique_visitors:,} visitors**
    — mas **{_pct_single:.1f}% dos visitors aparecem apenas 1 vez**.

    Sem agrupar os eventos em sessões, o sistema de recomendação não consegue capturar
    padrões de navegação dentro de uma mesma visita.

    Uma **sessão** é uma sequência contínua de atividade de um mesmo visitor.
    Se o gap entre dois eventos consecutivos excede um threshold, iniciamos uma nova sessão.
    O threshold padrão da indústria (ex: Google Analytics) é **30 minutos**.

    Antes de definir a heurística, vamos analisar a distribuição dos gaps entre eventos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Distribuição dos tipos de evento
    """)
    return


@app.cell
def _(events_with_date, pl):
    events_with_date["event"].value_counts().with_columns(
        (pl.col("count") / pl.col("count").sum() * 100).round(1).alias("pct")
    ).sort("count", descending=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Análise dos gaps entre eventos (mesmo visitor)

    Para cada visitor, calculamos o tempo entre eventos consecutivos.
    A distribuição mostra onde estão os "cortes naturais" para definir sessões.
    """)
    return


@app.cell
def _(events_with_date, pl):
    sorted_events = events_with_date.sort(["visitorid", "timestamp"])
    deltas = sorted_events.group_by("visitorid").agg(
        (pl.col("timestamp").diff().cast(pl.Float64) / 60000).alias("delta_min")
    )
    exploded_deltas = deltas.explode("delta_min").filter(pl.col("delta_min").is_not_null())

    gap_dist = exploded_deltas.with_columns(
        pl.col("delta_min").cut(
            breaks=[5, 10, 30, 60, 120, 1440, 10080, 43200]
        ).alias("gap_bin")
    ).group_by("gap_bin").agg(pl.len().alias("count")).sort("gap_bin").with_columns(
        (pl.col("count") / pl.col("count").sum() * 100).round(1).alias("pct")
    )
    gap_dist
    return


@app.cell
def _(mo):
    mo.md(r"""
    A maioria dos gaps (~61%) é inferior a 5 minutos — atividade bursty típica de navegação.
    O salto significativo está nos gaps >2h, que representam ~22% dos gaps.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Simulação: quantas sessões teríamos?

    Testamos dois thresholds para o gap máximo dentro de uma sessão:

    - **30 min** → ~1.761.676 sessões (padrão Google Analytics)
    - **60 min** → ~1.726.714 sessões

    A diferença é de apenas ~35 mil sessões. Isso acontece porque a maior parte
    dos gaps está abaixo de 30min ou acima de 2h — poucos gaps caem entre 30–60min.

    **Decisão: threshold de 30 minutos.** Alinha com o padrão da indústria e
    captura bem os agrupamentos naturais de navegação.
    """)
    return


@app.cell
def _(events_with_session, pl):
    session_stats = events_with_session.group_by("session_id").agg([
        pl.len().alias("n_events"),
        (pl.col("datetime").max() - pl.col("datetime").min()).alias("duration"),
        pl.col("event").unique().alias("event_types"),
    ])
    return (session_stats,)


@app.cell
def _(mo):
    mo.md(r"""
    # 6. Criando session_id

    Heurística: para cada `visitorid`, ordenamos por `timestamp`. Uma nova sessão inicia quando:

    1. O gap entre eventos consecutivos é **>= 30 minutos**, OU
    2. O evento anterior é uma **transaction** (compra finaliza a sessão)

    Abaixo comparamos o impacto de cada heurística.
    """)
    return


@app.cell
def _(events_with_date, pl):
    events_with_session = (
        events_with_date
        .sort(["visitorid", "timestamp"])
        .with_columns(
            pl.col("timestamp").diff().cast(pl.Float64).over("visitorid").alias("delta_ms"),
            pl.col("event").shift(1).over("visitorid").alias("prev_event"),
        )
        .with_columns(
            (
                (pl.col("delta_ms") / 60000 >= 30).fill_null(True)
                | (pl.col("prev_event") == "transaction")
            ).alias("is_new_session")
        )
        .with_columns(
            session_per_visitor = pl.col("is_new_session").cast(pl.Int32).cum_sum().over("visitorid")
        )
        .with_columns(
            session_id = pl.col("visitorid").cast(pl.Utf8) + "_" + pl.col("session_per_visitor").cast(pl.Utf8)
        )
        .drop(["delta_ms", "prev_event", "is_new_session", "session_per_visitor"])
    )
    events_with_session.head(10)
    return (events_with_session,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Distribuição de eventos por sessão
    """)
    return


@app.cell
def _(mo, plt, session_stats):
    fig_sess_events, ax_sess_events = plt.subplots()
    ax_sess_events.hist(session_stats["n_events"], bins=50, log=True)
    ax_sess_events.set_xlabel("Events per session")
    ax_sess_events.set_ylabel("Number of sessions (log)")
    ax_sess_events.set_title("Distribution of events per session")
    mo.mpl.interactive(fig_sess_events)
    return


@app.cell
def _(pl, session_stats):
    session_stats.select([
        pl.col("n_events").median().alias("median"),
        pl.col("n_events").quantile(0.9).alias("p90"),
        pl.col("n_events").quantile(0.99).alias("p99"),
        pl.col("n_events").max().alias("max"),
    ])
    return


@app.cell
def _(mo, pl, plt, session_stats):
    session_dur = session_stats.with_columns(
        (pl.col("duration").cast(pl.Int64) / 60_000_000).alias("duration_min")
    ).filter(pl.col("n_events") > 1)

    fig_dur, ax_dur = plt.subplots()
    ax_dur.hist(session_dur["duration_min"], bins=50, log=True)
    ax_dur.set_xlabel("Session duration (minutes)")
    ax_dur.set_ylabel("Number of sessions (log)")
    ax_dur.set_title("Session duration distribution")
    mo.mpl.interactive(fig_dur)
    return (session_dur,)


@app.cell
def _(pl, session_dur):
    pl.DataFrame({
        "metric": ["median", "p90", "p99", "max"],
        "duration_min": [
            round(session_dur["duration_min"].median(), 1),
            round(session_dur["duration_min"].quantile(0.9), 1),
            round(session_dur["duration_min"].quantile(0.99), 1),
            round(session_dur["duration_min"].max(), 1),
        ],
    })
    return


@app.cell
def _(events_with_date, events_with_session, pl):
    gap_only_sessions = (
        events_with_date
        .sort(["visitorid", "timestamp"])
        .with_columns(
            session_per_visitor = (
                (pl.col("timestamp").diff().cast(pl.Float64) / 60000 >= 30)
                .over("visitorid")
                .fill_null(False)
                .cast(pl.Int32)
                .cum_sum()
                .over("visitorid")
            )
        )
        .with_columns(
            session_id_gap_only = pl.col("visitorid").cast(pl.Utf8) + "_" + pl.col("session_per_visitor").cast(pl.Utf8)
        )
    )

    pl.DataFrame({
        "heuristic": ["Gap 30min only", "Gap 30min + purchase ends session"],
        "total_sessions": [
            gap_only_sessions["session_id_gap_only"].n_unique(),
            events_with_session["session_id"].n_unique(),
        ],
        "pct_increase": [
            0,
            events_with_session["session_id"].n_unique() - gap_only_sessions["session_id_gap_only"].n_unique(),
        ],
    })
    return


if __name__ == "__main__":
    app.run()
