"""Carga e limpeza dos dados brutos do Retail Rocket.

Le events.csv, item_properties e category_tree, aplica conversoes
de tipos e filtros basicos, e salva parquets processados.
"""

from pathlib import Path

import polars as pl


def load_events(path: str | Path) -> pl.DataFrame:
    """Carrega events.csv com tipos corretos e coluna datetime.

    Args:
        path: Caminho para events.csv.

    Returns:
        DataFrame com colunas timestamp, visitorid, event, itemid,
        transactionid, datetime.
    """
    return (
        pl.read_csv(path)
        .with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("datetime"),
        )
        .cast({"visitorid": pl.Int64, "itemid": pl.Int64})
    )


def load_item_properties(paths: list[str | Path]) -> pl.DataFrame:
    """Carrega e concatena item_properties_part1 e part2.

    Converte timestamp para datetime e identifica propriedades especiais
    (categoryid, available, propriedades numericas com prefixo 'n').
    """
    dfs = [pl.read_csv(p) for p in paths]
    df = pl.concat(dfs)
    return df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("property_time"),
    )


def load_category_tree(path: str | Path) -> pl.DataFrame:
    """Carrega category_tree.csv com categoryid e parentid."""
    return pl.read_csv(path).cast({"categoryid": pl.Int64, "parentid": pl.Int64})


def filter_relevant_properties(
    item_props: pl.DataFrame,
    extra_numeric_props: list[str] | None = None,
    top_n_hashes: int = 50,
) -> pl.DataFrame:
    """Filtra item_properties para manter apenas propriedades relevantes.

    Mantem:
    - categoryid e available (diretamente uteis)
    - Propriedades numericas com prefixo 'n' (default: property 790)
    - Top-N propriedades hashed por cobertura (numero de itens distintos)

    Args:
        item_props: DataFrame completo de item_properties.
        extra_numeric_props: IDs de propriedades numericas para manter.
        top_n_hashes: Numero de propriedades hashed para manter por cobertura.

    Returns:
        DataFrame filtrado.
    """
    special_props = {"categoryid", "available"}
    if extra_numeric_props is None:
        extra_numeric_props = ["790"]

    special_ids = special_props | set(extra_numeric_props)

    prop_coverage = (
        item_props.group_by("property")
        .agg(pl.col("itemid").n_unique().alias("n_items"))
        .sort("n_items", descending=True)
    )

    top_hashed = set(
        prop_coverage.filter(~pl.col("property").is_in(list(special_ids)))
        .head(top_n_hashes)["property"]
        .to_list()
    )

    keep = special_ids | top_hashed
    return item_props.filter(pl.col("property").is_in(list(keep)))


def decode_numeric_values(
    df: pl.DataFrame, property_col: str = "property", value_col: str = "value"
) -> pl.DataFrame:
    """Decodifica valores com prefixo 'n' para floats.

    Propriedades como '790' tem valores no formato 'n277.200'.
    Esta funcao cria uma coluna 'value_numeric' com o float decodificado,
    mantendo o valor original como string.
    """
    return df.with_columns(
        pl.when(pl.col(value_col).str.starts_with("n"))
        .then(pl.col(value_col).str.slice(1).cast(pl.Float64, strict=False))
        .otherwise(None)
        .alias("value_numeric"),
    )


def save_processed(df: pl.DataFrame, path: str | Path) -> Path:
    """Salva DataFrame processado como parquet.

    Args:
        df: DataFrame para salvar.
        path: Caminho de destino (sem extensao ou com .parquet).

    Returns:
        Path do arquivo salvo.
    """
    path = Path(path)
    if not path.suffix == ".parquet":
        path = path.with_suffix(".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path
