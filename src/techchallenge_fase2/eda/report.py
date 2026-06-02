"""Markdown report rendering for RetailRocket EDA."""

from __future__ import annotations

from typing import Any


def build_retailrocket_report(metrics: dict[str, Any]) -> str:
    """Render RetailRocket EDA metrics as a Markdown report.

    Args:
        metrics: Serializable metrics produced by RetailRocketAnalyzer.

    Returns:
        Markdown report ready to be versioned.
    """
    sections = [
        "# EDA RetailRocket Ecommerce",
        _executive_summary(metrics),
        _inventory(metrics),
        _quality(metrics),
        _events(metrics),
        _properties(metrics),
        _categories(metrics),
        _candidate_features(metrics),
        _risks_and_decisions(metrics),
    ]
    return "\n\n".join(sections) + "\n"


def _executive_summary(metrics: dict[str, Any]) -> str:
    check = metrics["minimum_interactions"]
    events = metrics["events"]
    return "\n".join(
        [
            "## Resumo executivo",
            f"- Dataset local analisado: `{metrics['dataset']['local_path']}`.",
            f"- Eventos observados: {_int(events['row_count'])}.",
            f"- Pares usuario-item unicos: {_int(events['unique_user_item_pairs'])}.",
            f"- Requisito minimo de interacoes atendido: {_yes_no(check['passes'])}.",
            f"- Esparsidade estimada da matriz: {_pct(events['sparsity'])}.",
        ],
    )


def _inventory(metrics: dict[str, Any]) -> str:
    rows = [_file_row(file_data) for file_data in _file_profiles(metrics)]
    headers = ["Arquivo", "Linhas", "MiB", "Colunas"]
    return "\n".join(["## Inventario dos arquivos", _table(headers, rows)])


def _quality(metrics: dict[str, Any]) -> str:
    rows = [_quality_row(file_data) for file_data in _file_profiles(metrics)]
    return "\n".join(
        [
            "## Qualidade e schema",
            _table(["Arquivo", "Schema esperado", "Ausentes", "Duplicidades"], rows),
            _column_types(metrics),
            _inconsistencies(metrics),
        ],
    )


def _events(metrics: dict[str, Any]) -> str:
    events = metrics["events"]
    lines = [
        "## Eventos e comportamento",
        _event_counts(events),
        f"- Usuarios unicos: {_int(events['unique_visitors'])}.",
        f"- Itens unicos em eventos: {_int(events['unique_items'])}.",
        f"- Periodo: {events['date_range']['start']} a {events['date_range']['end']}.",
        f"- Duplicidades exatas em events.csv: {_int(events['duplicate_rows'])}.",
        _monthly_counts(events),
    ]
    return "\n".join(lines)


def _properties(metrics: dict[str, Any]) -> str:
    properties = metrics["item_properties"]
    date_range = properties["date_range"]
    return "\n".join(
        [
            "## Propriedades de itens",
            f"- Linhas totais: {_int(properties['row_count'])}.",
            f"- Itens com metadados: {_int(properties['unique_items'])}.",
            f"- Propriedades unicas: {_int(properties['unique_properties'])}.",
            f"- Linhas `categoryid`: {_int(properties['categoryid_rows'])}.",
            f"- Itens com `categoryid`: {_int(properties['items_with_categoryid'])}.",
            f"- Periodo das propriedades: {date_range['start']} a {date_range['end']}.",
            f"- Nota sobre duplicidades: {properties['duplicate_note']}",
            _top_properties(properties),
        ],
    )


def _categories(metrics: dict[str, Any]) -> str:
    categories = metrics["category_tree"]
    return "\n".join(
        [
            "## Arvore de categorias",
            f"- Categorias: {_int(categories['unique_categories'])}.",
            f"- Categorias pai unicas: {_int(categories['unique_parent_categories'])}.",
            f"- Categorias raiz: {_int(categories['root_categories'])}.",
            f"- Duplicidades exatas em category_tree.csv: {_int(categories['duplicate_rows'])}.",
        ],
    )


def _candidate_features(metrics: dict[str, Any]) -> str:
    properties = metrics["item_properties"]
    category_items = _int(properties["items_with_categoryid"])
    top_properties = _top_property_ids(properties)
    return "\n".join(
        [
            "## Features candidatas",
            "- `visitorid` e `itemid`: chaves para embeddings e matriz implicita.",
            "- `event`: sinal implicito com pesos view=1, addtocart=3 e transaction=5.",
            f"- `categoryid`: feature categorica para {category_items} itens.",
            "- `available`: disponibilidade historica antes do evento.",
            "- `timestamp`: recencia, mes e janelas temporais para split sem vazamento.",
            f"- Propriedades frequentes ({top_properties}) podem virar metadados esparsos.",
        ],
    )


def _risks_and_decisions(metrics: dict[str, Any]) -> str:
    recommendations = "\n".join(f"- {item}" for item in metrics["recommendations"])
    return "\n".join(
        [
            "## Decisoes recomendadas",
            recommendations,
            "",
            "## Riscos e proximos passos",
            "- A matriz usuario-item e extremamente esparsa; avaliar metricas top-k.",
            "- Propriedades de itens sao historicas e volumosas; filtrar antes de criar features.",
            "- Separar validacao/teste no tempo para simular recomendacao em producao.",
        ],
    )


def _event_counts(events: dict[str, Any]) -> str:
    rows = [(event, _int(count)) for event, count in sorted(events["event_counts"].items())]
    return _table(["Evento", "Linhas"], rows)


def _monthly_counts(events: dict[str, Any]) -> str:
    rows = [(month, _int(count)) for month, count in events["monthly_counts"].items()]
    return "\n".join(["### Distribuicao temporal mensal", _table(["Mes", "Eventos"], rows)])


def _top_properties(properties: dict[str, Any]) -> str:
    rows = [(item["id"], _int(item["count"])) for item in properties["top_properties"]]
    return "\n".join(["### Top propriedades", _table(["Propriedade", "Linhas"], rows)])


def _file_profiles(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = [metrics["events"]]
    profiles.extend(metrics["item_properties"]["files"])
    profiles.append(metrics["category_tree"])
    return profiles


def _file_row(file_data: dict[str, Any]) -> tuple[str, str, str, str]:
    columns = ", ".join(file_data["columns"])
    return (
        file_data["file_name"],
        _int(file_data["row_count"]),
        f"{file_data['file_size_bytes'] / 1024 / 1024:.2f}",
        columns,
    )


def _quality_row(file_data: dict[str, Any]) -> tuple[str, str, str, str]:
    schema_ok = _yes_no(file_data["schema"]["matches_expected"])
    missing_values = _missing_summary(file_data["missing_values"])
    return (
        file_data["file_name"],
        schema_ok,
        missing_values,
        _int(file_data["duplicate_rows"]),
    )


def _column_types(metrics: dict[str, Any]) -> str:
    rows = _column_type_rows(metrics)
    return "\n".join(["### Tipos esperados", _table(["Arquivo", "Coluna", "Tipo"], rows)])


def _column_type_rows(metrics: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows = []
    for file_data in _file_profiles(metrics):
        rows.extend(_type_rows(file_data))
    return rows


def _type_rows(file_data: dict[str, Any]) -> list[tuple[str, str, str]]:
    file_name = file_data["file_name"]
    types = file_data["expected_column_types"]
    return [(file_name, column, column_type) for column, column_type in types.items()]


def _inconsistencies(metrics: dict[str, Any]) -> str:
    lines = ["### Inconsistencias e observacoes"]
    lines.extend(_schema_findings(metrics))
    lines.extend(_missing_findings(metrics))
    lines.extend(_duplicate_findings(metrics))
    return "\n".join(lines)


def _schema_findings(metrics: dict[str, Any]) -> list[str]:
    findings = []
    for file_data in _file_profiles(metrics):
        schema = file_data["schema"]
        if not schema["matches_expected"]:
            findings.append(f"- {file_data['file_name']} diverge das colunas esperadas.")
    return findings or ["- Todos os arquivos possuem as colunas esperadas."]


def _missing_findings(metrics: dict[str, Any]) -> list[str]:
    events = metrics["events"]
    categories = metrics["category_tree"]
    transaction_missing = _int(events["missing_values"]["transactionid"])
    root_categories = _int(categories["missing_values"]["parentid"])
    return [
        f"- `transactionid` vazio em {transaction_missing} eventos sem compra.",
        f"- `parentid` vazio em {root_categories} categorias raiz.",
    ]


def _duplicate_findings(metrics: dict[str, Any]) -> list[str]:
    events = metrics["events"]
    properties = metrics["item_properties"]
    event_duplicates = _int(events["duplicate_rows"])
    return [
        f"- Remover {event_duplicates} duplicidades exatas de events.csv no preprocess.",
        f"- {properties['duplicate_note']}",
    ]


def _missing_summary(missing_values: dict[str, int]) -> str:
    present = [(column, count) for column, count in missing_values.items() if count]
    if not present:
        return "sem ausentes"
    return ", ".join(f"{column}={_int(count)}" for column, count in present)


def _top_property_ids(properties: dict[str, Any]) -> str:
    top_ids = [item["id"] for item in properties["top_properties"][:3]]
    return ", ".join(top_ids)


def _table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}".replace(",", ".")


def _pct(value: float) -> str:
    return f"{value:.6%}"


def _yes_no(value: bool) -> str:
    return "sim" if value else "nao"
