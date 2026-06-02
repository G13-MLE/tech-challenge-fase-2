# EDA RetailRocket Ecommerce

## Resumo executivo
- Dataset local analisado: `D:\Dataset\archive`.
- Eventos observados: 2.756.101.
- Pares usuario-item unicos: 2.145.179.
- Requisito minimo de interacoes atendido: sim.
- Esparsidade estimada da matriz: 99.999352%.

## Inventario dos arquivos
| Arquivo | Linhas | MiB | Colunas |
| --- | --- | --- | --- |
| events.csv | 2.756.101 | 89.87 | timestamp, visitorid, event, itemid, transactionid |
| item_properties_part1.csv | 10.999.999 | 461.88 | timestamp, itemid, property, value |
| item_properties_part2.csv | 9.275.903 | 389.99 | timestamp, itemid, property, value |
| category_tree.csv | 1.669 | 0.01 | categoryid, parentid |

## Qualidade e schema
| Arquivo | Schema esperado | Ausentes | Duplicidades |
| --- | --- | --- | --- |
| events.csv | sim | transactionid=2.733.644 | 460 |
| item_properties_part1.csv | sim | sem ausentes | n/a |
| item_properties_part2.csv | sim | sem ausentes | n/a |
| category_tree.csv | sim | parentid=25 | 0 |
### Tipos esperados
| Arquivo | Coluna | Tipo |
| --- | --- | --- |
| events.csv | timestamp | unix_ms |
| events.csv | visitorid | integer_id |
| events.csv | event | category |
| events.csv | itemid | integer_id |
| events.csv | transactionid | nullable_integer_id |
| item_properties_part1.csv | timestamp | unix_ms |
| item_properties_part1.csv | itemid | integer_id |
| item_properties_part1.csv | property | category |
| item_properties_part1.csv | value | string |
| item_properties_part2.csv | timestamp | unix_ms |
| item_properties_part2.csv | itemid | integer_id |
| item_properties_part2.csv | property | category |
| item_properties_part2.csv | value | string |
| category_tree.csv | categoryid | integer_id |
| category_tree.csv | parentid | nullable_integer_id |
### Inconsistencias e observacoes
- Todos os arquivos possuem as colunas esperadas.
- `transactionid` vazio em 2.733.644 eventos sem compra.
- `parentid` vazio em 25 categorias raiz.
- Remover 460 duplicidades exatas de events.csv no preprocess.
- Duplicidades exatas nao sao rastreadas nos arquivos de propriedades.

## Eventos e comportamento
| Evento | Linhas |
| --- | --- |
| addtocart | 69.332 |
| transaction | 22.457 |
| view | 2.664.312 |
- Usuarios unicos: 1.407.580.
- Itens unicos em eventos: 235.061.
- Periodo: 2015-05-03 a 2015-09-18.
- Duplicidades exatas em events.csv: 460.
### Distribuicao temporal mensal
| Mes | Eventos |
| --- | --- |
| 2015-05 | 590.652 |
| 2015-06 | 610.393 |
| 2015-07 | 697.984 |
| 2015-08 | 553.362 |
| 2015-09 | 303.710 |

## Propriedades de itens
- Linhas totais: 20.275.902.
- Itens com metadados: 417.053.
- Propriedades unicas: 1.104.
- Linhas `categoryid`: 788.214.
- Itens com `categoryid`: 417.053.
- Periodo das propriedades: 2015-05-10 a 2015-09-13.
- Nota sobre duplicidades: Duplicidades exatas nao sao rastreadas nos arquivos de propriedades.
### Top propriedades
| Propriedade | Linhas |
| --- | --- |
| 888 | 3.000.398 |
| 790 | 1.790.516 |
| available | 1.503.639 |
| categoryid | 788.214 |
| 6 | 631.471 |
| 283 | 597.419 |
| 776 | 574.220 |
| 678 | 481.966 |
| 364 | 476.486 |
| 202 | 448.938 |

## Arvore de categorias
- Categorias: 1.669.
- Categorias pai unicas: 362.
- Categorias raiz: 25.
- Duplicidades exatas em category_tree.csv: 0.

## Features candidatas
- `visitorid` e `itemid`: chaves para embeddings e matriz implicita.
- `event`: sinal implicito com pesos view=1, addtocart=3 e transaction=5.
- `categoryid`: feature categorica para 417.053 itens.
- `available`: disponibilidade historica antes do evento.
- `timestamp`: recencia, mes e janelas temporais para split sem vazamento.
- Propriedades frequentes (888, 790, available) podem virar metadados esparsos.

## Decisoes recomendadas
- Usar split cronologico para evitar vazamento entre treino e avaliacao.
- Usar corte de treino em 2015-08-02 e validacao em 2015-08-25.
- Mapear eventos para pesos implicitos: view=1, addtocart=3, transaction=5.
- Usar categoryid como primeira feature de item.
- Tratar demais propriedades como metadados esparsos apos filtro de cardinalidade.
- Versionar CSVs brutos com DVC, sem commit direto no Git.

## Riscos e proximos passos
- A matriz usuario-item e extremamente esparsa; avaliar metricas top-k.
- Propriedades de itens sao historicas e volumosas; filtrar antes de criar features.
- Separar validacao/teste no tempo para simular recomendacao em producao.
