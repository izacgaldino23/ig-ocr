# Fluxo de dados

## Inferência: imagem → texto

```mermaid
flowchart TD
    img([imagem]) --> find{find_page}
    find -- página numa foto --> warp[desentorta a página]
    find -- já é uma página --> otsu
    warp --> otsu[binarização Otsu<br/>invertida para texto escuro]
    otsu --> cont[contornos RETR_CCOMP<br/>descarta furo e moldura h > 15%]
    cont --> grp[agrupa em linhas por sobreposição vertical]
    grp --> eh{é texto?}
    eh -- não --> drop[descarta: thin ou empty]
    eh -- sim --> split[separa linha fundida<br/>vale de tinta ou gutter de célula]
    split --> crop[recorte da linha<br/>+30% de altura à direita]
    crop --> crnn[CRNN, uma linha por vez<br/>tensor de 48 px de altura]
    crnn --> para[agrupa parágrafos pelo passo vertical]
    para --> txt([texto])
```

Cada descarte tem um motivo, impresso por `python -m src.ocr.recognize <imagem> --debug`:
`frame`, `thin`, `empty`.

O `+30%` à direita existe porque o filtro de ruído do segmentador descarta o
blob do ponto final; sem ele a linha termina antes do ponto.

## Tela de depuração: upload → duas etapas

```mermaid
flowchart LR
    up([upload]) --> prep[exif_transpose, cinza<br/>máx. 2500 px]
    prep --> a1["POST /api/analyze"]
    prep --> a2["POST /api/recognize"]
    a1 --> seg1[find_page + segment_page]
    seg1 --> ov[overlay com as linhas<br/>e o número de cada uma]
    a2 --> seg2[find_page + segment_page]
    seg2 --> rec[CRNN por linha<br/>sob um lock, na CPU]
    rec --> corr[corretor de acentos e caixa<br/>opcional]
    corr --> resp[crop, texto e parágrafo por linha]
```

As duas rotas segmentam de novo em vez de receber as caixas do cliente, então o
servidor não guarda estado entre etapas. O modelo carrega uma vez no `lifespan`;
o nome vem da variável `OCR_CHECKPOINT`, que é obrigatória.

## Treino: corpus → pesos

```mermaid
flowchart LR
    corpus[(floresta)] --> frases[frases]
    frases --> render[PageGenerator.render]
    render --> page[imagem da página + clues]
    page --> cortes[cortes de linha<br/>rótulo no ALPHABET<br/>bbox fora da página descartada]
    cortes --> shards[(shards<br/>+ font_usage.csv)]
    shards --> treino[train.py]
```

O `font_usage.csv` guarda o `font_id` de cada amostra; sem ele o
`compare_font_errors.py` não consegue abrir o shard.

Um corte cuja bbox passa da borda da página é descartado: o Pillow preencheria
a área faltante com preto e o rótulo continuaria com o texto inteiro.

## Onde fica cada etapa

| etapa | arquivo |
|---|---|
| página na foto, binarização, contornos, filtros, linhas | `src/ocr/analyzer.py` |
| recorte da linha, CRNN, parágrafos, texto | `src/ocr/recognize.py` |
| corretor de acentos e caixa | `src/ocr/postprocess.py` |
| página sintética e cortes de treino | `src/datagen/` |
| tela de depuração | `src/web/app.py` |
| limiares medidos nos documentos reais | `src/ocr/analyzer.py` (topo do módulo) |
