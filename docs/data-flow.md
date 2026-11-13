# Fluxo de dados

## Inferência: imagem → texto

```mermaid
flowchart TD
    img([imagem]) --> find{find_page}
    find -- página numa foto --> warp[desentorta a página]
    find -- já é uma página --> seg
    warp --> seg[segment_page]
    seg --> otsu[binarização Otsu]
    otsu --> cont[contornos RETR_CCOMP]
    cont --> f1[descarta moldura: h > 15% da página]
    f1 --> grp[agrupa em linhas por sobreposição vertical]
    grp --> f2[descarta linha fina ou vazia]
    f2 --> f3[separa linha fundida: vale de tinta ou gutter de célula]
    f3 --> crnn[CRNN, uma linha por vez]
    crnn --> para[agrupa parágrafos]
    para --> txt([texto])
```

Cada descarte tem um motivo, impresso por `python -m src.ocr.recognize <imagem> --debug`:
`frame`, `thin`, `empty`.

## Treino: corpus → pesos

```mermaid
flowchart LR
    corpus[(floresta)] --> frases[frases]
    frases --> render[PageGenerator.render]
    render --> page[imagem da página + clues]
    page --> cortes[cortes de linha]
    cortes --> shards[(shards)]
    shards --> treino[train.py]
```

## Onde fica cada etapa

| etapa | arquivo |
|---|---|
| página na foto, binarização, contornos, filtros, linhas | `src/ocr/analyzer.py` |
| recorte da linha, CRNN, parágrafos, texto | `src/ocr/recognize.py` |
| corretor de acentos e caixa | `src/ocr/postprocess.py` |
| página sintética e cortes de treino | `src/datagen/` |
| tela de depuração | `src/web/app.py` |
