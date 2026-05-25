# web_main_content_extraction_multipage

GNN выделяет основной текст веб-страницы: классифицирует узлы DOM по признакам стабильности соседних страниц; работает и на одиночных страницах.

Модель размечает каждый узел HTML-документа как:
- `sure-main-content` — основной контент (статья, текст),
- `sure-template-content` — шаблонное окружение (хедер, футер, навигация, реклама).

## Структура репозитория

```
web_main_content_extraction_multipage/
├── README.md                       # этот файл
├── requirements.txt                # зависимости (torch, torch-geometric, fasttext, …)
├── download_fasttext.py            # скрипт для скачивания cc.en.300.bin (7.2 ГБ)
├── src/
│   ├── stability_v3_inference.py   # CLI инференса
│   ├── config.py                   # пути и гиперпараметры
│   ├── gnn_model.py                # архитектура BiDirSAGEStabV3
│   ├── sage_features.py            # извлечение признаков узлов
│   ├── sage_graph.py               # HTML → граф
│   ├── xpath_utils.py              # XPath / порядок узлов
│   ├── dataset.py                  # чтение HTML + наследование меток из классов
│   ├── annotate_dom.py             # разметка data-ml для сырых HTML (fallback)
│   ├── vectorize.py                # плоский вектор data-ml
│   └── text_extraction.py          # извлечение видимого текста узла
├── htmldiff/                       # пакет для вычисления stability-признаков
├── models/
│   ├── stability_v3_weak_gnn.pt    # чекпоинт лучшей модели (1.6 МБ)
│   ├── sage_preprocessors.pt       # scaler + label encoders + tag vocab (319 КБ)
│   └── scaler_stab.pkl             # scaler для stability-признаков
└── samples/
    ├── habr1/index.html            # основная страница для теста (Habr)
    ├── habr_siblings/              # соседние страницы того же домена
    │   ├── habr2.html
    │   └── habr3.html
    ├── dzen1/index.html            # второй пример (Дзен)
    └── dzen_siblings/
        ├── dzen2.html
        └── dzen3.html
```

## Установка

### 1. Клонирование и виртуальное окружение

```powershell
git clone https://github.com/Saelaress/web_main_content_extraction_multipage.git web_main_content_extraction_multipage
cd web_main_content_extraction_multipage

python -m venv venv
.\venv\Scripts\activate

python -m pip install -U pip
pip install -r requirements.txt
```

> Если `pip install torch-geometric` выдаёт ошибку из-за
> torch-scatter/torch-sparse, эти зависимости опциональны — `SAGEConv` работает
> без них на CPU. См. https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 2. Скачивание FastText (cc.en.300.bin, ~7.2 ГБ)

```powershell
python download_fasttext.py
```

Скрипт скачивает архив, распаковывает и кладёт в `models/cc.en.300.bin`.
`src/config.py` автоматически найдёт файл там.

Альтернатива — скачать вручную с https://fasttext.cc/docs/en/crawl-vectors.html
и положить как `models/cc.en.300.bin`, либо указать путь через переменную
окружения:

```powershell
$env:FASTTEXT_MODEL_PATH = "D:\path\to\cc.en.300.bin"
```

## Запуск инференса

### Минимальный пример

```powershell
python src\stability_v3_inference.py samples\habr1\index.html `
       --siblings-dir samples\habr_siblings `
       --format text `
       --output output\habr1_main.txt
```

По умолчанию применяются лучшие per-group пороги `0.64/0.75` (оптимум на val,
test F1<sub>main</sub> = 0.7696). Переопределить вручную — `--thresholds W,N`.
Флаг `--argmax` откатывает к argmax `0.5/0.5` (baseline F1<sub>main</sub> ≈ 0.7247).

### Форматы вывода

| `--format` | Что получается                                                       |
| ---------- | -------------------------------------------------------------------- |
| `text`     | Только основной текст (узлы с label = main, без шаблонов). **По умолчанию.** |
| `html`     | Размеченный HTML с CSS-классами `sure-main-content`/`sure-template-content`. |

### Без соседних страниц

Если у пользователя нет соседей того же домена — флаг `--siblings-dir` можно
опустить.

```powershell
python src\stability_v3_inference.py samples\habr1\index.html --format text
```

## Что внутри лучшей модели

**BiDirSAGEStabV3** — двунаправленный GraphSAGE поверх DOM-дерева с двумя
регуляризациями:

- **DropEdge** (p = 0.1): случайное удаление рёбер графа на тренировке.
- **Feature Dropout** (p = 0.2): зануление stability-блока признаков.

Признаки узла:
- Категориальные эмбеддинги тегов (`<tag>`, `<parent>`, `<grandparent>`).
- FastText-эмбеддинг видимого текста узла (300d → 64d).
- TF-IDF подобный вектор CSS-классов.
- 48 числовых признаков из data-ml схемы (структурные, текстовые, мета).
- 5 stability-признаков (рассчитываются через htmldiff между основной
  страницей и её соседями).

Per-group thresholds для классификации main:
- Узлы со stab-сигналом (`n_pages_compared > 0`): `thr = 0.64`.
- Узлы без stab-сигнала: `thr = 0.75`.

## Лицензия и атрибуция

Код инференса и модель — часть дипломной работы автора. FastText-веса
распространяются под Creative Commons Attribution-Share-Alike 3.0
(см. https://fasttext.cc/docs/en/crawl-vectors.html#license).
