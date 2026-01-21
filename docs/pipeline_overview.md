# Marker: как устроен конвейер PDF → Markdown/JSON/HTML

> Этот документ объясняет, как главные компоненты проекта взаимодействуют между собой. Он рассчитан на разработчика, который знаком с Python, но ещё не работал с Marker/Surya.

## 1. Путь данных от CLI до Markdown

1. **CLI (convert_single.py / convert.py)**
   - Параметры командной строки парсятся через `ConfigParser`.
   - Создаётся словарь моделей (`create_model_dict()`), где лежат весовые файлы Surya и вспомогательные артефакты (см. подробности ниже).
   - Выбирается конвертер (обычно `PdfConverter`) и renderer (по умолчанию Markdown).
2. **PdfConverter**
   - Получает `artifact_dict`, список процессоров, renderer и LLM-сервис.
   - Строит `Document` через `DocumentBuilder` и прогоняет все процессоры в `default_processors`.
   - По завершении отдаёт renderer-у готовый `Document`.
3. **Renderer (MarkdownRenderer/HTMLRenderer/JSONRenderer)**
   - Собирает дерево блоков в нужный формат. MarkdownRenderer дополнительно превращает HTML в Markdown (через Markdownify) и возвращает `MarkdownOutput` (markdown + изображения + metadata).
4. **(Опционально) Перевод Markdown**
   - Если флаг `--translate` задан, `translate_rendered_output()` разбивает текст на чанки и отправляет их в выбранный LLM-сервис (Ollama, Gemini и т.п.).
   - Результат сохраняется в `MarkdownOutput.markdown`, а в `metadata` добавляется информация о переводе.

### 1.1 ConfigParser и CLI

`ConfigParser.common_options` — это декоратор, который «навешивает» на CLI стандартный набор опций (output_dir, processors, translate, llm_service и т.д.). Внутри используется Click:

```python
@staticmethod
def common_options(fn):
    fn = click.option("--output_dir", ...)(fn)
    ...
    return fn
```

Когда в `convert_single.py` мы объявляем функцию CLI:

```python
@ConfigParser.common_options
def convert_single_cli(fpath: str, **kwargs):
    ...
```

Click автоматически генерирует аргументы. После парсинга `ConfigParser(kwargs)` предоставляет:

- `generate_config_dict()` — сводит flag-и в json-подобный словарь (включая `debug` режим, `page_range`, `config_json`).
- `get_processors()` — возвращает список классов процессоров, если пользователь передал `--processors`.
- `get_llm_service()` — по флагам `--use_llm`/`--llm_service` возвращает строку пути к сервису.

Таким образом, CLI лишь собирает конфигурацию и передаёт её `PdfConverter`.

## 2. DocumentBuilder и Surya (пошагово)

`DocumentBuilder` описан в `marker/builders/document.py` и вызывается конвертером. Почти все ключевые классы (builder-ы, процессоры, рендеры, сервисы) реализуют `__call__`, поэтому экземпляры можно вызывать как функции.

```python
def __call__(self, provider, layout_builder, line_builder, ocr_builder):
    document = self.build_document(provider)
    layout_builder(document, provider)
    line_builder(document, provider)
    if not self.disable_ocr:
        ocr_builder(document, provider)
    return document
```

### 2.1 PdfProvider

Поставляет страницы и изображения из PDF. Возвращает:
- low-res изображения (LayoutBuilder, LineBuilder),
- high-res изображения (OCR, LLM-кропы),
- bounding boxes страниц.

### 2.2 LayoutBuilder (Surya Layout)

Файл `marker/builders/layout.py`.

1. **`__call__`** решает, использовать ли `surya_layout()` или `forced_layout`, затем вызывает:
   - `add_blocks_to_pages()` — добавляет блоки (SectionHeader, Text, Table и др.) на страницы;
   - `expand_layout_blocks()` — расширяет блоки, если Surya выделила слишком узкие области.
2. **`surya_layout()`** вызывает `surya.layout.LayoutPredictor`:
   ```python
   layout_results = self.layout_model(
       [p.get_image(highres=False) for p in pages],
       batch_size=int(self.get_batch_size()),
   )
   ```
   `self.layout_model` — это Surya Layout, у которой тоже реализован `__call__`.
3. **`add_blocks_to_pages()`**:
   - Нормализует координаты боксов к размеру страницы.
   - Для каждого бокса подбирает класс (`BlockTypes[label]`) и создаёт блок в `PageGroup`.
   - Сохраняет `top_k` вероятности.
4. **`expand_layout_blocks()`**:
   - Измеряет зазоры между блоками.
   - Расширяет polygon блока до столкновения с соседями, но максимум до `max_expand_frac`.

### 2.3 LineBuilder и OcrBuilder

- `LineBuilder` (файл `marker/builders/line.py`) объединяет символы/спаны в строки, вычисляет высоту линий, baseline и т.д. Эти данные нужны, например, `SectionHeaderProcessor`.
- `OcrBuilder` (файл `marker/builders/ocr.py`) вызывает Surya OCR для high-res изображений и записывает текст в блоки. Если OCR отключён (`disable_ocr=True`), текст берётся из PDF, если он уже есть.

### 2.4 Итог

После `DocumentBuilder`:
- `Document.pages` — список `PageGroup` с низкоуровневыми блоками и изображениями;
- у блоков есть `polygon`, `text`, `html`, `top_k` — «сырая» разметка;
- далее вступают процессоры, которые доводят структуру до финального вида.

## 3. Процессоры (с кодом и пояснениями)

### 3.1 Базовый класс

Все процессоры наследуются от `BaseProcessor` (`marker/processors/__init__.py`):

```python
class BaseProcessor:
    block_types: Tuple[BlockTypes] | None = None

    def __init__(self, config=None):
        assign_config(self, config)

    def __call__(self, document: Document, *args, **kwargs):
        raise NotImplementedError
```

* `block_types` сообщает, какие блоки обрабатываются (например, только `SectionHeader`).
* `__call__` нужно переопределить, чтобы реализовать логику.
* Процессор вызывается так: `processor(document)`.

`PdfConverter` просто итерируется по списку процессоров:

```python
for processor in self.processor_list:
    processor(document)
```

Поэтому любой процессор может изменять `document` на месте.

### 3.2 SectionHeaderProcessor

Файл `marker/processors/sectionheader.py`.

Главная цель — присвоить блокам типа `SectionHeader` правильный уровень заголовка (h1-h6).

```python
def __call__(self, document):
    line_heights = {}
    for page in document.pages:
        for block in page.children:
            if block.block_type not in self.block_types:
                continue
            if block.structure is not None:
                line_heights[block.id] = block.line_height(document)
            else:
                line_heights[block.id] = 0
                block.ignore_for_output = True

    heading_ranges = self.bucket_headings(line_heights.values())
```

* `block.line_height(document)` использует данные `LineBuilder`.
* Если структура `None`, блок помечается `ignore_for_output`, чтобы он не попал в финальный HTML.

Дальше блоки распределяются по диапазонам высот:

```python
for block in page.children:
    if block.block_type not in self.block_types:
        continue
    block_height = line_heights.get(block.id, 0)
    if block_height > 0:
        for idx, (min_height, max_height) in enumerate(heading_ranges):
            if block_height >= min_height * self.height_tolerance:
                block.heading_level = idx + 1
                break
    if block.heading_level is None:
        block.heading_level = self.default_level
```

Полученное `heading_level` потом используется `SectionHeader.assemble_html()` для генерации `<h1>...`.

### 3.3 LineMergeProcessor и др.

`LineMergeProcessor` (файл `marker/processors/line_merge.py`) объединяет короткие строки в абзацы, чтобы рендерер не превращал каждую строку в отдельный `<p>`. Это пример процессора без `block_types`: он анализирует весь документ и строит новые блоки `Text`.

Другие примеры:
- `EquationProcessor` (`marker/processors/equation.py`) ищет окружения вроде `$$...$$`, помечает их как `BlockTypes.Equation`.
- `ListProcessor` определяет номера списков (1., (a) и т.д.) и строит иерархию `ListGroup`/`ListItem`.

### 3.4 LLM-процессоры

Существуют «простые» (`BaseLLMSimpleBlockProcessor`) и «сложные» (`BaseLLMComplexBlockProcessor`) LLM-процессоры.

#### Простые (на блок)

Например, `LLMTableProcessor` получает один блок таблицы, формирует prompt и вызывает `llm_service`. Чтобы не тратить по запросу на каждый блок, `PdfConverter.initialize_processors()` собирает все простые LLM-процессоры в один `LLMSimpleBlockMetaProcessor`. Он перебирает блоки и распределяет их по конкретным процессорам.

#### Сложные (на набор блоков/страницу)

`LLMSectionHeaderProcessor` (файл `marker/processors/llm/llm_sectionheader.py`) наследуется от `BaseLLMComplexBlockProcessor`.

Основные методы:

1. `rewrite_blocks()` собирает все section header и формирует JSON.
2. `process_rewriting()`:
   ```python
   prompt = self.page_prompt.replace(
       "{{section_header_json}}", json.dumps(section_header_json)
   )
   response = self.llm_service(
       prompt, None, document.pages[0], SectionHeaderSchema
   )
   ```
   Здесь `self.llm_service` — объект сервиса (например, Ollama), у которого тоже есть `__call__`. Ему передаётся prompt, блок (для метаданных), схема ожидаемого ответа (`SectionHeaderSchema`).
3. Если `response["correction_type"] == "corrections_needed"`, вызывается `handle_rewrites()`, которая обновляет HTML конкретных блоков.

Все LLM-процессоры прогоняются только если включён флаг `use_llm`. В противном случае `BaseLLMProcessor.__init__` не задаёт `self.llm_service`, и `__call__` просто возвращается.

## 3.5 create_model_dict: как подготавливаются модели Surya

Функция живёт в `marker/models.py`:

```python
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
...

def create_model_dict(device=None, dtype=None, attention_implementation=None) -> dict:
    return {
        "layout_model": LayoutPredictor(
            FoundationPredictor(
                checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT,
                attention_implementation=attention_implementation,
                device=device,
                dtype=dtype,
            )
        ),
        "recognition_model": RecognitionPredictor(
            FoundationPredictor(
                checkpoint=surya_settings.RECOGNITION_MODEL_CHECKPOINT,
                attention_implementation=attention_implementation,
                device=device,
                dtype=dtype,
            )
        ),
        "table_rec_model": TableRecPredictor(device=device, dtype=dtype),
        "detection_model": DetectionPredictor(device=device, dtype=dtype),
        "ocr_error_model": OCRErrorPredictor(device=device, dtype=dtype),
    }
```

Разбор:

1. **FoundationPredictor** — базовый «столб» Surya. Он умеет подгружать чекпоинты (layout/recognition) и поддерживает разные attention-реализации (`flash`, `sdpa` и т.п.). Параметры `device`/`dtype` передаются далее, поэтому вы можете принудительно использовать, например, `device="cpu"`, `dtype=torch.float16`.
2. **LayoutPredictor / RecognitionPredictor** поверх FoundationPredictor добавляют свою специфику: layout — для разметки блоков, recognition — для OCR. Они получают готовый `FoundationPredictor`, уже указывающий путь к весам (`surya_settings.LAYOUT_MODEL_CHECKPOINT` и т.д.).
3. Остальные модели (`TableRecPredictor`, `DetectionPredictor`, `OCRErrorPredictor`) не требуют foundation и инициализируются напрямую.
4. Возвращается обычный словарь, который потом пробрасывается почти во все компоненты. Каждый ключ можно использовать как «артефакт» при создании зависимостей (через `BaseConverter.resolve_dependencies`). Например, `LayoutBuilder` объявляет `layout_model` в сигнатуре `__init__`, поэтому он автоматически получает `artifact_dict["layout_model"]`.
5. В тестах (`tests/conftest.py`) эта функция используется так же, чтобы переиспользовать единый набор весов между тестами.

## 4. PdfConverter внутри: зависимости, процессоры и render

Класс `PdfConverter` (`marker/converters/pdf.py`) отвечает за сборку всех зависимостей. Псевдокод инициализации:

```python
class PdfConverter(BaseConverter):
    default_processors = (...большая последовательность...)

    def __init__(self, artifact_dict, processor_list=None, renderer=None, llm_service=None, config=None):
        self.artifact_dict = artifact_dict
        if processor_list is None:
            processor_list = self.default_processors
        if renderer is None:
            renderer = MarkdownRenderer

        # Инициализация LLM
        if llm_service:
            llm_service = self.resolve_dependencies(llm_service_cls)
        elif config.get("use_llm", False):
            llm_service = self.resolve_dependencies(self.default_llm_service)
        self.artifact_dict["llm_service"] = llm_service

        # Инициализируем процессоры и renderer
        self.processor_list = self.initialize_processors(processor_list)
        self.renderer = renderer
```

Важный метод `resolve_dependencies()` (наследуется от `BaseConverter`):

```python
def resolve_dependencies(self, cls):
    for param_name, param in inspect.signature(cls.__init__).parameters.items():
        if param_name == "self": continue
        elif param_name == "config": resolved_kwargs[param_name] = self.config
        elif param.name in self.artifact_dict:
            resolved_kwargs[param_name] = self.artifact_dict[param_name]
        elif param.default != inspect.Parameter.empty:
            resolved_kwargs[param_name] = param.default
        else:
            raise ValueError(...)
    return cls(**resolved_kwargs)
```

То есть если класс требует `layout_model`, он автоматически получит `artifact_dict["layout_model"]`. Это избавляет от ручного пробрасывания зависимостей.

### 4.1 build_document

```python
def build_document(self, filepath):
    provider_cls = provider_from_filepath(filepath)
    layout_builder = self.resolve_dependencies(self.layout_builder_class)
    line_builder = self.resolve_dependencies(LineBuilder)
    ocr_builder = self.resolve_dependencies(OcrBuilder)
    provider = provider_cls(filepath, self.config)
    document = DocumentBuilder(self.config)(provider, layout_builder, line_builder, ocr_builder)

    structure_builder_cls = self.resolve_dependencies(StructureBuilder)
    structure_builder_cls(document)

    for processor in self.processor_list:
        processor(document)

    return document
```

Здесь видно, что после `DocumentBuilder` дополнительно запускается `StructureBuilder` (он строит связи между блоками, создаёт иерархию). Затем последовательно вызываются все процессоры.

### 4.2 initialize_processors и LLMSimpleBlockMetaProcessor

`initialize_processors()` разделяет процессоры на «простые LLM» и остальные:

```python
simple_llm_processors = [p for p in processors if issubclass(type(p), BaseLLMSimpleBlockProcessor)]
other_processors = [...]

meta = LLMSimpleBlockMetaProcessor(processor_lst=simple_llm_processors, llm_service=self.llm_service, config=self.config)
other_processors.insert(insert_position, meta)
return other_processors
```

Это значит, что в итоговом списке останется один `LLMSimpleBlockMetaProcessor`, который внутри уже знает про конкретные `LLMTableProcessor`, `LLMImageDescriptionProcessor` и т.п. Таким образом, `PdfConverter` контролирует, в каком месте цепочки происходят вызовы LLM.

## 5. Рендеры

- **MarkdownRenderer**
  - Берёт HTML, собранный блоками, пропускает через Markdownify.
  - Управляет пагинацией, escape-символами, преобразует таблицы в Markdown или оставляет HTML.
  - Возвращает `MarkdownOutput(markdown, images, metadata)`.
- **HTMLRenderer** просто возвращает HTML + изображения.
- **JSONRenderer** оставляет иерархию блоков в JSON.
- **ChunkRenderer** выдаёт «плоский» список блоков (удобно для RAG).

### 5.1 MarkdownRenderer: подробности

Работает поверх `HTMLRenderer` (от которого наследуется). Ключевые шаги (`marker/renderers/markdown.py`):

```python
document_output = document.render(self.block_config)
full_html, images = self.extract_html(document, document_output)
markdown = self.md_cls.convert(full_html)
markdown = cleanup_text(markdown)
return MarkdownOutput(markdown=markdown, images=images, metadata=...)
```

* `document.render()` рекурсивно вызывает `assemble_html()` всех блоков.
* `extract_html()` собирает HTML и сопутствующие картинки (сохраняются на диск при сохранении результата).
* `Markdownify` (настройки — heading_style, bullets, escape_* и т.п.) преобразует HTML в Markdown.
* Возвращается `MarkdownOutput` — pydantic-модель с полями `markdown`, `images`, `metadata`.

Ключевой момент: renderer ничего не знает про перевод или CLI. Он получает готовый `Document` и выдаёт форматированный вывод.

## 6. Перевод Markdown (внутренности)

Файл `marker/translation/translator.py`. Основные функции:

1. `translate_rendered_output()` — входная точка.

```python
def translate_rendered_output(rendered, target_language, llm_service):
    language_info = _resolve_language(target_language)
    chunks = _chunk_text(rendered.markdown)
    translated_segments = [
        _translate_chunk(llm_service, chunk, language_info) for chunk in chunks
    ]
    rendered.markdown = "".join(translated_segments)
    rendered.metadata["translation"] = {...}
    return rendered
```

2. `_resolve_language()` нормализует введённое имя (поддерживаются алиасы) через `rapidfuzz.process.extractOne`.
3. `_chunk_text()` делит Markdown по параграфам с ограничением `MAX_TRANSLATION_CHARS` (по умолчанию 1200). Это предотвращает переполнение контекста LLM.
4. `_translate_chunk()`:

```python
prompt_base = PROMPT_TEMPLATE.format(...)
for attempt in range(1, MAX_TRANSLATION_ATTEMPTS + 1):
    prompt = prompt_base + (REMINDER, если попытка > 1)
    response = llm_service(
        prompt=prompt,
        image=None,
        block=None,
        response_schema=TranslationResponse,
        timeout=getattr(llm_service, "timeout", None),
    )
    translation_text = response.get("translation", "") ...
    cleaned = _clean_translated_output(translation_text.strip())
    if _translation_is_valid(chunk, cleaned):
        return cleaned
logger.error("LLM translation failed; returning original chunk.")
return chunk
```

5. `_clean_translated_output()` убирает служебные фразы, `<MARKDOWN>` теги, пустые строки.
6. `_translation_is_valid()` проверяет, что текст не пустой, содержит достаточно кириллицы. Если LLM прислал системное сообщение вроде «Please provide...», chunk отклоняется и делается новая попытка.

Таким образом, перевод — это постпроцесс над `MarkdownOutput`, не затрагивающий основную структуру документа. LLM сервис используется тот же, что и для `use_llm` процессоров (если пользователь не задал другой).

## 6. Как это собрать воедино (пример)

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.translation import translate_rendered_output

models = create_model_dict()
converter = PdfConverter(
    artifact_dict=models,
    config={"use_llm": True},
    llm_service="marker.services.ollama.OllamaService",
)
rendered = converter("2601.00100v1.pdf")  # MarkdownOutput
rendered = translate_rendered_output(
    rendered,
    target_language="Russian",
    llm_service=models["llm_service"],
)
print(rendered.markdown[:500])
```

## 7. Что где лежит в коде

| Компонент | Файл | Примечания |
|-----------|------|------------|
| Конвертер и последовательность процессоров | `marker/converters/pdf.py` | Содержит `default_processors` и логику сборки документа |
| DocumentBuilder/Surya | `marker/builders/document.py`, `marker/builders/layout.py`, `marker/builders/ocr.py`, `marker/builders/line.py` | LayoutBuilder напрямую использует Surya; OcrBuilder — Surya OCR |
| Процессоры | `marker/processors/*` | Каждый модуль отвечает за свой тип блоков |
| LLM процессоры | `marker/processors/llm/*` | Содержат промпты и обработку ответов от LLM |
| Renderer | `marker/renderers/*.py` | Markdown/HTML/JSON/Chunks |
| CLI и конфигурация | `convert_single.py`, `marker/config/parser.py` | CLI запускает конвертер и сохраняет результат |
| Перевод | `marker/translation/translator.py` | Логика chunking + вызов LLM + постобработка |

## 8. FAQ

- **Нужно ли Surya, если в PDF встроенный текст?** Да, хотя бы LayoutBuilder будет предсказывать структуру. OCR можно отключить (`force_ocr=False`, `disable_ocr=True`), но layout всё равно нужен.
- **Как отключить «тяжёлые» LLM-процессоры?** Передайте `--processors` и перечислите нужные классы без `LLM*`, либо настроьте JSON-конфиг, чтобы удалить конкретные процессоры.
- **Можно ли использовать один LLM для всех шагов?** Да. CLI опция `--llm_service` задаёт реализацию, и она автоматически пробрасывается в процессоры и в переводчик.

## 9. Что посмотреть дальше

- `README.md` — полный обзор возможностей и примеры CLI.
- `tests/` — показательные юнит-тесты для отдельных блоков.
- `marker/output.py` — вспомогательные функции для преобразования дерева блоков в текст/HTML.

Теперь у вас есть общее представление, как Surya, процессоры и рендеры складываются в единый конвейер Marker.
