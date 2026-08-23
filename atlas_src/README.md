# atlas_src — наш форк исходников Duck (TAAF / ARC3-Inference)

Снято с Kaggle-датасета `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`
(`taaf-kaggle-bundle.json`: `created_at 2026-08-07T17:59:49`, ветка
`feature/animation-awareness`) — это тот же bundle, что смонтирован в
`notebooks_duck/submission.ipynb` через `DATASET_SOURCES`. Наш прежний
снэпшот в `reference/duck-source/` был от 12.06 и устарел; см.
[[arc-agi-3-atlas-substrate]] / `docs/atlas_v1_22.08.md`.

Лицензия — MIT (`src/ARC3-Inference/pyproject.toml`), сам bundle — CC0-1.0
(`README.dataset.md`). Правообладание задекларировано, повторная публикация
легальна.

## Что здесь лежит

Ровно layout, который ждёт `taaf_kaggle_run.ipynb` при поиске бандла по
маркеру `taaf-kaggle-bundle.json`:

- `benchmark_initial.pkl`, `deploy_target.pkl` — пиклы `taaf.benchmark.Benchmark`,
  `taaf.game_api.GameAPI`, `arc_agi.base.OperationMode`,
  `inference.framework.solver.HarnessSolver`, `taaf.deploy_kaggle.KaggleTarget`
  (проверено `pickletools.dis`, без импорта — все имена модулей совпадают
  с деревом в `src/`).
- `src/ARC3-Inference/` — агент (промпты, turn loop, инструменты).
- `src/tufa-arc-agi-framework/` — харнесс (`taaf.*`), включая пакет `arc_agi`
  как обычную PyPI-зависимость (`arc_agi>=0.9.8` в pyproject, НЕ вендорится).
- `setup_commands.json`/`teardown_commands.json`/`preamble.txt` — то, что
  ноутбук выполняет при старте/остановке (vLLM, veriфикация GPU и т.п.).

## Жёсткое ограничение: не переименовывать пакеты

`benchmark_initial.pkl`/`deploy_target.pkl` — это ПИКЛЫ, не исходный код:
они хранят только квалифицированные имена классов (`module.ClassName`),
а не байткод. Значит редактировать МЕТОДЫ этих классов в `src/` можно
свободно, но папки/имена пакетов (`taaf`, `inference`, `arc_agi`,
каталоги `ARC3-Inference/`, `tufa-arc-agi-framework/`) трогать нельзя —
иначе `pickle.load` в cell 7 ноутбука падает.

## Дальше

1. Здесь вносим правки (C0, каталог ловушек, промпты, turn loop).
2. Пересобираем бандл-датасет (маркер + пиклы как есть + обновлённый `src/`)
   и публикуем как **свой** Kaggle-датасет — не трогая
   `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`.
3. В `notebooks_duck/submission.ipynb` (ячейка `DATASET_SOURCES`) меняем
   `jakobbrggen/...` на наш слаг.

Оба последних шага — внешние, видимые действия на Kaggle-аккаунте; делать
без отдельного подтверждения не стоит (см. [[feedback-kaggle-push-consent]]:
формально это не GPU-пуш и не тратит квоту, но тот же принцип "показать,
что публикуем, и подождать добро").
