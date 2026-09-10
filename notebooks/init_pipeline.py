import os
import json
import shutil
from pathlib import Path


def init_experiment(root: str, config: dict) -> None:
    """
    Инициализирует эксперимент:
    - создаёт папку root/artifacts/<experiment_name>,
    - копирует туда root/config.yaml,
    - создаёт status.json с полями init=True, остальные False.

    Действие выполняется только если status.json отсутствует или его поле init == False.
    """
    # Извлекаем имя эксперимента из конфига
    try:
        experiment_name = config.general.experiment_name
    except KeyError:
        raise KeyError("В конфиге отсутствует поле 'general.experiment_name'")

    # Пути
    root_path = Path(root)
    artifacts_dir = root_path / 'artifacts' / experiment_name
    status_path = artifacts_dir / 'status.json'
    config_src = root_path / 'config.yaml'
    config_dst = artifacts_dir / 'config.yaml'

    # Проверяем, нужно ли выполнять инициализацию
    should_init = False
    if not status_path.exists():
        should_init = True
    else:
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
            if not status_data.get('init', False):
                should_init = True
        except (json.JSONDecodeError, OSError):
            # Если файл повреждён или не читается, считаем, что нужна инициализация
            should_init = True

    if not should_init:
        print(f"Инициализация пропущена: {status_path} уже существует и init=True")
        return

    # Создаём целевую директорию (и все промежуточные)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Копируем config.yaml
    if not config_src.exists():
        raise FileNotFoundError(f"Исходный файл конфигурации не найден: {config_src}")
    shutil.copy2(config_src, config_dst)

    # Создаём status.json
    status_data = {
        'init': True,
        'dataset': False,
        'features': False,
        'choose_model': False,
        'optimization': False,
        'fitting': False
    }
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, indent=4, ensure_ascii=False)

    print(f"Инициализация выполнена. Папка: {artifacts_dir}")
