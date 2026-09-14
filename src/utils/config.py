from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "config.yaml"
)

LOCAL_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "config.local.yaml"
)


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"File konfigurasi tidak ditemukan: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(
            f"Isi konfigurasi tidak valid: {path}"
        )

    return content


def deep_merge(
    base: dict,
    override: dict,
) -> dict:
    """
    Menggabungkan konfigurasi lokal ke konfigurasi utama
    tanpa menghapus bagian lain.
    """

    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = value

    return result


def load_config(
    config_path: Optional[str] = None,
) -> dict:
    """
    Membaca config.yaml kemudian menerapkan
    config.local.yaml jika tersedia.
    """

    if config_path is None:
        main_path = DEFAULT_CONFIG_PATH
    else:
        main_path = Path(config_path)

        if not main_path.is_absolute():
            main_path = PROJECT_ROOT / main_path

    config = read_yaml(main_path)

    if LOCAL_CONFIG_PATH.exists():
        local_config = read_yaml(
            LOCAL_CONFIG_PATH
        )

        config = deep_merge(
            config,
            local_config,
        )

    return config


def get_config_path(
    config: dict,
    key: str,
) -> Path:
    value = config.get(
        "paths",
        {},
    ).get(key)

    if not value:
        raise KeyError(
            f"Path '{key}' tidak ditemukan "
            "di konfigurasi."
        )

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()