from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import yaml


REPO_ROOT = Path(__file__).parent.parent.parent


@dataclass
class EloConfig:
    k: float = 32
    home_advantage: float = 100
    mov: bool = True


@dataclass
class FormConfig:
    window_n: int = 10


@dataclass
class RngConfig:
    seed: int = 42


@dataclass
class Config:
    db_schema: str = "wc2026"
    elo: EloConfig = field(default_factory=EloConfig)
    form: FormConfig = field(default_factory=FormConfig)
    train_cutoff_date: str = "2024-01-01"
    rng: RngConfig = field(default_factory=RngConfig)
    raw_data_dir: str = "data/raw"


def load_config(path: Path | None = None) -> Config:
    config_path = path or REPO_ROOT / "config.yaml"
    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return Config(
        db_schema=raw.get("db_schema", "wc2026"),
        elo=EloConfig(**raw.get("elo", {})),
        form=FormConfig(**raw.get("form", {})),
        train_cutoff_date=raw.get("train_cutoff_date", "2024-01-01"),
        rng=RngConfig(**raw.get("rng", {})),
        raw_data_dir=raw.get("raw_data_dir", "data/raw"),
    )


if __name__ == "__main__":
    cfg = load_config()
    print(f"db_schema: {cfg.db_schema}")
    print(f"elo.k: {cfg.elo.k}, home_advantage: {cfg.elo.home_advantage}, mov: {cfg.elo.mov}")
    print(f"form.window_n: {cfg.form.window_n}")
    print(f"train_cutoff_date: {cfg.train_cutoff_date}")
    print(f"rng.seed: {cfg.rng.seed}")
    print(f"raw_data_dir: {cfg.raw_data_dir}")
