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
class FifaConfig:
    seasons: list[int] = field(default_factory=lambda: [2010, 2026])
    raw_subdir: str = "fifa"


@dataclass
class SourceDirConfig:
    raw_subdir: str = ""


@dataclass
class PlayerDataConfig:
    fifa: FifaConfig = field(default_factory=FifaConfig)
    fm: SourceDirConfig = field(default_factory=lambda: SourceDirConfig("fm"))
    caps: SourceDirConfig = field(default_factory=lambda: SourceDirConfig("caps"))


@dataclass
class SquadConfig:
    size: int = 26
    starting_xi: int = 11
    substitutes: int = 15
    formation: dict[str, int] = field(
        default_factory=lambda: {"gk": 1, "def": 4, "mid": 3, "att": 3}
    )


@dataclass
class RosterSource:
    tournament: str
    edition_year: int
    start_date: str
    wiki_page: str


@dataclass
class RostersConfig:
    raw_subdir: str = "rosters"
    sources: list[RosterSource] = field(default_factory=list)


@dataclass
class Config:
    db_schema: str = "wc2026"
    elo: EloConfig = field(default_factory=EloConfig)
    form: FormConfig = field(default_factory=FormConfig)
    train_cutoff_date: str = "2024-01-01"
    rng: RngConfig = field(default_factory=RngConfig)
    raw_data_dir: str = "data/raw"
    player_data: PlayerDataConfig = field(default_factory=PlayerDataConfig)
    squad: SquadConfig = field(default_factory=SquadConfig)
    rosters: RostersConfig = field(default_factory=RostersConfig)
    tournaments: list[str] = field(default_factory=list)


def load_config(path: Path | None = None) -> Config:
    config_path = path or REPO_ROOT / "config.yaml"
    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    pd_raw = raw.get("player_data", {})
    player_data = PlayerDataConfig(
        fifa=FifaConfig(**pd_raw.get("fifa", {})),
        fm=SourceDirConfig(**pd_raw.get("fm", {"raw_subdir": "fm"})),
        caps=SourceDirConfig(**pd_raw.get("caps", {"raw_subdir": "caps"})),
    )

    r_raw = raw.get("rosters", {})
    rosters = RostersConfig(
        raw_subdir=r_raw.get("raw_subdir", "rosters"),
        sources=[RosterSource(**s) for s in r_raw.get("sources", [])],
    )

    return Config(
        db_schema=raw.get("db_schema", "wc2026"),
        elo=EloConfig(**raw.get("elo", {})),
        form=FormConfig(**raw.get("form", {})),
        train_cutoff_date=raw.get("train_cutoff_date", "2024-01-01"),
        rng=RngConfig(**raw.get("rng", {})),
        raw_data_dir=raw.get("raw_data_dir", "data/raw"),
        player_data=player_data,
        squad=SquadConfig(**raw.get("squad", {})),
        rosters=rosters,
        tournaments=raw.get("tournaments", []),
    )


if __name__ == "__main__":
    cfg = load_config()
    print(f"db_schema: {cfg.db_schema}")
    print(f"elo.k: {cfg.elo.k}, home_advantage: {cfg.elo.home_advantage}, mov: {cfg.elo.mov}")
    print(f"form.window_n: {cfg.form.window_n}")
    print(f"train_cutoff_date: {cfg.train_cutoff_date}")
    print(f"rng.seed: {cfg.rng.seed}")
    print(f"raw_data_dir: {cfg.raw_data_dir}")
    print(f"fifa.seasons: {cfg.player_data.fifa.seasons}, fifa.raw_subdir: {cfg.player_data.fifa.raw_subdir}")
    print(f"squad: {cfg.squad.size} = {cfg.squad.starting_xi} + {cfg.squad.substitutes}, formation: {cfg.squad.formation}")
    print(f"rosters: {len(cfg.rosters.sources)} sources → {[s.wiki_page for s in cfg.rosters.sources]}")
    print(f"tournaments: {len(cfg.tournaments)} configured")
