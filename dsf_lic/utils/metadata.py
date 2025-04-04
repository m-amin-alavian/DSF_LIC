from typing import NamedTuple
from pathlib import Path

import pandas as pd

import yaml


def read_yaml_file(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


class Settings(NamedTuple):
    first_year: int
    projection_year: int
    last_year: int
    residency_based: bool
    discount_rate: float


class Metadata:
    external_loans: dict
    internal_financing: dict
    variables: dict[str, dict]

    def __init__(self):
        self.package_path = Path(__file__).parents[1]
        self.setting = self._read_settings()
        self._load_metadata()

    def reload_metadata(self) -> None:
        self._load_metadata()

    def _read_settings(self) -> Settings:
        settings = read_yaml_file(self.package_path.joinpath("settings.yaml"))
        return Settings(
            first_year = settings["First Year"],
            projection_year = settings["First Year of Projections"],
            last_year = settings["Last Year"],
            residency_based =
            settings["Definition of External/Domestic Debt"] == "Residency-Based",
            discount_rate = settings["Discount Rate"],
        )

    def _load_metadata(self) -> None:
        for metadata in ["external_loans", "internal_financing"]:
            metadata_path = self.package_path.joinpath("metadata", f"{metadata}.yaml")
            setattr(
                self,
                metadata,
                read_yaml_file(metadata_path)
            )
        self.variables = {}
        for path in self.package_path.joinpath("metadata", "variables").iterdir():
            self.variables[path.stem] = read_yaml_file(path)

    @property
    def year_index(self) -> pd.Index:
        return pd.RangeIndex(
            self.setting.first_year,
            self.setting.last_year + 1,
        )

    @property
    def projection_year_index(self) -> pd.Index:
        return pd.RangeIndex(
            self.setting.projection_year,
            self.setting.last_year + 1,
        )

    @property
    def internal_interest_rates(self) -> pd.DataFrame:
        file_path = self.package_path.joinpath("metadata", "internal_interest_rates.csv")
        return (
            pd.read_csv(file_path, index_col=0)
            .transpose()
            .pipe(lambda df: df.set_axis(df.index.astype(int))) # type: ignore
            .reindex(self.projection_year_index)
            .ffill()
        )
        
    @property
    def repayment_schedule(self) -> pd.DataFrame:
        file_path = file_path = self.package_path.joinpath("metadata", "repayment_schedule.csv")
        return (
            pd.read_csv(file_path, index_col=0)
            .transpose()
            .pipe(lambda df: df.set_axis(df.index.astype(int))) # type: ignore
        )
    

metadata = Metadata()
