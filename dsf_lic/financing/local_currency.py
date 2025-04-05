from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd

from ..utils.metadata import metadata


class DebtInfo(NamedTuple):
    disbursement: str
    interest_rate_series: pd.Series
    grace_period: int
    loan_maturity: int
    repayment_schedule: Optional[str] = None
    prefix: str = ""
    sub_group: str = ""
    group: str = ""
    description: Optional[str] = None


class DisbursementInfo(NamedTuple):
    value: float
    year: int
    interest_rate: float
    grace_period: int
    loan_maturity: int


def get_debt_info(name: str) -> DebtInfo:
    financing_info = DebtInfo(
        interest_rate_series=metadata.internal_interest_rates[name],
        **metadata.internal_financing[name]
    )
    return financing_info


def create_year_debt_table(debt_info: DisbursementInfo, data: pd.DataFrame) -> pd.DataFrame:
    debt_table = (
        pd.DataFrame(index=metadata.projection_year_index)
        .assign(
            cumulative = lambda df: df.index.to_series().ge(debt_info.year).mul(debt_info.value),

            amortization =
            lambda df: df.index.to_series()
            .between(
                debt_info.year + debt_info.grace_period + 1,
                debt_info.year + debt_info.loan_maturity,
            )
            .mul(debt_info.value).div(debt_info.loan_maturity - debt_info.grace_period),

            amortization_usd = lambda df: df["amortization"] / data["ENDA"],

            stock_of_debt = lambda df: df["cumulative"] - df["amortization"].cumsum(),
            stock_of_debt_usd = lambda df: df["stock_of_debt"] / data["ENDE"],

            interest = lambda df: df["stock_of_debt"].shift(1, fill_value=0) * debt_info.interest_rate / 100,
            interest_usd = lambda df: df["interest"] / data["ENDA"],

            total_debt_service_usd = lambda df: df["amortization_usd"] + df["interest_usd"],

            pv_multiplier = lambda df:
            pd.Series(metadata.setting.discount_rate, index=df.index)
            .div(100).add(1).cumprod(),

            tds_usd_npv = lambda df: 
            df["total_debt_service_usd"].shift(-1 , fill_value=0)
            .div(df["pv_multiplier"]).iloc[::-1].cumsum().iloc[::-1]
            .mul(df["pv_multiplier"].shift(1, fill_value=1)),

            pv_usd = lambda df: df[["stock_of_debt_usd", "tds_usd_npv"]].min(axis="columns"),
        )
    )
    return debt_table


def create_debt_table(
    name: str,
    data: pd.DataFrame,
) -> pd.DataFrame:
    financing_info = get_debt_info(name)
    
    debt_list: list[dict] = (
        pd.concat(
            [
                data[financing_info.disbursement].rename("value"),
                financing_info.interest_rate_series.rename("interest_rate"),
            ],
            axis="columns",
            join="inner",
        )
        .rename_axis("year")
        .reset_index()
        .assign(
            grace_period = financing_info.grace_period,
            loan_maturity = financing_info.loan_maturity
        )
        .to_dict("records")
    )

    debt_info_list = [DisbursementInfo(**debt) for debt in debt_list]

    debt_tables = {
        debt.year: create_year_debt_table(debt, data)
        for debt in debt_info_list
    }
    aggregate_debt_table = (
        pd.concat(
            debt_tables,
            names=["bond_year", "repayment_year"],
        )
        .groupby("repayment_year").sum()
        .join(data[financing_info.disbursement].rename("disbursement"))
        .assign(
            disbursement_usd = lambda df: df["disbursement"] / data["ENDA"],
        )
    )
    return aggregate_debt_table
