from pathlib import Path
from typing import NamedTuple, Optional

import pandas as pd

import yaml


first_year = 2018
last_year = 2070

discount = 5

class InternalFinancingInfo(NamedTuple):
    held_by_residents: str
    held_by_non_residents: str
    interest_rate_series: pd.Series
    grace_period: int
    loan_maturity: int
    repayment_schedule: Optional[str] = None
    description: Optional[str] = None


class InternalDebtInfo(NamedTuple):
    value: float
    year: int
    interest_rate: float
    grace_period: int
    loan_maturity: int



def get_internal_financing_metadata() -> dict:
    with Path("dsf", "metadata", "internal_financing.yaml").open() as file:
        internal_financing = yaml.safe_load(file)
    return internal_financing


def get_financing_info(name: str) -> InternalFinancingInfo:
    financing_metadata = get_internal_financing_metadata()
    interest_rate_series = (
        pd.read_csv(Path("dsf", "metadata", "internal_interest_rates.csv"), index_col=0)
        .transpose()
        .loc[:, name]
    )
    interest_rate_series.index = interest_rate_series.index.astype("Int64")
    interest_rate_series = (
        pd.Series(
            interest_rate_series,
            index=pd.RangeIndex(first_year, last_year + 1),
        )
        .ffill()
    )

    financing_info = InternalFinancingInfo(
        interest_rate_series=interest_rate_series,
        **financing_metadata[name]
    )
    return financing_info


def create_debt_pv_table_for_year(debt_info: InternalDebtInfo, data: pd.DataFrame) -> pd.DataFrame:
    pv_table = (
        pd.DataFrame(index=pd.RangeIndex(first_year, last_year + 1))
        .assign(
            cumulative = lambda df: df.index.to_series().ge(debt_info.year).mul(debt_info.value),

            amortization =
            lambda df: df.index.to_series()
            .between(debt_info.year + debt_info.grace_period + 1, debt_info.year + debt_info.loan_maturity)
            .mul(debt_info.value).div(debt_info.loan_maturity - debt_info.grace_period),

            amortization_usd = lambda df: df["amortization"] / data["ENDA"],

            stock_of_debt = lambda df: df["cumulative"] - df["amortization"].cumsum(),
            stock_of_debt_usd = lambda df: df["stock_of_debt"] / data["ENDE"],

            interest = lambda df: df["stock_of_debt"].shift(1, fill_value=0) * debt_info.interest_rate / 100,
            interest_usd = lambda df: df["interest"] / data["ENDA"],

            total_debt_service_usd = lambda df: df["amortization_usd"] + df["interest_usd"],

            pv_multiplier = lambda df: pd.Series(discount, index=df.index).div(100).add(1).cumprod(),

            tds_usd_npv = lambda df: 
            df["total_debt_service_usd"].shift(-1 , fill_value=0)
            .div(df["pv_multiplier"]).iloc[::-1].cumsum().iloc[::-1]
            .mul(df["pv_multiplier"].shift(1, fill_value=1)),

            pv_usd = lambda df: df[["stock_of_debt_usd", "tds_usd_npv"]].min(axis="columns"),
        )
    )
    return pv_table


def create_aggregate_financing_table(
    name: str,
    data: pd.DataFrame,
    for_residents: Optional[bool] = True
) -> pd.DataFrame:
    financing_info = get_financing_info(name)
    if for_residents == True:
        values = data[financing_info.held_by_residents]
    elif for_residents == False:
        values = data[financing_info.held_by_non_residents]
    elif for_residents is None:
        raise NotImplementedError
    else:
        raise ValueError
    
    debt_list: list[dict] = (
        pd.concat(
            [
                values.rename("value"),
                financing_info.interest_rate_series.rename("interest_rate")
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

    debt_info_list = [InternalDebtInfo(**debt) for debt in debt_list]

    debt_tables = {
        debt.year: create_debt_pv_table_for_year(debt, data)
        for debt in debt_info_list
    }
    aggregate_table = (
        pd.concat(
            debt_tables,
            names=["bond_year", "repayment_year"],
        )
        .groupby("repayment_year").sum()
    )
    return aggregate_table
