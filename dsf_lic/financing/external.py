from typing import NamedTuple, Optional
from pathlib import Path

import pandas as pd

import yaml

first_year = 2018
last_year = 2070

discount = 5


class ExternalLoanInfo(NamedTuple):
    disbursement: str
    interest_rate: float
    grace_period: int
    loan_maturity: int
    repayment_schedule: Optional[str] = None
    description: Optional[str] = None


def get_external_loans_metadata() -> dict:
    with Path("dsf", "metadata", "external_loans.yaml").open() as file:
        external_loans = yaml.safe_load(file)
    return external_loans


def get_external_loan_info(name: str) -> ExternalLoanInfo:
    external_loans = get_external_loans_metadata()
    return ExternalLoanInfo(**external_loans[name])


def create_general_pv_table(loan_info: ExternalLoanInfo) -> pd.DataFrame:
    return (
        pd.DataFrame(index=pd.RangeIndex(last_year-first_year+1))
        .assign(
            amortization=
            lambda df: df.index.to_series()
            .between(loan_info.grace_period + 1, loan_info.loan_maturity)
            .mul(100).div(loan_info.loan_maturity-loan_info.grace_period)
            if loan_info.repayment_schedule is None else
            pd.read_csv("dsf/metadata/repayment_schedule.csv", index_col=0)
            .loc[loan_info.repayment_schedule]
            .rename_axis("index", axis="index").reset_index()
            .astype({"index": "Int16"}).set_index("index"),

            debt_stock=lambda df: 100 - df["amortization"].cumsum(),

            interest = lambda df: df["debt_stock"].shift(1, fill_value=0) * loan_info.interest_rate / 100,

            total_debt_service = lambda df: df["amortization"] + df["interest"],

            pv_multiplier = lambda df: pd.Series(discount, index=df.index).div(100).add(1).cumprod(),

            pv = lambda df: 
            df["total_debt_service"]
            .shift(-1 , fill_value=0)
            .div(df["pv_multiplier"])
            .iloc[::-1].cumsum().iloc[::-1]
            .mul(df["pv_multiplier"].shift(1, fill_value=1))
        )
    )


def create_external_loan_table(name: str, data: pd.DataFrame) -> pd.DataFrame:
    loan_info = get_external_loan_info(name)
    debt_table  = (
        (
            pd.Series(
                data[loan_info.disbursement],
                index = pd.RangeIndex(first_year, last_year + 1),
                name = "disbursement",
                dtype="Float64"
            )
            .fillna(0.0)
        )
        .to_frame()
        .assign(
            cumulative = lambda df: df["disbursement"].cumsum(),

            amortization = lambda df:
            df["cumulative"].shift(loan_info.grace_period + 1, fill_value=0)
            .sub(df["cumulative"].shift(loan_info.loan_maturity + 1, fill_value=0))
            .div(loan_info.loan_maturity - loan_info.grace_period),

            stock_of_new_forex_debt = lambda df:
            df["disbursement"].sub(df["amortization"]).cumsum().clip(0).round(6),

            interest = lambda df:
            df["stock_of_new_forex_debt"]
            .shift(1, fill_value=0).mul(loan_info.interest_rate).div(100),

            total_debt_service = lambda df: df["amortization"] + df["interest"],

            pv = lambda df: _calculate_pv(df, loan_info)
        )
    )
    return debt_table


def create_external_loan_table_for_all(data: pd.DataFrame) -> pd.DataFrame:
    loan_names = get_external_loans_metadata()
    loan_table = pd.concat(
        {name: create_external_loan_table(name, data) for name in loan_names},
        axis="columns",
    )
    return loan_table


def _calculate_pv(df: pd.DataFrame, loan_info: ExternalLoanInfo) -> pd.Series:
    first_period_pv = create_general_pv_table(loan_info).loc[0, "pv"]

    pv_values = []
    pv = 0
    for _, row in df.iterrows():
        pv = (
            max(pv * (1 + discount / 100) - row["total_debt_service"], 0) +
            row["disbursement"] * first_period_pv / 100
        )
        pv_values.append(pv)
    pv_series = pd.Series(pv_values, index=df.index).clip(0).round(6)
    return pv_series
