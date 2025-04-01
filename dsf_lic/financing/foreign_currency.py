from typing import NamedTuple, Optional

import pandas as pd

from ..utils.metadata import metadata


class DebtInfo(NamedTuple):
    disbursement: str
    interest_rate: float
    grace_period: int
    loan_maturity: int
    repayment_schedule: Optional[str] = None
    description: Optional[str] = None



def get_debt_info(name: str, external: bool = True) -> DebtInfo:
    if external:
        return DebtInfo(**metadata.external_loans[name])
    else:
        interest_rate = (
            metadata.internal_interest_rates[name]
            .loc[metadata.setting.projection_year:metadata.setting.projection_year+10]
            .mean()
        )
        return DebtInfo(
            **metadata.internal_financing[name],
            interest_rate = interest_rate
        )


def create_general_pv_table(loan_info: DebtInfo) -> pd.DataFrame:
    return (
        pd.DataFrame(
            index=pd.RangeIndex(
                metadata.setting.last_year - metadata.setting.projection_year + 1
            )
        )
        .assign(
            amortization=
            lambda df: df.index.to_series()
            .between(loan_info.grace_period + 1, loan_info.loan_maturity)
            .mul(100).div(loan_info.loan_maturity-loan_info.grace_period)
            if loan_info.repayment_schedule is None else
            metadata.repayment_schedule[loan_info.repayment_schedule],

            debt_stock=lambda df: 100 - df["amortization"].cumsum(),

            interest = lambda df:
            df["debt_stock"].shift(1, fill_value=0) * loan_info.interest_rate / 100,

            total_debt_service = lambda df: df["amortization"] + df["interest"],

            pv_multiplier = lambda df:
            pd.Series(metadata.setting.discount_rate, index=df.index)
            .div(100).add(1).cumprod(),

            pv = lambda df: 
            df["total_debt_service"].shift(-1 , fill_value=0)
            .div(df["pv_multiplier"]).iloc[::-1].cumsum().iloc[::-1]
            .mul(df["pv_multiplier"].shift(1, fill_value=1))
        )
    )


def create_debt_table(name: str, data: pd.DataFrame, external: bool = True) -> pd.DataFrame:
    loan_info = get_debt_info(name, external)
    debt_table  = (
        (
            pd.Series(
                data[loan_info.disbursement],
                index = metadata.projection_year_index,
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


def _calculate_pv(df: pd.DataFrame, loan_info: DebtInfo) -> pd.Series:
    first_period_pv = create_general_pv_table(loan_info).loc[0, "pv"]

    pv_values = []
    pv = 0
    for _, row in df.iterrows():
        pv = (
            max(
                pv *
                (1 + metadata.setting.discount_rate / 100) -
                row["total_debt_service"],
                0,
            ) +
            row["disbursement"] * first_period_pv / 100
        )
        pv_values.append(pv)
    pv_series = pd.Series(pv_values, index=df.index).clip(0).round(6)
    return pv_series
