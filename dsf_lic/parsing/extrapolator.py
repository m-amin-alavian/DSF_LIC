from typing import Optional, NamedTuple, Literal

import pandas as pd

from ..utils.metadata import metadata


class ExtrapolateOptions(NamedTuple):
    method: Literal["arithmetic", "geometric"] = "arithmetic"
    number_of_periods: int = 5
    last_year: Optional[int | Literal["projection_year"]] = None
    offset: int = 0


def extrapolate(
    column: pd.Series,
    parameters: dict | bool = True,
) -> pd.Series:
    if parameters is False:
        return column

    parameters = {} if parameters is True else parameters
    options = ExtrapolateOptions(**parameters)

    if options.last_year is None:
        last_year = column.loc[lambda s: s.notna()].index[-1]
    elif options.last_year == "projection_year":
        last_year = metadata.setting.projection_year
    else:
        last_year = options.last_year
    last_value = column.loc[last_year]


    if options.method == "arithmetic":
        growth_average = (
            column
            .loc[:last_year-options.offset]
            .div(column.shift(1))
            .sub(1)
            .loc[lambda s: s.notna()]
            .iloc[-options.number_of_periods:]
            .mean()
        )
    else:
        raise NotImplementedError

    extrapolated_values = (
        pd.Series(
            growth_average,
            index=column.loc[last_year+1:].index # type: ignore
        )
        .add(1).cumprod().mul(last_value)
    )
    column.loc[last_year:] = None
    column = column.fillna(extrapolated_values)

    return column
