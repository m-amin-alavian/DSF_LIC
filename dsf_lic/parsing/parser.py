from pathlib import Path
import importlib

import pandas as pd

from ..utils.metadata import metadata
from .extrapolator import extrapolate


def open_data() -> pd.DataFrame:
    metadata.reload_metadata()
    data = load_input()
    data = apply_metadata(data)
    data = data.round(6)
    return data


def load_input() -> pd.DataFrame:
    file_path = Path("data", "inputs.csv")
    return (
        pd.read_csv(file_path, index_col=0)
        .transpose()
        .pipe(lambda df: df.set_axis(df.index.astype(int))) # type: ignore
        .reindex(metadata.year_index)
    )


def apply_metadata(data: pd.DataFrame) -> pd.DataFrame:
    for column_name in metadata.variables:
        data = apply_metadata_for_column(column_name, data)
    return data


def apply_metadata_for_column(
    column_name: str,
    data: pd.DataFrame
) -> pd.DataFrame:
    column_metadata = metadata.variables[column_name]

    if column_metadata["Source"] == "Input":
        assert column_name in data.columns
    elif column_metadata["Source"] == "Calculation" and "Formula" in column_metadata:
        formula = column_metadata["Formula"]
        if isinstance(formula, dict) and "Residency_Based" in formula:
            if metadata.setting.residency_based:
                formula = formula["Residency_Based"]
            else:
                formula = formula["Currency_Based"]
        if isinstance(formula, str):
            local_dict = {
                "projection_year": metadata.setting.projection_year
            }
            data[column_name] = data.eval(formula, local_dict=local_dict)
        else:
            raise NotImplementedError
    elif column_metadata["Source"] == "Calculation" and "Function" in column_metadata:
        variable_functions = importlib.import_module("dsf_lic.metadata.variable_functions")
        function_info = column_metadata["Function"]
        if isinstance(function_info, str):
            function_name = function_info
            parameters = {}
        elif isinstance(function_info, dict):
            function_name, parameters = list(function_info.items())[0]
        parameters["data"] = data
        func = getattr(variable_functions, function_name)
        data[column_name] = func(**parameters)
    else:
        raise KeyError
        

    if "Extrapolate" in column_metadata:
        data[column_name] = extrapolate(
            data[column_name],
            column_metadata["Extrapolate"]
        )

    return data
