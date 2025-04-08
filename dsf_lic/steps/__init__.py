from .old_debt_pv import main as add_old_debt_pv
from .foreign_currency_financing import main as add_foreign_currency_financing
from .local_currency_financing import main as add_local_currency_financing
from .mlt_debts import main as add_mlt_debts
from .per_gdp import main as add_per_gdp


__all__ = [
    "add_old_debt_pv",
    "add_foreign_currency_financing",
    "add_local_currency_financing",
    "add_mlt_debts",
    "add_per_gdp",
]
