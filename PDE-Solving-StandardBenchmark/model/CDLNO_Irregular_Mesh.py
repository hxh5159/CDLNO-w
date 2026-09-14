"""Elasticity adapter; no image reshape or point reordering."""
from cdlno.standard import StaticStandardModel


class Model(StaticStandardModel):
    structured_adapter = False
