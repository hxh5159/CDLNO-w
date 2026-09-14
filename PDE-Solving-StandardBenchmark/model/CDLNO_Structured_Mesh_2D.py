"""Darcy/Airfoil/Pipe adapter; mathematics live in the shared cdlno package."""
from cdlno.standard import StaticStandardModel


class Model(StaticStandardModel):
    structured_adapter = True
