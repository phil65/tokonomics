"""The public cost types resolve at runtime (pydantic / get_type_hints users)."""

from decimal import Decimal
import typing

from pydantic import TypeAdapter

from tokonomics.toko_types import ModelCosts, ModelCostTier, TokenCosts


def test_cost_types_resolve():
    for tp in (ModelCostTier, ModelCosts, TokenCosts):
        hints = typing.get_type_hints(tp)
        assert Decimal in {
            typing.get_args(h)[0] if typing.get_args(h) else h for h in hints.values()
        }


def test_model_costs_validate_with_pydantic():
    costs = TypeAdapter(ModelCosts).validate_python({
        "input_cost_per_token": "0.000001",
        "output_cost_per_token": "0.000002",
    })
    assert costs["input_cost_per_token"] == Decimal("0.000001")
