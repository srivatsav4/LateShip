"""Pydantic request/response models. Unknown fields (e.g. review_score) are rejected with 422."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config import BR_STATES


class OrderRequest(BaseModel):
    """Information available at purchase time. Seller-history fields come from a point-in-time lookup."""

    model_config = ConfigDict(extra="forbid")

    estimated_delivery_days: float = Field(gt=0, le=365, allow_inf_nan=False, description="Promised days from purchase")
    freight_value: float = Field(ge=0, le=100_000, allow_inf_nan=False)
    product_price: float = Field(ge=0, le=1_000_000, allow_inf_nan=False, description="Total item price")
    purchase_weekday: int = Field(ge=0, le=6, description="0=Monday")
    purchase_month: int = Field(ge=1, le=12)
    item_count: int = Field(ge=1, le=100)
    seller_state: str
    customer_state: str
    seller_customer_distance_km: Optional[float] = Field(default=None, ge=0, le=20_000, allow_inf_nan=False)
    product_weight: Optional[float] = Field(default=None, ge=0, le=1_000_000, allow_inf_nan=False, description="grams")
    product_volume: Optional[float] = Field(default=None, ge=0, le=100_000_000, allow_inf_nan=False, description="cm^3")
    product_category: Optional[str] = Field(default=None, max_length=100)
    seller_hist_order_count: int = Field(default=0, ge=0)
    seller_hist_late_rate: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    seller_orders_last_30d: int = Field(default=0, ge=0)

    @field_validator("seller_state", "customer_state")
    @classmethod
    def _valid_state(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in BR_STATES:
            raise ValueError(f"must be a Brazilian state code (UF), got '{v}'")
        return v


class PredictResponse(BaseModel):
    late_probability: float
    risk_level: str
    flag: bool
    threshold: float
    recommended_action: str
    model_version: str


class Contribution(BaseModel):
    feature: str
    value: Optional[float | str | int | bool] = None
    impact: float
    direction: str


class ExplainResponse(PredictResponse):
    base_value_log_odds: float
    raw_margin_log_odds: float
    contributions: list[Contribution]
    explanation_note: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
