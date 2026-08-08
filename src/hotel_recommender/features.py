from __future__ import annotations

import numpy as np
import pandas as pd

from .data import validate_catalog, validate_interactions
from .domain import EVENT_WEIGHTS, FEATURE_NAMES


def build_item_stats(history: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    validate_interactions(history)
    validate_catalog(items)

    event_counts = (
        history.assign(value=1)
        .pivot_table(
            index="hotel_id",
            columns="event",
            values="value",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=["view", "click", "booking"], fill_value=0)
    )
    weighted = (
        history.assign(weight=history["event"].map(EVENT_WEIGHTS))
        .groupby("hotel_id")["weight"]
        .sum()
    )
    stats = (
        items[["hotel_id"]]
        .set_index("hotel_id")
        .join(event_counts, how="left")
        .join(weighted.rename("weighted_events"), how="left")
        .fillna(0.0)
    )

    max_weight = max(float(stats["weighted_events"].max()), 1.0)
    exposures = stats["view"] + stats["click"] + stats["booking"]
    stats["item_popularity"] = stats["weighted_events"] / max_weight
    stats["historical_booking_rate"] = (stats["booking"] + 1.0) / (exposures + 20.0)
    return stats.reset_index()


def build_user_profiles(history: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    joined = history.merge(items, on="hotel_id", how="inner", validate="many_to_one")
    joined = joined.assign(weight=joined["event"].map(EVENT_WEIGHTS).astype(float))

    base = joined.groupby("user_id").apply(
        lambda frame: pd.Series(
            {
                "mean_price": np.average(frame["price"], weights=frame["weight"]),
                "mean_stars": np.average(frame["stars"], weights=frame["weight"]),
            }
        ),
        include_groups=False,
    )

    city = joined.groupby(["user_id", "city"], as_index=False)["weight"].sum()
    city_total = city.groupby("user_id")["weight"].transform("sum")
    city["city_affinity"] = city["weight"] / city_total.clip(lower=1e-12)

    kind = joined.groupby(["user_id", "hotel_type"], as_index=False)["weight"].sum()
    kind_total = kind.groupby("user_id")["weight"].transform("sum")
    kind["type_affinity"] = kind["weight"] / kind_total.clip(lower=1e-12)

    profiles = base.reset_index()
    profiles.attrs["city_affinity"] = city[["user_id", "city", "city_affinity"]]
    profiles.attrs["type_affinity"] = kind[["user_id", "hotel_type", "type_affinity"]]
    return profiles


def candidate_features(
    user_id: int,
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    items: pd.DataFrame,
    item_stats: pd.DataFrame,
    user_profiles: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=FEATURE_NAMES, index=candidates.index)

    stats = item_stats.set_index("hotel_id")
    profile_row = user_profiles[user_profiles["user_id"] == user_id]
    if profile_row.empty:
        mean_price = float(items["price"].median())
        mean_stars = float(items["stars"].median())
    else:
        mean_price = float(profile_row.iloc[0]["mean_price"])
        mean_stars = float(profile_row.iloc[0]["mean_stars"])

    city_table = user_profiles.attrs.get("city_affinity", pd.DataFrame())
    type_table = user_profiles.attrs.get("type_affinity", pd.DataFrame())
    city_lookup = (
        city_table[city_table["user_id"] == user_id]
        .set_index("city")["city_affinity"]
        .to_dict()
        if not city_table.empty
        else {}
    )
    type_lookup = (
        type_table[type_table["user_id"] == user_id]
        .set_index("hotel_type")["type_affinity"]
        .to_dict()
        if not type_table.empty
        else {}
    )

    user_history = history[history["user_id"] == user_id]
    prior = user_history.pivot_table(
        index="hotel_id",
        columns="event",
        values="timestamp",
        aggfunc="count",
        fill_value=0,
    ).reindex(columns=["view", "click"], fill_value=0)

    out = pd.DataFrame(index=candidates.index)
    out["item_popularity"] = candidates["hotel_id"].map(stats["item_popularity"]).fillna(0.0)
    out["city_affinity"] = candidates["city"].map(city_lookup).fillna(0.0)
    out["type_affinity"] = candidates["hotel_type"].map(type_lookup).fillna(0.0)
    out["price_fit"] = np.exp(-np.abs(candidates["price"] - mean_price) / max(mean_price, 1.0))
    out["star_fit"] = 1.0 - (np.abs(candidates["stars"] - mean_stars) / 4.0).clip(0.0, 1.0)
    out["prior_views"] = (
        candidates["hotel_id"]
        .map(prior["view"] if "view" in prior else {})
        .fillna(0)
        .clip(upper=10)
        / 10.0
    )
    out["prior_clicks"] = (
        candidates["hotel_id"]
        .map(prior["click"] if "click" in prior else {})
        .fillna(0)
        .clip(upper=10)
        / 10.0
    )
    out["historical_booking_rate"] = (
        candidates["hotel_id"].map(stats["historical_booking_rate"]).fillna(0.0)
    )
    out["log_price"] = np.log1p(candidates["price"]) / 10.0
    out["stars_scaled"] = candidates["stars"] / 5.0
    return out.loc[:, list(FEATURE_NAMES)].astype(float)
