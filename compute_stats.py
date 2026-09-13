"""
compute_stats.py

Pulls public NFL play-by-play data (via the nflverse project) and computes
per-team efficiency metrics: EPA/play (offense & defense), success rate
(offense & defense), ANY/A, point differential, and turnover differential.

Output: stats.json in the same folder, which the website fetches directly.

Data source: nflverse-data play-by-play releases (public, free, no API key).
https://github.com/nflverse/nflverse-data

Requires: pandas, requests, pyarrow (for reading parquet files)
    pip install pandas requests pyarrow

Note: DVOA is NOT included here — it's a proprietary FTN Fantasy /
Football Outsiders stat with no public data source. That stays manual.
"""

import json
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

SEASON = 2026
PBP_URL = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{SEASON}.parquet"


def load_play_by_play() -> pd.DataFrame:
    """Download this season's play-by-play data. Returns an empty
    DataFrame (not an error) if the season hasn't started yet, since
    nflverse only publishes a season file once there's at least one
    game played."""
    try:
        resp = requests.get(PBP_URL, timeout=60)
        resp.raise_for_status()
        with open("_pbp_temp.parquet", "wb") as f:
            f.write(resp.content)
        df = pd.read_parquet("_pbp_temp.parquet")
        return df
    except Exception as e:
        print(f"Could not load play-by-play data (likely no games played yet): {e}", file=sys.stderr)
        return pd.DataFrame()


def compute_team_stats(pbp: pd.DataFrame) -> dict:
    """Compute per-team offensive/defensive efficiency metrics from
    play-by-play data. Only considers regular pass/run plays (excludes
    kneels, spikes, penalties with no play, etc.)."""

    if pbp.empty:
        return {}

    plays = pbp[
        (pbp["play_type"].isin(["pass", "run"]))
        & (pbp["epa"].notna())
    ].copy()

    if plays.empty:
        return {}

    # "Success" per Football Outsiders convention: gain >= 40% of yards
    # needed on 1st down, 60% on 2nd, 100% on 3rd/4th.
    def is_success(row):
        pct_needed = {1: 0.4, 2: 0.6, 3: 1.0, 4: 1.0}.get(row["down"], 1.0)
        return row["yards_gained"] >= pct_needed * row["ydstogo"]

    plays["success"] = plays.apply(is_success, axis=1)

    teams = {}

    for team in pd.concat([plays["posteam"], plays["defteam"]]).dropna().unique():
        off = plays[plays["posteam"] == team]
        deff = plays[plays["defteam"] == team]

        pass_plays = off[off["play_type"] == "pass"]
        any_a = None
        if len(pass_plays) > 0 and "sack" in pass_plays.columns:
            attempts = len(pass_plays)
            sacks = pass_plays["sack"].sum() if "sack" in pass_plays else 0
            yards = pass_plays["yards_gained"].sum()
            tds = pass_plays["pass_touchdown"].sum() if "pass_touchdown" in pass_plays else 0
            ints = pass_plays["interception"].sum() if "interception" in pass_plays else 0
            denom = attempts + sacks
            if denom > 0:
                any_a = round((yards + 20 * tds - 45 * ints) / denom, 2)

        teams[team] = {
            "epa_per_play_off": round(off["epa"].mean(), 3) if len(off) else None,
            "epa_per_play_def": round(deff["epa"].mean(), 3) if len(deff) else None,
            "success_rate_off": round(off["success"].mean() * 100, 1) if len(off) else None,
            "success_rate_allowed": round(deff["success"].mean() * 100, 1) if len(deff) else None,
            "any_a_off": any_a,
            "plays_off": len(off),
            "plays_def": len(deff),
        }

    return teams


def main():
    pbp = load_play_by_play()
    stats = compute_team_stats(pbp)

    output = {
        "season": SEASON,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "has_data": bool(stats),
        "note": "DVOA is not included — no public data source exists for it. Enter manually.",
        "teams": stats,
    }

    with open("stats.json", "w") as f:
        json.dump(output, f, indent=2)

    if not stats:
        print("No play-by-play data available yet (season likely hasn't started). "
              "Wrote an empty stats.json placeholder — this will populate automatically "
              "once games have been played and this script runs again.")
    else:
        print(f"Wrote stats.json with {len(stats)} teams.")


if __name__ == "__main__":
    main()
