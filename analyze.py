import sys

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import zscore


def load_transfers(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    before = len(df)
    df = df.drop_duplicates()
    if before - len(df):
        print(f"убрал {before - len(df)} дублей")

    return df.sort_values("timestamp").reset_index(drop=True)


def split_dust_real(df):
    dust_amount = df["amount"].min()
    dust = df[df["amount"] == dust_amount]
    real = df[df["amount"] > dust_amount]
    print(f"dust: {len(dust)}, real: {len(real)}")
    return dust, real


def build_graph(df):
    edges = df.groupby(["from", "to"]).agg(weight=("amount", "sum"), count=("amount", "size")).reset_index()

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["from"], row["to"], weight=row["weight"], count=row["count"])

    return G


def _find_batches_windowed(df, window_sec=1.0):
    """Group outgoing transfers by sender and a sliding 1-second window."""
    batches = []
    for sender, grp in df.groupby("from"):
        grp = grp.sort_values("timestamp")
        times = grp["timestamp"].values
        tos = grp["to"].values
        used = np.zeros(len(grp), dtype=bool)
        for i in range(len(grp)):
            if used[i]:
                continue
            t0 = times[i]
            window_mask = (
                (times >= t0)
                & (times <= t0 + np.timedelta64(int(window_sec * 1000), "ms"))
                & (~used)
            )
            batches.append({"from": sender, "t0": t0, "to": list(tos[window_mask])})
            used |= window_mask
    return pd.DataFrame(batches)


def suspicion_score(df: pd.DataFrame, feat_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Composite suspicion score per account.

    Mirrors the scoring used in notebook.ipynb: weighted z-scores of
    out_degree, batch_count, jan_active, recipient_repeat_ratio and velocity.
    """
    all_accounts = sorted(set(df["from"]) | set(df["to"]))

    # 1. out_degree
    out_degree = df.groupby("from").size().reindex(all_accounts, fill_value=0)

    # 2. batch_count (1-second windowed batches)
    batch_windowed = _find_batches_windowed(df, window_sec=1.0)
    batch_count = batch_windowed.groupby("from").size().reindex(all_accounts, fill_value=0)

    # 3. jan_active flag (sender activity in Jan 10-15 window)
    jan_window = (df["timestamp"] >= "2026-01-10") & (df["timestamp"] < "2026-01-15")
    jan_senders = set(df.loc[jan_window, "from"])
    jan_active = pd.Series([1 if acc in jan_senders else 0 for acc in all_accounts], index=all_accounts)

    # 4. recipient_repeat_ratio
    sender_repeat = {}
    for sender, grp in df.groupby("from"):
        recipients = grp["to"].tolist()
        seen = set()
        repeats = 0
        for r in recipients:
            if r in seen:
                repeats += 1
            seen.add(r)
        sender_repeat[sender] = repeats / len(recipients) if recipients else 0.0
    recipient_repeat = pd.Series([sender_repeat.get(acc, 0.0) for acc in all_accounts], index=all_accounts)

    # 5. velocity: median time from receipt to next send
    velocities = {}
    for acc in all_accounts:
        in_tx = df[df["to"] == acc][["timestamp"]].sort_values("timestamp")
        out_tx = df[df["from"] == acc][["timestamp"]].sort_values("timestamp")
        deltas = []
        for _, in_row in in_tx.iterrows():
            later = out_tx[out_tx["timestamp"] > in_row["timestamp"]]
            if not later.empty:
                deltas.append((later.iloc[0]["timestamp"] - in_row["timestamp"]).total_seconds())
        velocities[acc] = np.median(deltas) if deltas else np.nan
    velocity = pd.Series(velocities)
    velocity_filled = velocity.fillna(velocity.median())

    # Build score components
    score_components = pd.DataFrame(index=all_accounts)
    score_components["out_degree_z"] = zscore(out_degree.values)
    score_components["batch_count_z"] = zscore(batch_count.values)
    score_components["jan_active"] = jan_active.values
    score_components["recipient_repeat_z"] = zscore(recipient_repeat.values)
    score_components["velocity_z"] = -zscore(velocity_filled.values)  # lower velocity = more suspicious

    weights = {
        "out_degree_z": 0.35,
        "batch_count_z": 0.35,
        "jan_active": 0.10,
        "recipient_repeat_z": 0.10,
        "velocity_z": 0.10,
    }
    score_components["score"] = sum(score_components[col] * w for col, w in weights.items())

    # Format output
    result = score_components.reset_index().rename(columns={"index": "account"})
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result


def _top_components(row):
    """Return the two component names with largest absolute contribution."""
    contrib_cols = ["out_degree_z", "batch_count_z", "jan_active", "recipient_repeat_z", "velocity_z"]
    contrib = {col: row[col] for col in contrib_cols}
    sorted_contrib = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    return ", ".join(f"{col}={row[col]:.2f}" for col, _ in sorted_contrib[:2])


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/transfers.csv"

    df = load_transfers(path)
    dust_df, real_df = split_dust_real(df)

    G_full = build_graph(df)
    G_real = build_graph(real_df)

    print(f"full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
    print(f"real graph: {G_real.number_of_nodes()} nodes, {G_real.number_of_edges()} edges")

    scores = suspicion_score(df)
    print("\nTop-10 accounts by suspicion score:")
    for _, row in scores.head(10).iterrows():
        print(f"  {row['account']:10s}  score={row['score']:7.3f}  ({_top_components(row)})")
