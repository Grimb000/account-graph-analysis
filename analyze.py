import pandas as pd
import networkx as nx


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


if __name__ == "__main__":
    df = load_transfers("data/transfers.csv")
    dust_df, real_df = split_dust_real(df)

    G_full = build_graph(df)
    G_real = build_graph(real_df)

    print(f"full graph: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
    print(f"real graph: {G_real.number_of_nodes()} nodes, {G_real.number_of_edges()} edges")