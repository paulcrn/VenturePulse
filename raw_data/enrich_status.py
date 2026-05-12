"""
enrich_status.py
================
Enriches investments_VC.csv with current status data from the
HuggingFace Crunchbase dataset (opensporks/crunchbase, as of Aug 2024).

Matching logic:
  Our dataset:  permalink = "/organization/slack"
  HF dataset:   permalink = "slack"
  → Strip "/organization/" → direct join

Important:
  - The original 'status' from investments_VC.csv is NOT modified
  - Instead, a new column 'status_enriched' is added
  - Mapping HF → original categories:
      HF "closed"  → "closed"
      HF "active"  → "operating"
      no match     → original value kept

Installation (one-time):
  pip install datasets pandas

Run:
  python enrich_status.py

  Optional: adjust paths
  python enrich_status.py --input data/investments_VC.csv --output data/enriched_investments.csv
"""

import argparse
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_DATASET_ID = "opensporks/crunchbase"
HF_COLUMNS    = ["permalink", "operating_status"]

DEFAULT_INPUT  = "investments_VC.csv"
DEFAULT_OUTPUT = "enriched_investments.csv"

# Mapping HF values → original categories from investments_VC.csv
HF_STATUS_MAP = {
    "active":  "operating",
    "closed":  "closed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_permalink(p: str) -> str:
    """'/organization/slack'  →  'slack'"""
    if not isinstance(p, str):
        return ""
    return p.replace("/organization/", "").strip().lower()


def load_hf_dataset(hf_id: str, columns: list) -> pd.DataFrame:
    """
    Loads the HuggingFace dataset in streaming mode.
    Streaming avoids loading all 2.81M rows into RAM at once.
    Takes about 3–5 minutes.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Das 'datasets'-Paket fehlt.\n"
            "Bitte installieren mit:  pip install datasets"
        )

    print(f"[1/4] Lade HuggingFace-Dataset '{hf_id}' (Streaming)...")
    print("      Kann 3–5 Minuten dauern bei 2.81M Zeilen.\n")

    ds   = load_dataset(hf_id, split="train", streaming=True)
    rows = []

    for i, row in enumerate(ds):
        rows.append({col: row.get(col) for col in columns})
        if (i + 1) % 500_000 == 0:
            print(f"      {i+1:,} Zeilen geladen...")

    hf_df = pd.DataFrame(rows)
    print(f"      Fertig — {len(hf_df):,} Zeilen geladen.")
    print(f"      HF operating_status Verteilung:\n"
          f"{hf_df['operating_status'].value_counts(dropna=False).to_string()}\n")
    return hf_df


def load_investments(path: str) -> pd.DataFrame:
    """Loads investments_VC.csv with the correct encoding and column cleanup."""
    print(f"[2/4] Lade '{path}'...")
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df = df.rename(columns={
        " market ":            "market",
        " funding_total_usd ": "funding_total_usd"
    })
    print(f"      {len(df):,} Zeilen geladen.")
    print(f"      Status-Verteilung (original):\n"
          f"{df['status'].value_counts(dropna=False).to_string()}\n")
    return df


def merge_and_enrich(df: pd.DataFrame, hf: pd.DataFrame) -> pd.DataFrame:
    """
    Join on permalink, then populate status_enriched.

    Rules:
      1. Non-operating companies (acquired, closed):
             status_enriched = status  (unchanged)
      2. Operating companies WITH an HF match:
             HF "closed"  → status_enriched = "closed"
             HF "active"  → status_enriched = "operating"
      3. Operating companies WITHOUT an HF match:
             status_enriched = "operating"  (unchanged)

    status_enriched only contains values that also appear in the original:
    "operating", "acquired", "closed"
    """
    print("[3/4] Matche Permalinks und erstelle status_enriched...")

    # Normalize permalinks
    df["_pl"] = df["permalink"].apply(clean_permalink)
    hf["_pl"] = hf["permalink"].apply(
        lambda p: p.strip().lower() if isinstance(p, str) else ""
    )

    # Build lookup dict: permalink_clean → HF operating_status
    hf_lookup = (
        hf.dropna(subset=["_pl"])
        .set_index("_pl")["operating_status"]
        .to_dict()
    )

    # Only operating companies are in scope
    operating_mask = df["status"] == "operating"
    n_operating    = operating_mask.sum()
    print(f"      Operating companies im Datensatz: {n_operating:,}")

    # Initialize status_enriched with original values (acquired + closed unchanged)
    df["status_enriched"] = df["status"].copy()

    # For operating companies: look up the HF status and map it to original categories
    def resolve(permalink_clean: str) -> str:
        hf_val = hf_lookup.get(permalink_clean)       # None if no match
        if hf_val is None:
            return "operating"                         # no match → unchanged
        return HF_STATUS_MAP.get(hf_val, "operating") # map to original category

    df.loc[operating_mask, "status_enriched"] = (
        df.loc[operating_mask, "_pl"].apply(resolve)
    )

    # Clean up temp column
    df.drop(columns=["_pl"], inplace=True)

    # Compute statistics
    n_now_closed = (operating_mask & (df["status_enriched"] == "closed")).sum()
    n_with_match = sum(
        1 for pl in df.loc[operating_mask, "permalink"].apply(clean_permalink)
        if pl in hf_lookup
    )

    print(f"      Matches gefunden:        {n_with_match:,} "
          f"({n_with_match / n_operating * 100:.1f}%)")
    print(f"        → neu als 'closed':    {n_now_closed:,}")
    print(f"        → bleibt 'operating':  {n_with_match - n_now_closed:,}")
    print(f"      Kein Match (operating):  {n_operating - n_with_match:,}\n")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Concise closing report."""
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)

    orig     = df["status"].value_counts(dropna=False)
    enriched = df["status_enriched"].value_counts(dropna=False)

    print("\nStatus original  →  status_enriched")
    print("-" * 40)
    for val in ["operating", "acquired", "closed"]:
        o = orig.get(val, 0)
        e = enriched.get(val, 0)
        diff = e - o
        diff_str = f"  ({diff:+,})" if diff != 0 else ""
        print(f"  {val:<12}  {o:>7,}  →  {e:>7,}{diff_str}")

    n_updated = (
        (df["status"] == "operating") &
        (df["status_enriched"] == "closed")
    ).sum()

    print(f"\n→ {n_updated:,} Companies von 'operating' auf 'closed' aktualisiert")
    print(f"→ Datenquelle: HuggingFace opensporks/crunchbase (Stand Aug 2024)")
    print(f"→ Originale 'status'-Spalte ist unverändert erhalten")
    print("=" * 55)
    print()
    print("Nutzung im Notebook:")
    print("  df = pd.read_csv('enriched_investments.csv', encoding='utf-8')")
    print("  # 'status'          → Original (unverändert)")
    print("  # 'status_enriched' → Angereichert, gleiche Kategorien")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enrich investments_VC.csv with HuggingFace Crunchbase status data"
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT,
                        help=f"Pfad zur investments_VC.csv (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Ausgabepfad (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    # 1. Load investments
    df = load_investments(args.input)

    # 2. Load HuggingFace dataset
    hf = load_hf_dataset(HF_DATASET_ID, HF_COLUMNS)

    # 3. Merge & enrich
    df_enriched = merge_and_enrich(df, hf)

    # 4. Save
    output_path = Path(args.output)
    print(f"[4/4] Speichere nach '{output_path}'...")
    df_enriched.to_csv(output_path, index=False, encoding="utf-8")
    print(f"      Gespeichert ({output_path.stat().st_size / 1e6:.1f} MB)\n")

    # 5. Summary
    print_summary(df_enriched)


if __name__ == "__main__":
    main()
