import pandas as pd

mv = pd.read_parquet("outputs/run9_ready/splits/meta_val.parquet")
mt = pd.read_parquet("outputs/run9_ready/splits/meta_test.parquet")

# 1. What is the dup pattern within meta_val?
dup_mask = mv["variant_id"].duplicated(keep=False)
print(f"meta_val rows that are part of a duplicate variant_id: {dup_mask.sum():,}")
print()
print("Sample duplicate group (first variant_id with duplicates):")
first_dup = mv.loc[dup_mask, "variant_id"].iloc[0]
sample = mv[mv["variant_id"] == first_dup][["variant_id", "gene_symbol", "transcript_id", "consequence", "label"]]
print(sample.to_string())
print()

# 2. How does (variant_id + transcript_id) behave?
combo = mv["variant_id"].astype(str) + "|" + mv["transcript_id"].astype(str)
print(f"meta_val (variant_id + transcript_id) nunique: {combo.nunique():,} of {len(mv):,}")
combo_t = mt["variant_id"].astype(str) + "|" + mt["transcript_id"].astype(str)
print(f"meta_test (variant_id + transcript_id) nunique: {combo_t.nunique():,} of {len(mt):,}")
print()

# 3. The 46 overlap variants: are they same gene? Different transcripts?
overlap = set(mv["variant_id"]) & set(mt["variant_id"])
print(f"Overlap variant_ids (showing first 5):")
for vid in list(overlap)[:5]:
    val_rows = mv[mv["variant_id"] == vid][["gene_symbol", "transcript_id", "consequence", "label"]]
    test_rows = mt[mt["variant_id"] == vid][["gene_symbol", "transcript_id", "consequence", "label"]]
    print(f"\n  {vid}:")
    print(f"    in meta_val:  {len(val_rows)} rows, genes={val_rows['gene_symbol'].unique().tolist()}")
    print(f"    in meta_test: {len(test_rows)} rows, genes={test_rows['gene_symbol'].unique().tolist()}")

# 4. Are the 46 overlapping variants in the SAME gene (potential leak) or DIFFERENT genes (transcript artifact)?
overlap_same_gene = 0
overlap_diff_gene = 0
for vid in overlap:
    val_genes = set(mv[mv["variant_id"] == vid]["gene_symbol"].dropna())
    test_genes = set(mt[mt["variant_id"] == vid]["gene_symbol"].dropna())
    if val_genes & test_genes:
        overlap_same_gene += 1
    else:
        overlap_diff_gene += 1
print(f"\nOf 46 overlap variants:")
print(f"  same gene in val and test:      {overlap_same_gene} (potential split leak)")
print(f"  different genes in val and test: {overlap_diff_gene} (transcript multimapping artifact)")
