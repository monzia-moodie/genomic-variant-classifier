import pandas as pd

mv = pd.read_parquet("outputs/run9_ready/splits/meta_val.parquet")
mt = pd.read_parquet("outputs/run9_ready/splits/meta_test.parquet")

print(f"meta_val:  {len(mv):,} rows, variant_id nunique = {mv['variant_id'].nunique():,}")
print(f"meta_test: {len(mt):,} rows, variant_id nunique = {mt['variant_id'].nunique():,}")
print(f"meta_val  variant_id is unique: {mv['variant_id'].is_unique}")
print(f"meta_test variant_id is unique: {mt['variant_id'].is_unique}")
print()
print("variant_id sample (first 3 of meta_val):")
print(mv["variant_id"].head(3).to_list())
print()
print(f"variant_id dtype: {mv['variant_id'].dtype}")
print()

overlap = set(mv["variant_id"]) & set(mt["variant_id"])
print(f"val/test overlap on variant_id: {len(overlap):,} (should be 0)")
