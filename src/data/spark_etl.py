"""
Spark ETL Pipeline for Genomic Variant Classification
=======================================================
Transforms canonical DataFrames (from database_connectors.py) into a
queryable, enriched variant database using PySpark.

Pipeline stages:
  1. Ingest     — read raw parquet/CSV from data/raw/
  2. Normalize  — standardize chromosomes, alleles, coordinates
  3. Enrich     — attach ACMG evidence codes and derived features
  4. Deduplicate — variant_id-level dedup across sources
  5. Write       — partition by chromosome to data/processed/

CHANGES FROM PHASE 1:
  - This module was never written to disk in Phase 1 (Bug 3 fixed).
  - Duplicate create_spark_session definitions (cells 49, 71, 72) merged
    into one canonical version here (Issue M).
  - Cell 71 (JAR-conflict version) removed; simpler cell 72 logic used.
  - from __future__ import annotations added (Issue N).
  - Unused `import pyspark` at top of create_spark_session fixed.
  - regex_extract filter fixed: .cast("boolean") replaced with != "" check
    (Spark's regexp_extract returns "" not null on no-match).

Run locally:
    python -m src.data.spark_etl
Or via CLI:
    spark-submit src/data/spark_etl.py
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
VARIANT_SCHEMA = StructType([
    StructField("variant_id",     StringType(),  False),
    StructField("source_db",      StringType(),  True),
    StructField("chrom",          StringType(),  True),
    StructField("pos",            IntegerType(), True),
    StructField("ref",            StringType(),  True),
    StructField("alt",            StringType(),  True),
    StructField("gene_symbol",    StringType(),  True),
    StructField("transcript_id",  StringType(),  True),
    StructField("consequence",    StringType(),  True),
    StructField("pathogenicity",  StringType(),  True),
    StructField("allele_freq",    FloatType(),   True),
    StructField("clinical_sig",   StringType(),  True),
    StructField("protein_change", StringType(),  True),
    StructField("fasta_seq",      StringType(),  True),
    StructField("source_id",      StringType(),  True),
])

# Chromosome normalization map
CHROM_MAP: dict[str, str] = {
    **{str(i):      str(i) for i in range(1, 23)},
    **{f"chr{i}":   str(i) for i in range(1, 23)},
    "X":    "X",  "chrX":  "X",
    "Y":    "Y",  "chrY":  "Y",
    "MT":   "MT", "chrM":  "MT", "chrMT": "MT", "M": "MT",
}

ACMG_PATHOGENIC = ["pathogenic", "likely_pathogenic"]
ACMG_BENIGN     = ["benign",     "likely_benign"]


# ---------------------------------------------------------------------------
# Spark session factory (single canonical version — Issue M)
# ---------------------------------------------------------------------------
def create_spark_session(
    app_name:        str = "GenomicVariantETL",
    master:          str = "local[*]",
    executor_memory: str = "4g",
    driver_memory:   str = "8g",
) -> SparkSession:
    """
    Create or retrieve a SparkSession.

    Uses the default PySpark configuration. JAR management is handled
    by PySpark's bundled installation and does not require manual wget.
    Adaptive Query Execution (AQE) is enabled for efficient shuffle handling.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory",   driver_memory)
        .config("spark.sql.shuffle.partitions",              "200")
        .config("spark.sql.adaptive.enabled",                "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Stage 1: Ingest
# ---------------------------------------------------------------------------
def ingest(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Read all parquet/CSV files from raw_dir into a single Spark DataFrame."""
    raw_path  = Path(raw_dir)
    parquet_files = list(raw_path.glob("**/*.parquet"))
    csv_files     = list(raw_path.glob("**/*.csv"))

    dfs: list[DataFrame] = []

    if parquet_files:
        pq_df = spark.read.parquet(str(raw_path / "**" / "*.parquet"))
        dfs.append(pq_df)
        logger.info("Ingested parquet files.")

    if csv_files:
        csv_df = (
            spark.read
            .option("header",       "true")
            .option("inferSchema",  "false")
            .csv(str(raw_path / "**" / "*.csv"))
        )
        # Cast columns that exist in both schema and CSV
        for field_def in VARIANT_SCHEMA:
            if field_def.name in csv_df.columns:
                csv_df = csv_df.withColumn(
                    field_def.name,
                    F.col(field_def.name).cast(field_def.dataType),
                )
        dfs.append(csv_df)

    if not dfs:
        logger.warning("No data files found in %s. Returning empty DataFrame.", raw_dir)
        return spark.createDataFrame([], VARIANT_SCHEMA)

    result = dfs[0]
    for df in dfs[1:]:
        result = result.unionByName(df, allowMissingColumns=True)
    return result


# ---------------------------------------------------------------------------
# Stage 2: Normalize
# ---------------------------------------------------------------------------
def normalize(df: DataFrame) -> DataFrame:
    """
    Standardize chromosome names, upper-case alleles, and filter invalid rows.

    CHANGE: regexp_extract().cast("boolean") → regexp_extract() != ""
    Spark's regexp_extract returns an empty string (not null) when there
    is no match, so .cast("boolean") would evaluate "" as False correctly,
    but is less readable and semantically confusing. Using != "" is explicit.
    """
    chrom_map_expr = F.create_map(
        *[val for pair in [(F.lit(k), F.lit(v)) for k, v in CHROM_MAP.items()] for val in pair]
    )

    valid_allele_pattern = r"^[ACGTNacgtn*-]+$"

    df = (
        df
        .withColumn("chrom", F.coalesce(chrom_map_expr[F.col("chrom")], F.col("chrom")))
        .withColumn("ref",   F.upper(F.col("ref")))
        .withColumn("alt",   F.upper(F.col("alt")))
        .filter(
            F.col("ref").isNotNull()   &
            F.col("alt").isNotNull()   &
            F.col("chrom").isNotNull() &
            F.col("pos").isNotNull()   &
            (F.regexp_extract(F.col("ref"), valid_allele_pattern, 0) != "") &
            (F.regexp_extract(F.col("alt"), valid_allele_pattern, 0) != "")
        )
        .withColumn(
            "variant_id",
            F.when(
                F.col("variant_id").isNull(),
                F.concat_ws(
                    ":", F.col("source_db"), F.col("chrom"),
                    F.col("pos").cast(StringType()), F.col("ref"), F.col("alt"),
                ),
            ).otherwise(F.col("variant_id")),
        )
    )
    return df


# ---------------------------------------------------------------------------
# Stage 3: Enrich
# ---------------------------------------------------------------------------
def enrich(df: DataFrame) -> DataFrame:
    """
    Attach derived features:
      - acmg_label       : binary ACMG classification (1=pathogenic, 0=benign)
      - variant_type     : SNV | insertion | deletion | complex
      - allele_freq_bin  : AF category for stratified analysis
    """
    df = (
        df
        .withColumn(
            "acmg_label",
            F.when(F.col("pathogenicity").isin(ACMG_PATHOGENIC), 1)
             .when(F.col("pathogenicity").isin(ACMG_BENIGN),     0)
             .otherwise(F.lit(None).cast(IntegerType())),
        )
        .withColumn("ref_len", F.length(F.col("ref")))
        .withColumn("alt_len", F.length(F.col("alt")))
        .withColumn(
            "variant_type",
            F.when((F.col("ref_len") == 1) & (F.col("alt_len") == 1), "SNV")
             .when(F.col("ref_len") < F.col("alt_len"),                "insertion")
             .when(F.col("ref_len") > F.col("alt_len"),                "deletion")
             .otherwise("complex"),
        )
        .withColumn(
            "allele_freq_bin",
            F.when(F.col("allele_freq").isNull(),   "unknown")
             .when(F.col("allele_freq") < 0.0001,   "ultra_rare")
             .when(F.col("allele_freq") < 0.001,    "rare")
             .when(F.col("allele_freq") < 0.01,     "low_freq")
             .when(F.col("allele_freq") < 0.05,     "common")
             .otherwise("very_common"),
        )
        .drop("ref_len", "alt_len")
    )
    return df


# ---------------------------------------------------------------------------
# Stage 4: Deduplicate
# ---------------------------------------------------------------------------
def deduplicate(df: DataFrame) -> DataFrame:
    """
    Dedup on variant_id, preferring ClinVar > gnomAD > UniProt > others
    for the authoritative pathogenicity label.
    """
    source_priority = (
        F.when(F.col("source_db") == "clinvar",  1)
         .when(F.col("source_db") == "gnomad",   2)
         .when(F.col("source_db") == "uniprot",  3)
         .otherwise(4)
    )
    window = Window.partitionBy("variant_id").orderBy(source_priority)

    df = (
        df
        .withColumn("_priority_rank", F.row_number().over(window))
        .filter(F.col("_priority_rank") == 1)
        .drop("_priority_rank")
    )
    logger.info("Deduplicated to %d unique variants.", df.count())
    return df


# ---------------------------------------------------------------------------
# Stage 5: Write
# ---------------------------------------------------------------------------
def write_output(
    df: DataFrame,
    output_dir: str,
    partition_by: str = "chrom",
    fmt: str = "parquet",
) -> None:
    """Write processed data partitioned by chromosome."""
    (
        df.write
        .mode("overwrite")
        .partitionBy(partition_by)
        .format(fmt)
        .save(output_dir)
    )
    logger.info("Written to %s (partitioned by %s).", output_dir, partition_by)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    raw_dir:    str = "data/raw",
    output_dir: str = "data/processed",
    master:     str = "local[*]",
) -> DataFrame:
    spark = create_spark_session(master=master)

    logger.info("=== Stage 1: Ingest ===")
    df = ingest(spark, raw_dir)

    logger.info("=== Stage 2: Normalize ===")
    df = normalize(df)

    logger.info("=== Stage 3: Enrich ===")
    df = enrich(df)

    logger.info("=== Stage 4: Deduplicate ===")
    df = deduplicate(df)

    logger.info("=== Stage 5: Write ===")
    write_output(df, output_dir)

    df.groupBy("source_db", "pathogenicity", "variant_type") \
      .count() \
      .orderBy("source_db", "count", ascending=False) \
      .show(40, truncate=False)

    return df


# ---------------------------------------------------------------------------
# Optional MongoDB sink
# ---------------------------------------------------------------------------
def write_to_mongodb(
    df: DataFrame,
    connection_uri: str,
    database:   str = "genomic_variants",
    collection: str = "variants",
) -> None:
    """
    Write final DataFrame to MongoDB.
    Requires: spark.jars.packages = org.mongodb.spark:mongo-spark-connector_2.12:10.3.0
    """
    (
        df.write
        .format("mongodb")
        .mode("overwrite")
        .option("spark.mongodb.write.connection.uri", connection_uri)
        .option("spark.mongodb.write.database",       database)
        .option("spark.mongodb.write.collection",     collection)
        .save()
    )
    logger.info("Written %d variants to MongoDB %s.%s", df.count(), database, collection)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Genomic Variant Spark ETL")
    parser.add_argument("--raw-dir",    default="data/raw",       help="Input directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--master",     default="local[*]",       help="Spark master URL")
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.output_dir, args.master)
