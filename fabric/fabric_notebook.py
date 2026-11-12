# %% [markdown]
# # ShipSense - Medallion pipeline in Microsoft Fabric (PySpark)
# Same logic as `src/lakehouse.py`, but on Spark + OneLake.
# **Before running:** attach your Lakehouse to this notebook and upload `erp_shipments.csv` to `Files/raw/`.
# `spark` is pre-created in Fabric notebooks. Tables written with `saveAsTable` appear under **Tables**.

# %%
from pyspark.sql import functions as F

# Make failed date parses return NULL instead of raising (we try 3 formats on purpose).
spark.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")
spark.conf.set("spark.sql.ansi.enabled", "false")

DELAY_THRESHOLD_HOURS = 48  # rule from shipment_status_codes.md

# %% [markdown]
# ## BRONZE - raw copy, every column as text, plus lineage columns

# %%
bronze = (
    spark.read.option("header", True).csv("Files/raw/erp_shipments.csv")  # no inferSchema = all strings
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source", F.lit("erp_shipments.csv"))
)
bronze.write.mode("overwrite").format("delta").saveAsTable("bronze_erp_shipments")
print("bronze rows:", spark.table("bronze_erp_shipments").count())  # expect 42

# %% [markdown]
# ## SILVER - one rule per data-quality problem

# %%
def parse_date(col):
    # rule 3: try the 3 formats found in the data; first one that works wins (DD/MM, never MM/DD)
    return F.coalesce(
        F.to_date(F.col(col), "yyyy-MM-dd"),
        F.to_date(F.col(col), "dd-MMM-yyyy"),
        F.to_date(F.col(col), "dd/MM/yyyy"),
    )

silver = (
    spark.table("bronze_erp_shipments")
    .dropDuplicates(["shipment_id"])                                       # rule 1: duplicates
    .withColumn("container_no", F.upper(F.trim("container_no")))           # rule 2: lowercase containers
    .withColumn("etd", parse_date("etd"))
    .withColumn("original_eta", parse_date("original_eta"))
    .withColumn("current_eta", parse_date("current_eta"))
    .withColumn("teu", F.coalesce(F.col("teu").cast("int"), F.lit(1)))    # rule 4: empty teu -> 1
    .withColumn("freight_usd", F.col("freight_usd").cast("double"))
    .withColumn("delay_hours", F.datediff("current_eta", "original_eta") * 24)
    .withColumn("is_delayed", F.col("delay_hours") > DELAY_THRESHOLD_HOURS)
    .drop("_ingested_at", "_source")
)
silver.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable("silver_shipments")
print("silver rows:", spark.table("silver_shipments").count())  # expect 40
display(spark.table("silver_shipments").limit(5))

# %% [markdown]
# ## GOLD - business KPIs with Spark SQL

# %%
spark.sql("""
    CREATE OR REPLACE TABLE gold_kpi_by_carrier AS
    SELECT carrier,
           COUNT(*)                                                      AS shipments,
           CAST(SUM(teu) AS INT)                                         AS teu,
           ROUND(100.0 * AVG(CASE WHEN is_delayed THEN 0 ELSE 1 END), 1) AS on_time_pct,
           ROUND(AVG(delay_hours), 1)                                    AS avg_delay_hours
    FROM silver_shipments GROUP BY carrier
""")
spark.sql("""
    CREATE OR REPLACE TABLE gold_kpi_by_port AS
    SELECT pod AS port,
           COUNT(*)                                  AS shipments,
           CAST(SUM(CAST(is_delayed AS INT)) AS INT) AS delayed,
           CAST(SUM(freight_usd) AS BIGINT)          AS freight_usd
    FROM silver_shipments GROUP BY pod
""")
display(spark.sql("SELECT * FROM gold_kpi_by_carrier ORDER BY on_time_pct"))
# expect (ERP data only, no live portal ETAs): Oceanic 53.8, Meridian 76.9, BlueWave 78.6

# %% [markdown]
# ## SQL - the same question a manager or Power BI would ask

# %%
display(spark.sql("""
    SELECT shipment_id, container_no, customer, carrier, pod, delay_hours
    FROM silver_shipments
    WHERE is_delayed
    ORDER BY delay_hours DESC
"""))

# %% [markdown]
# ## OPTIONAL - call Azure OpenAI from the notebook (key from Azure Key Vault, never hard-coded)
# Needs: a Key Vault with secret `aoai-key`, and your Fabric user given "Key Vault Secrets User" on it.
# Fill in the 3 values below and set RUN_AI = True.

# %%
RUN_AI = False
if RUN_AI:
    from openai import OpenAI
    VAULT_URL = "https://<your-vault>.vault.azure.net/"
    AOAI_ENDPOINT = "https://<your-resource>.openai.azure.com"
    DEPLOYMENT = "gpt-4.1-mini"

    key = notebookutils.credentials.getSecret(VAULT_URL, "aoai-key")  # Fabric built-in helper
    client = OpenAI(base_url=f"{AOAI_ENDPOINT}/openai/v1/", api_key=key)
    kpis = spark.table("gold_kpi_by_carrier").toPandas().to_string(index=False)
    reply = client.chat.completions.create(
        model=DEPLOYMENT, temperature=0,
        messages=[{"role": "user", "content": f"In 3 bullet points, summarise this carrier performance:\n{kpis}"}],
    )
    print(reply.choices[0].message.content)
