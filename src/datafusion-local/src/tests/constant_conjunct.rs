//! A pushed-down conjunct that references no column.
//!
//! Expression simplification turns `NOT (s = s)` into `s IS NULL AND NULL`: a
//! column conjunct and a literal `Boolean(NULL)` one. The literal reads no
//! column, and `build_row_filter` used to drop such a conjunct. Since DataFusion
//! removes the `FilterExec` when it pushes a predicate down, the scan is the only
//! place the predicate is applied, so dropping a conjunct *widens* the filter and
//! rows that cannot match come back.
//!
//! The visible symptom is a three-way partition that does not reconstruct the
//! scan: for a predicate `P`, `WHERE P`, `WHERE NOT P` and `WHERE P IS NULL` must
//! together return each row exactly once. Rows whose `P` is NULL were returned by
//! both `WHERE NOT P` and `WHERE P IS NULL`.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

/// The predicate under test. `s = s` is NULL wherever `s` is NULL, so a row with
/// a NULL `s`, an `f` above the threshold and an `id` outside the range makes the
/// whole predicate NULL.
const P: &str = "((t1.f <= 333.0 OR t1.s = t1.s) OR t1.id BETWEEN 1 AND 7)";

/// 12000 rows, more than the 8192-row default cache batch, so the scan spans more
/// than one cached batch. Every third row has a NULL `s`; `f` cycles 1..1000, so
/// two thirds of those NULL rows sit above the 333.0 threshold.
fn write_t1(path: &Path) {
    let rows = 12000i64;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("f", DataType::Float64, true),
        Field::new("s", DataType::Utf8, true),
    ]));
    let id: Int64Array = (1..=rows).collect::<Vec<_>>().into();
    let f: Float64Array = (1..=rows)
        .map(|i| Some((i % 1000) as f64))
        .collect::<Vec<_>>()
        .into();
    let s: StringArray = (1..=rows)
        .map(|i| (i % 3 != 0).then(|| format!("str{}", i % 17)))
        .collect::<Vec<_>>()
        .into();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(id), Arc::new(f), Arc::new(s)]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn liquid_ctx(dir: &Path) -> SessionContext {
    let parquet = dir.join("t1.parquet");
    write_t1(&parquet);
    let cache_dir = dir.join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();
    let (ctx, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir)
        .build(SessionConfig::new())
        .await
        .unwrap();
    ctx.register_parquet(
        "t1",
        parquet.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();
    ctx
}

/// The `s` column as a sorted multiset, NULL rendered as `<null>`. `s` is
/// projected as a string view, hence the cast.
async fn s_values(ctx: &SessionContext, sql: &str) -> Vec<String> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut out = Vec::new();
    for batch in batches {
        let column = arrow::compute::cast(batch.column(0), &DataType::Utf8).unwrap();
        let column = column.as_any().downcast_ref::<StringArray>().unwrap();
        for row in 0..column.len() {
            out.push(match column.is_null(row) {
                true => "<null>".to_string(),
                false => column.value(row).to_string(),
            });
        }
    }
    out.sort();
    out
}

/// `NOT (s = s)` is NULL where `s` is NULL and FALSE everywhere else, so it
/// matches no row. Simplification leaves it as `s IS NULL AND NULL`, and dropping
/// the literal conjunct returns every row with a NULL `s`.
#[tokio::test]
async fn constant_null_conjunct_is_still_applied() {
    let dir = TempDir::new().unwrap();
    let ctx = liquid_ctx(dir.path()).await;

    // Cold reads through the source and fills the cache; warm is served from it,
    // a separate evaluation path.
    for pass in ["cold", "warm"] {
        let rows = s_values(&ctx, "SELECT t1.s FROM t1 WHERE NOT (t1.s = t1.s)").await;
        assert_eq!(rows, Vec::<String>::new(), "{pass}");
    }
}

/// `WHERE P`, `WHERE NOT P` and `WHERE P IS NULL` partition the table: together
/// they must return exactly the rows of the unfiltered scan, each once.
#[tokio::test]
async fn three_way_partition_reconstructs_the_scan() {
    let dir = TempDir::new().unwrap();
    let ctx = liquid_ctx(dir.path()).await;

    for pass in ["cold", "warm"] {
        let unfiltered = s_values(&ctx, "SELECT t1.s FROM t1").await;

        let mut partitioned = s_values(&ctx, &format!("SELECT t1.s FROM t1 WHERE {P}")).await;
        partitioned.extend(s_values(&ctx, &format!("SELECT t1.s FROM t1 WHERE NOT {P}")).await);
        partitioned.extend(s_values(&ctx, &format!("SELECT t1.s FROM t1 WHERE {P} IS NULL")).await);
        partitioned.sort();

        assert_eq!(partitioned.len(), unfiltered.len(), "{pass}");
        assert_eq!(partitioned, unfiltered, "{pass}");
    }
}
