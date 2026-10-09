//! A scan that reads a virtual column.
//!
//! `file_row_index()` is answered by the parquet reader: projection pushdown
//! rewrites it into a row-number virtual column on the scan's `TableSchema`,
//! which `ParquetSource` derives from the parquet reader. The query must return
//! the same rows with and without the cache.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::util::pretty::pretty_format_batches;
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

use crate::LiquidCacheLocalBuilder;

/// 12000 rows, more than the 8192-row default batch, so row indexes span batches.
fn write_t1(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let id: Int64Array = (0..12000i64).map(|i| i * 10).collect::<Vec<_>>().into();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(id)]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

async fn register(ctx: &SessionContext, parquet: &Path) {
    ctx.register_parquet(
        "t1",
        parquet.to_str().unwrap(),
        ParquetReadOptions::default(),
    )
    .await
    .unwrap();
}

async fn run(ctx: &SessionContext, sql: &str) -> String {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

#[tokio::test]
async fn file_row_index_matches_datafusion() {
    let dir = TempDir::new().unwrap();
    let parquet = dir.path().join("t1.parquet");
    write_t1(&parquet);

    let plain = SessionContext::new();
    register(&plain, &parquet).await;

    let cache_dir = dir.path().join("cache");
    std::fs::create_dir_all(&cache_dir).unwrap();
    let (liquid, _cache) = LiquidCacheLocalBuilder::new()
        .with_cache_dir(cache_dir)
        .build(SessionConfig::new())
        .await
        .unwrap();
    register(&liquid, &parquet).await;

    let sql = "SELECT file_row_index() AS pos, id FROM t1 \
               WHERE id % 997 = 0 ORDER BY pos";
    let expected = run(&plain, sql).await;
    // Twice: the second run is served from the cache when the scan is cached.
    assert_eq!(run(&liquid, sql).await, expected);
    assert_eq!(run(&liquid, sql).await, expected);
}
