//! Optimizers for the Parquet module

mod lineage;

use std::{collections::HashSet, sync::Arc};

use datafusion::{
    catalog::memory::DataSourceExec,
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    datasource::{
        physical_plan::{FileSource, ParquetSource},
        source::DataSource,
        table_schema::TableSchema,
    },
    physical_expr::{PhysicalExpr, projection::ProjectionExprs, utils::collect_columns},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::ExecutionPlan,
};

pub(crate) use lineage::HintAnalyzer;
pub use lineage::LineageHints;

use crate::{LiquidCacheParquetRef, LiquidParquetSource, cache::ColumnLineages};

/// Physical optimizer rule for local mode liquid cache.
///
/// Rewrites `DataSourceExec` parquet scans to use [`LiquidParquetSource`], and
/// in the same pass derives typed lineage expressions from the full physical plan
/// (via the lineage analyzer) and attaches each scan's hints to its source.
#[derive(Debug)]
pub struct LocalModeOptimizer {
    cache: LiquidCacheParquetRef,
    prefetch: bool,
}

impl LocalModeOptimizer {
    /// Create an optimizer with an existing cache instance
    pub fn new(cache: LiquidCacheParquetRef) -> Self {
        Self {
            cache,
            prefetch: true,
        }
    }

    /// Create an optimizer with an existing cache instance
    pub fn with_cache(cache: LiquidCacheParquetRef) -> Self {
        Self::new(cache)
    }

    /// Enable or disable row-group prefetching.
    pub fn with_prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }
}

impl PhysicalOptimizerRule for LocalModeOptimizer {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>, datafusion::error::DataFusionError> {
        let analysis = HintAnalyzer::analyze(&plan);
        let cache = self.cache.clone();
        let prefetch = self.prefetch;
        let mut convert = |node: &Arc<dyn ExecutionPlan>, hints: ColumnLineages| {
            convert_parquet_scan(node, &cache, hints, prefetch)
        };
        Ok(lineage::rewrite_with_hints(plan, &mut convert, &analysis))
    }

    fn name(&self) -> &str {
        "LocalModeLiquidCacheOptimizer"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Rewrite the data source plan to use liquid cache, attaching `hints` (keyed by
/// file-schema column name) to every parquet scan it rewrites.
///
/// This is the entry point used by the cache server, where hints are derived on
/// the client (which has the full plan) and shipped alongside the pushed
/// fragment, which is always single-scan.
pub fn rewrite_data_source_plan_with_hints(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    hints: &ColumnLineages,
) -> Arc<dyn ExecutionPlan> {
    plan.transform_up(
        |node| match convert_parquet_scan(&node, cache, hints.clone(), true) {
            Some(new_node) => Ok(Transformed::new(
                new_node,
                true,
                TreeNodeRecursion::Continue,
            )),
            None => Ok(Transformed::no(node)),
        },
    )
    .unwrap()
    .data
}

/// Rewrite the data source plan to use liquid cache (no lineage expressions).
pub fn rewrite_data_source_plan(
    plan: Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
) -> Arc<dyn ExecutionPlan> {
    rewrite_data_source_plan_with_hints(plan, cache, &ColumnLineages::default())
}

/// The virtual columns this scan reads that the liquid path cannot produce.
///
/// The liquid read path has no notion of DataFusion's virtual columns. It carries
/// the [`TableSchema`] across faithfully but never produces one, so a scan that
/// reads a virtual column gets back a batch which simply lacks it, and a predicate
/// over one cannot be rewritten against the file schemas either. Such a scan has
/// to stay on `ParquetSource`, which derives virtual columns from the parquet
/// reader.
///
/// Declining a scan costs it the cache, so this stays narrow: a virtual column the
/// scan actually reads, not the mere presence of one on the table. A provider that
/// declares a row-position column on every table keeps the cache for the queries
/// that never project one. No projection at all is the one broad case, and it is
/// not a guess — the scan then reads the whole table schema, virtual columns
/// included.
///
/// Positional reads are what reaches here: applying positional deletes, and row
/// lineage, both project a reader-produced physical row position, which is an
/// absolute index into the file. A plausible but shifted position would associate
/// a delete with the wrong row, so declining is the sound answer while the liquid
/// reader cannot generate positions itself.
fn unproducible_virtual_columns(
    table_schema: &TableSchema,
    projection: Option<&ProjectionExprs>,
    filter: Option<&Arc<dyn PhysicalExpr>>,
) -> Option<String> {
    let virtual_columns = table_schema.virtual_columns();
    if virtual_columns.is_empty() {
        return None;
    }

    let mut needed: Vec<&str> = Vec::new();
    match projection {
        None => needed.extend(virtual_columns.iter().map(|field| field.name().as_str())),
        Some(projection) => {
            let mut read: HashSet<String> = projection
                .expr_iter()
                .flat_map(|expr| collect_columns(&expr))
                .map(|column| column.name().to_string())
                .collect();
            // The pushed-down predicate is rewritten against the file schemas in
            // the reader too, so a conjunct over a virtual column fails exactly as
            // a projection over one does. The row filter's own check cannot catch
            // it: that resolves against the table schema, which does hold the
            // virtual columns.
            if let Some(filter) = filter {
                read.extend(
                    collect_columns(filter)
                        .into_iter()
                        .map(|column| column.name().to_string()),
                );
            }
            needed.extend(
                virtual_columns
                    .iter()
                    .map(|field| field.name().as_str())
                    .filter(|name| read.contains(*name)),
            );
        }
    }

    if needed.is_empty() {
        return None;
    }
    Some(needed.join("`, `"))
}

/// If `node` is a `DataSourceExec` over a `ParquetSource`, return an equivalent
/// node backed by [`LiquidParquetSource`] carrying `hints`.
fn convert_parquet_scan(
    node: &Arc<dyn ExecutionPlan>,
    cache: &LiquidCacheParquetRef,
    hints: ColumnLineages,
    prefetch: bool,
) -> Option<Arc<dyn ExecutionPlan>> {
    let data_source_exec = node.downcast_ref::<DataSourceExec>()?;
    let (file_scan_config, parquet_source) =
        data_source_exec.downcast_to_file_source::<ParquetSource>()?;

    let pushed_filter = parquet_source.filter();
    if let Some(names) = unproducible_virtual_columns(
        parquet_source.table_schema(),
        parquet_source.projection(),
        pushed_filter.as_ref(),
    ) {
        // At info: this silently turns the cache off for a scan, and the only
        // symptom is that queries stop getting faster.
        log::info!(
            "liquid_cache scan BYPASS: the read path cannot produce virtual column(s) `{names}`"
        );
        return None;
    }

    let new_source =
        LiquidParquetSource::from_parquet_source(parquet_source.clone(), cache.clone())
            .with_lineages(Arc::new(hints))
            .with_prefetch(prefetch);

    let mut new_config = file_scan_config.clone();
    new_config.file_source = Arc::new(new_source);
    let new_file_source: Arc<dyn DataSource> = Arc::new(new_config);
    Some(Arc::new(DataSourceExec::new(new_file_source)))
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::Path};

    use arrow::{array::Int32Array, record_batch::RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        common::{ScalarValue, stats::Precision},
        datasource::physical_plan::{FileScanConfig, FileSource},
        logical_expr::Operator,
        physical_expr::expressions::{BinaryExpr, Column, Literal},
        physical_plan::{
            PhysicalExpr, collect, display::DisplayableExecutionPlan, filter_pushdown::PushedDown,
        },
        prelude::SessionContext,
    };
    use liquid_cache::{
        cache::{AlwaysHydrate, TranscodeEvict},
        cache_policies::LiquidPolicy,
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use crate::LiquidCacheParquet;

    use super::*;

    async fn make_cache(path: &Path) -> LiquidCacheParquetRef {
        let store = t4::mount(path.join("liquid_cache.t4")).await.unwrap();
        Arc::new(
            LiquidCacheParquet::new(
                8192,
                1000000,
                usize::MAX,
                store,
                Box::new(LiquidPolicy::new()),
                Box::new(TranscodeEvict),
                Box::new(AlwaysHydrate::new()),
            )
            .await,
        )
    }

    fn liquid_source(plan: &Arc<dyn ExecutionPlan>) -> LiquidParquetSource {
        let mut source = None;
        plan.apply(|node| {
            if let Some(plan) = node.downcast_ref::<DataSourceExec>() {
                let config = plan.data_source().downcast_ref::<FileScanConfig>().unwrap();
                source = Some(
                    config
                        .file_source()
                        .downcast_ref::<LiquidParquetSource>()
                        .unwrap()
                        .clone(),
                );
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        source.unwrap()
    }

    async fn rewrite_plan_inner(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let expected_schema = plan.schema();
        let tmp_dir = tempfile::tempdir().unwrap();
        let liquid_cache = make_cache(tmp_dir.path()).await;
        let rewritten = rewrite_data_source_plan(plan, &liquid_cache);

        rewritten
            .apply(|node| {
                if let Some(plan) = node.downcast_ref::<DataSourceExec>() {
                    let data_source = plan.data_source();
                    let source = data_source.downcast_ref::<FileScanConfig>().unwrap();
                    let file_source = source.file_source();
                    let _parquet_source =
                        file_source.downcast_ref::<LiquidParquetSource>().unwrap();
                    let schema = source.file_schema().as_ref();
                    assert_eq!(schema, expected_schema.as_ref());
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        rewritten
    }

    #[tokio::test]
    async fn test_plan_rewrite() {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "nano_hits",
            "../../examples/nano_hits.parquet",
            Default::default(),
        )
        .await
        .unwrap();
        let df = ctx
            .sql("SELECT * FROM nano_hits WHERE \"URL\" like 'https://%' limit 10")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let rewritten = rewrite_plan_inner(plan).await;

        let displayed = DisplayableExecutionPlan::new(rewritten.as_ref())
            .indent(true)
            .to_string();
        assert!(displayed.contains("predicate="), "{displayed}");

        rewritten
            .apply(|node| {
                if let Some(plan) = node.downcast_ref::<DataSourceExec>() {
                    let statistics = plan.data_source().partition_statistics(None)?;
                    assert!(!matches!(statistics.num_rows, Precision::Exact(_)));
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();

        // Supported filters are conjoined onto the predicate; unsupported ones
        // are handed back to the parent.
        let source = liquid_source(&rewritten);
        let url_index = source.table_schema().file_schema().index_of("URL").unwrap();
        let supported: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("URL", url_index)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some(
                "https://example.com".into(),
            )))),
        ));
        let unsupported: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("missing", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("value".into())))),
        ));
        let result = source
            .try_pushdown_filters(
                vec![supported, unsupported],
                &datafusion::config::ConfigOptions::new(),
            )
            .unwrap();
        assert!(matches!(
            result.filters.as_slice(),
            [PushedDown::Yes, PushedDown::No]
        ));
        let predicate = result.updated_node.unwrap().filter().unwrap().to_string();
        assert!(predicate.contains(" AND "), "{predicate}");
        assert!(predicate.contains("https://example.com"), "{predicate}");
        assert!(!predicate.contains("missing"), "{predicate}");
    }

    fn write_bloom_file(path: &Path) {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let properties = WriterProperties::builder()
            .set_bloom_filter_enabled(true)
            .build();
        let mut writer = ArrowWriter::try_new(
            File::create(path).unwrap(),
            schema.clone(),
            Some(properties),
        )
        .unwrap();
        for values in [[1, 2, 4], [1, 3, 4]] {
            writer
                .write(
                    &RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(values.to_vec()))],
                    )
                    .unwrap(),
                )
                .unwrap();
            writer.flush().unwrap();
        }
        writer.close().unwrap();
    }

    #[tokio::test]
    async fn prunes_row_group_with_bloom_filter() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let parquet_path = tmp_dir.path().join("bloom.parquet");
        write_bloom_file(&parquet_path);

        let ctx = SessionContext::new();
        ctx.register_parquet("t", parquet_path.to_str().unwrap(), Default::default())
            .await
            .unwrap();
        let plan = ctx
            .sql("SELECT * FROM t WHERE a = 2")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let cache = make_cache(tmp_dir.path()).await;
        let rewritten = rewrite_data_source_plan(plan, &cache);
        let metrics = liquid_source(&rewritten).metrics().clone();

        let batches = collect(rewritten, ctx.task_ctx()).await.unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);

        let metric = metrics
            .clone_inner()
            .sum_by_name("row_groups_pruned_bloom_filter")
            .unwrap();
        let datafusion::physical_plan::metrics::MetricValue::PruningMetrics {
            pruning_metrics, ..
        } = metric
        else {
            panic!("unexpected metric: {metric:?}");
        };
        assert_eq!(pruning_metrics.pruned(), 1);
    }

    /// Declining a scan costs it the cache, so the guard must key on what the scan
    /// reads, not on what the table declares. A row-position column present on the
    /// table but absent from the projection keeps the cache; no projection at all
    /// reads the whole table schema and does not.
    #[test]
    fn only_a_virtual_column_the_scan_reads_costs_the_cache() {
        use arrow_schema::Fields;
        use datafusion::physical_expr::expressions::{BinaryExpr, col, lit};
        use datafusion::physical_expr::projection::ProjectionExpr;

        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Int64, true),
        ]));
        let row_pos = Field::new("row_number", DataType::Int64, false)
            .with_extension_type(parquet::arrow::RowNumber);

        // No virtual columns at all: nothing to refuse, whatever the projection.
        let plain = TableSchema::builder(Arc::clone(&file_schema)).build();
        assert_eq!(unproducible_virtual_columns(&plain, None, None), None);

        let positional = TableSchema::builder(Arc::clone(&file_schema))
            .with_virtual_columns(Fields::from(vec![row_pos.clone()]))
            .build();
        let full = positional.table_schema();

        // Declared but not read: still cached.
        let only_id =
            ProjectionExprs::new(vec![ProjectionExpr::new(col("id", full).unwrap(), "id")]);
        assert_eq!(
            unproducible_virtual_columns(&positional, Some(&only_id), None),
            None
        );

        // Read: refused, and named.
        let reads_pos = ProjectionExprs::new(vec![ProjectionExpr::new(
            col("row_number", full).unwrap(),
            "row_number",
        )]);
        assert_eq!(
            unproducible_virtual_columns(&positional, Some(&reads_pos), None).as_deref(),
            Some("row_number")
        );

        // No projection reads the whole table schema, virtual columns included.
        assert_eq!(
            unproducible_virtual_columns(&positional, None, None).as_deref(),
            Some("row_number")
        );

        // Read only by the pushed-down predicate: refused too. The reader rewrites
        // the predicate against the file schemas as well, and the row filter's own
        // check passes it because that resolves against the table schema.
        let pos_predicate: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            col("row_number", full).unwrap(),
            Operator::Gt,
            lit(0i64),
        ));
        assert_eq!(
            unproducible_virtual_columns(&positional, Some(&only_id), Some(&pos_predicate))
                .as_deref(),
            Some("row_number")
        );
    }

    /// The guard at its call site, with an ordinary scan as a positive control, so
    /// deleting it from `convert_parquet_scan` fails a test instead of silently
    /// restoring a plan that cannot execute.
    #[tokio::test]
    async fn a_scan_reading_a_virtual_column_stays_on_parquet_source() {
        use arrow_schema::Fields;
        use datafusion::datasource::listing::PartitionedFile;
        use datafusion::datasource::physical_plan::FileScanConfigBuilder;
        use datafusion::execution::object_store::ObjectStoreUrl;

        let file_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let scan = |table_schema: TableSchema| -> Arc<dyn ExecutionPlan> {
            let source = Arc::new(ParquetSource::new(table_schema)) as Arc<dyn FileSource>;
            let config = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
                .with_file(PartitionedFile::new("t.parquet", 16))
                .build();
            Arc::new(DataSourceExec::new(Arc::new(config)))
        };

        let tmp_dir = tempfile::tempdir().unwrap();
        let cache = make_cache(tmp_dir.path()).await;

        // Positive control: an ordinary scan is still handed to the cache.
        let plain = scan(TableSchema::builder(Arc::clone(&file_schema)).build());
        assert!(
            convert_parquet_scan(&plain, &cache, ColumnLineages::default(), true).is_some(),
            "an ordinary scan must still convert to the liquid source"
        );

        // `ParquetSource::new` projects the whole table schema, so the positional
        // scan reads the row-position column and cannot be served.
        let positional = scan(
            TableSchema::builder(file_schema)
                .with_virtual_columns(Fields::from(vec![
                    Field::new("row_number", DataType::Int64, false)
                        .with_extension_type(parquet::arrow::RowNumber),
                ]))
                .build(),
        );
        assert!(
            convert_parquet_scan(&positional, &cache, ColumnLineages::default(), true).is_none(),
            "a scan reading a virtual column must stay on ParquetSource"
        );
    }
}
