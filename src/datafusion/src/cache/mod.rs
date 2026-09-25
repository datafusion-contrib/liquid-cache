//! This module contains the cache implementation for the Parquet reader.
//!

use crate::io::ParquetCacheMetadata;
use crate::reader::{LiquidPredicate, extract_multi_column_or};
use crate::sync::{Mutex, RwLock};
use ahash::AHashMap;
use arrow::array::{BooleanArray, RecordBatch, RecordBatchOptions};
use arrow::buffer::BooleanBuffer;
use arrow_schema::{ArrowError, Field, Schema, SchemaRef};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use liquid_cache::cache::EvictionPolicy;
use liquid_cache::cache::{
    CacheEntry, CacheExpression, CachePolicy, EntryID, EventTrace, HydrationPolicy, LiquidCache,
    LiquidCacheBuilder,
};
use parquet::arrow::arrow_reader::ArrowPredicate;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod column;
mod id;
mod stats;

pub use column::{CachedColumn, CachedColumnRef};
pub(crate) use column::{InsertArrowArrayError, PrefetchOutcome};
pub(crate) use id::ColumnAccessPath;
pub use id::{BatchID, ParquetArrayID};

/// Typed lineage expressions for a single file, keyed by file-schema column name.
///
/// Produced by the physical lineage analyzer (local mode) or shipped from
/// the client (Flight mode), and attached to the
/// [`LiquidParquetSource`](crate::LiquidParquetSource) that opens the file.
pub type ColumnLineages = HashMap<String, Arc<CacheExpression>>;

/// The identity of a Parquet object within an object store.
///
/// Object paths are only unique within their object store, so both components
/// are required to keep cached data from different stores isolated.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ParquetFileIdentity {
    object_store_url: ObjectStoreUrl,
    path: String,
}

impl ParquetFileIdentity {
    /// Create an identity from an object store URL and an object path.
    pub fn new(object_store_url: ObjectStoreUrl, path: String) -> Self {
        Self {
            object_store_url,
            path,
        }
    }
}

/// One column of a row group: (file column index, field, lineage expression, is-predicate).
type CachedColumnSpec = (u64, Arc<Field>, Option<Arc<CacheExpression>>, bool);

#[derive(Default, Debug)]
pub(crate) struct RowGroupSnapshots {
    entries: RwLock<AHashMap<EntryID, Arc<CacheEntry>>>,
    selections: RwLock<AHashMap<BatchID, BooleanBuffer>>,
}

impl RowGroupSnapshots {
    pub(crate) fn get(&self, entry_id: &EntryID) -> Option<Arc<CacheEntry>> {
        self.entries.read().unwrap().get(entry_id).cloned()
    }

    pub(crate) fn insert(&self, entry_id: EntryID, entry: Arc<CacheEntry>) {
        self.entries.write().unwrap().insert(entry_id, entry);
    }

    pub(crate) fn selection(&self, batch_id: BatchID) -> Option<BooleanBuffer> {
        self.selections.read().unwrap().get(&batch_id).cloned()
    }

    pub(crate) fn insert_selection(&self, batch_id: BatchID, selection: BooleanBuffer) {
        self.selections.write().unwrap().insert(batch_id, selection);
    }
}

#[derive(Default, Debug)]
struct ColumnMaps {
    // invariant: Arc::ptr_eq(map[field.name()], map[field.id()])
    by_id: AHashMap<u64, CachedColumnRef>,
    by_name: AHashMap<String, CachedColumnRef>,
}

/// A row group in the cache.
#[derive(Debug)]
pub struct CachedRowGroup {
    columns: ColumnMaps,
    cache_store: Arc<LiquidCache>,
    snapshots: Arc<RowGroupSnapshots>,
}

impl CachedRowGroup {
    /// Create a new row group.
    /// The column_ids are the indices of the columns in the file schema.
    /// So they may not start from 0.
    fn new(
        cache_store: Arc<LiquidCache>,
        row_group_idx: u64,
        file_idx: u64,
        columns: &[CachedColumnSpec],
        snapshots: Arc<RowGroupSnapshots>,
    ) -> Self {
        let mut column_maps = ColumnMaps::default();
        for (column_id, field, expression, is_predicate_column) in columns {
            let column_access_path = ColumnAccessPath::new(file_idx, row_group_idx, *column_id);
            let column = Arc::new(CachedColumn::new(
                Arc::clone(field),
                Arc::clone(&cache_store),
                column_access_path,
                expression.clone(),
                *is_predicate_column,
                Arc::clone(&snapshots),
            ));
            column_maps.by_id.insert(*column_id, column.clone());
            column_maps.by_name.insert(field.name().to_string(), column);
        }

        Self {
            columns: column_maps,
            cache_store,
            snapshots,
        }
    }

    /// Returns the batch size configured for this cached row group.
    pub fn batch_size(&self) -> usize {
        self.cache_store.config().batch_size()
    }

    /// Get a column from the row group.
    pub fn get_column(&self, column_id: u64) -> Option<CachedColumnRef> {
        self.columns.by_id.get(&column_id).cloned()
    }

    /// Get a column from the row group by its field name.
    pub fn get_column_by_name(&self, column_name: &str) -> Option<CachedColumnRef> {
        if let Some(column) = self.columns.by_name.get(column_name) {
            return Some(column.clone());
        }

        // DataFusion may carry qualified names in physical expressions
        // (e.g. "table.col"), while cache fields are keyed by file schema names.
        let unqualified = column_name.rsplit('.').next().unwrap_or(column_name);
        self.columns.by_name.get(unqualified).cloned()
    }

    pub(crate) fn snapshot_selection(&self, batch_id: BatchID) -> Option<BooleanBuffer> {
        self.snapshots.selection(batch_id)
    }

    /// Evaluate a predicate on a row group.
    #[fastrace::trace]
    pub async fn evaluate_selection_with_predicate(
        &self,
        batch_id: BatchID,
        selection: &BooleanBuffer,
        predicate: &mut LiquidPredicate,
    ) -> Option<Result<BooleanArray, ArrowError>> {
        let column_ids = predicate.predicate_column_ids();

        if column_ids.len() == 1 {
            // If we only have one column, we can short-circuit and try to evaluate the predicate on encoded data.
            let column_id = column_ids[0];
            let cache = self.get_column(column_id as u64)?;
            return cache
                .eval_predicate_with_filter(batch_id, selection, predicate)
                .await;
        } else if column_ids.len() >= 2 {
            // Try to extract multiple column-literal expressions from OR structure
            if let Some(column_exprs) =
                extract_multi_column_or(predicate.physical_expr_physical_column_index())
            {
                let mut combined_buffer: Option<BooleanArray> = None;

                for (col_name, expr) in column_exprs {
                    let column = self.get_column_by_name(col_name)?;
                    let snapshot_liquid = match column.snapshot_entry(batch_id) {
                        Some(entry) => match entry.as_ref() {
                            CacheEntry::MemoryLiquid(array) => Some(Arc::clone(array)),
                            _ => {
                                combined_buffer = None;
                                break;
                            }
                        },
                        None => None,
                    };
                    let expr = expr
                        .transform_up(|expr| {
                            if let Some(column) = expr.downcast_ref::<Column>() {
                                Ok(Transformed::yes(Arc::new(Column::new(column.name(), 0))
                                    as Arc<dyn PhysicalExpr>))
                            } else {
                                Ok(Transformed::no(expr))
                            }
                        })
                        .ok()?
                        .data;
                    let liquid_expr = column.liquid_expr_for_predicate(expr);
                    let liquid_expr = match liquid_expr {
                        Some(expr) => expr,
                        None => {
                            combined_buffer = None;
                            break;
                        }
                    };
                    let entry_id = column.entry_id(batch_id).into();
                    let liquid_array = match snapshot_liquid {
                        Some(array) => Some(array),
                        None => self.cache_store.try_read_liquid(&entry_id).await,
                    };
                    let liquid_array = match liquid_array {
                        None => {
                            combined_buffer = None;
                            break;
                        }
                        Some(array) => array,
                    };
                    let buffer = liquid_array.try_eval_predicate(&liquid_expr, selection);

                    combined_buffer = Some(match combined_buffer {
                        None => buffer,
                        Some(existing) => {
                            arrow::compute::kernels::boolean::or_kleene(&existing, &buffer).ok()?
                        }
                    });
                }

                if let Some(result) = combined_buffer {
                    return Some(Ok(result));
                }
            }
        }
        // A conjunct that reads no column still has to be evaluated, over a batch
        // that carries only the row count the selection implies.
        if column_ids.is_empty() {
            let options =
                RecordBatchOptions::new().with_row_count(Some(selection.count_set_bits()));
            let record_batch =
                RecordBatch::try_new_with_options(Arc::new(Schema::empty()), Vec::new(), &options)
                    .ok()?;
            return Some(predicate.evaluate(record_batch));
        }

        // Otherwise, we need to first convert the data into arrow arrays.
        let mut arrays = Vec::new();
        let mut fields = Vec::new();
        for column_id in column_ids {
            let column = self.get_column(column_id as u64)?;
            let array = column
                .get_arrow_array_with_filter(batch_id, selection)
                .await?;
            arrays.push(array);
            fields.push(column.field());
        }
        let schema = Arc::new(Schema::new(fields));
        let record_batch = RecordBatch::try_new(schema, arrays).unwrap();
        let boolean_array = predicate.evaluate(record_batch).unwrap();
        Some(Ok(boolean_array))
    }
}

pub(crate) type CachedRowGroupRef = Arc<CachedRowGroup>;

/// A file in the cache.
#[derive(Debug)]
pub struct CachedFile {
    cache_store: Arc<LiquidCache>,
    file_id: u64,
    file_schema: SchemaRef,
    lineages: Arc<ColumnLineages>,
}

impl CachedFile {
    fn new(
        cache_store: Arc<LiquidCache>,
        file_id: u64,
        file_schema: SchemaRef,
        lineages: Arc<ColumnLineages>,
    ) -> Self {
        Self {
            cache_store,
            file_id,
            file_schema,
            lineages,
        }
    }

    /// Create a row group handle scoped to the current query context.
    pub fn create_row_group(
        &self,
        row_group_id: u64,
        predicate_column_ids: Vec<usize>,
    ) -> CachedRowGroupRef {
        self.create_row_group_with_snapshots(row_group_id, predicate_column_ids, Arc::default())
    }

    pub(crate) fn create_row_group_with_snapshots(
        &self,
        row_group_id: u64,
        predicate_column_ids: Vec<usize>,
        snapshots: Arc<RowGroupSnapshots>,
    ) -> CachedRowGroupRef {
        let columns: Vec<CachedColumnSpec> = self
            .file_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(idx, field)| {
                let is_predicate_column = predicate_column_ids.contains(&idx);
                let expression = self.lineages.get(field.name()).cloned();
                (
                    idx as u64,
                    Arc::clone(field),
                    expression,
                    is_predicate_column,
                )
            })
            .collect();

        Arc::new(CachedRowGroup::new(
            self.cache_store.clone(),
            row_group_id,
            self.file_id,
            &columns,
            snapshots,
        ))
    }

    /// Return the configured cache batch size.
    pub fn batch_size(&self) -> usize {
        self.cache_store.config().batch_size()
    }

    /// Return the full file schema tracked by the cache entry.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.file_schema)
    }
}

/// A reference to a cached file.
pub(crate) type CachedFileRef = Arc<CachedFile>;

/// The main cache structure.
#[derive(Debug)]
pub struct LiquidCacheParquet {
    /// Map object-store-qualified file identity to file id.
    files: Mutex<AHashMap<ParquetFileIdentity, u64>>,

    cache_store: Arc<LiquidCache>,

    current_file_id: AtomicU64,
}

/// A reference to the main cache structure.
pub type LiquidCacheParquetRef = Arc<LiquidCacheParquet>;

impl LiquidCacheParquet {
    /// Create a new cache for parquet files.
    pub async fn new(
        batch_size: usize,
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        store: t4::Store,
        cache_policy: Box<dyn CachePolicy>,
        eviction_policy: Box<dyn EvictionPolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
    ) -> Self {
        Self::new_with_eviction_concurrency(
            batch_size,
            max_memory_bytes,
            max_disk_bytes,
            store,
            cache_policy,
            eviction_policy,
            hydration_policy,
            !cfg!(test),
        )
        .await
    }

    /// Create a new cache for parquet files with explicit victim eviction concurrency.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_eviction_concurrency(
        batch_size: usize,
        max_memory_bytes: usize,
        max_disk_bytes: usize,
        store: t4::Store,
        cache_policy: Box<dyn CachePolicy>,
        eviction_policy: Box<dyn EvictionPolicy>,
        hydration_policy: Box<dyn HydrationPolicy>,
        evict_victims_concurrently: bool,
    ) -> Self {
        assert!(batch_size.is_power_of_two());
        let metadata = Arc::new(ParquetCacheMetadata::new());
        let cache_storage = LiquidCacheBuilder::new()
            .with_batch_size(batch_size)
            .with_max_memory_bytes(max_memory_bytes)
            .with_max_disk_bytes(max_disk_bytes)
            .with_eviction_policy(eviction_policy)
            .with_cache_policy(cache_policy)
            .with_hydration_policy(hydration_policy)
            .with_metadata(metadata)
            .with_store(store)
            .with_evict_victims_concurrently(evict_victims_concurrently)
            .build()
            .await;

        LiquidCacheParquet {
            files: Mutex::new(AHashMap::new()),
            cache_store: cache_storage,
            current_file_id: AtomicU64::new(0),
        }
    }

    /// Register a file in the cache.
    pub fn register_or_get_file(
        &self,
        file_identity: ParquetFileIdentity,
        full_file_schema: SchemaRef,
    ) -> CachedFileRef {
        self.register_or_get_file_with_hints(file_identity, full_file_schema, Arc::default())
    }

    /// Register a file in the cache, attaching typed lineage expressions derived from
    /// the query plan (keyed by file-schema column name).
    pub fn register_or_get_file_with_hints(
        &self,
        file_identity: ParquetFileIdentity,
        full_file_schema: SchemaRef,
        lineages: Arc<ColumnLineages>,
    ) -> CachedFileRef {
        let mut files = self.files.lock().unwrap();
        let file_id = *files
            .entry(file_identity)
            .or_insert_with(|| self.current_file_id.fetch_add(1, Ordering::Relaxed));
        drop(files);

        Arc::new(CachedFile::new(
            self.cache_store.clone(),
            file_id,
            full_file_schema,
            lineages,
        ))
    }

    /// Get the batch size of the cache.
    pub fn batch_size(&self) -> usize {
        self.cache_store.config().batch_size()
    }

    /// Get the max memory bytes of the cache.
    pub fn max_memory_bytes(&self) -> usize {
        self.cache_store.config().max_memory_bytes()
    }

    /// Get the max disk bytes of the cache.
    pub fn max_disk_bytes(&self) -> usize {
        self.cache_store.config().max_disk_bytes()
    }

    /// Get the memory usage of the cache in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        self.cache_store.budget().memory_usage_bytes()
    }

    /// Get the disk usage of the cache in bytes.
    pub fn disk_usage_bytes(&self) -> usize {
        self.cache_store.budget().disk_usage_bytes()
    }

    /// Flush the cache trace to a file.
    pub fn flush_trace(&self, to_file: impl AsRef<Path>) {
        self.cache_store.observer().flush_cache_trace(to_file);
    }

    /// Enable the cache trace.
    pub fn enable_trace(&self) {
        self.cache_store.observer().enable_cache_trace();
    }

    /// Disable the cache trace.
    pub fn disable_trace(&self) {
        self.cache_store.observer().disable_cache_trace();
    }

    /// Reset the cache.
    ///
    /// # Safety
    /// This is unsafe because resetting the cache while other threads are using the cache may cause undefined behavior.
    /// You should only call this when no one else is using the cache.
    pub unsafe fn reset(&self) {
        let mut files = self.files.lock().unwrap();
        files.clear();
        self.cache_store.reset();
    }

    /// Flush all memory-based entries to disk while preserving their format.
    /// Arrow entries become DiskArrow, Liquid entries become DiskLiquid.
    /// Entries already on disk are left unchanged.
    ///
    /// This is for admin use only.
    /// This has no guarantees that some new entry will not be inserted in the meantime, or some entries are promoted to memory again.
    /// You mostly want to use this when no one else is using the cache.
    pub async fn flush_data(&self) -> Result<(), liquid_cache::cache::CacheFull> {
        self.cache_store.flush_all_to_disk().await
    }

    /// Get the storage of the cache.
    pub fn storage(&self) -> &Arc<LiquidCache> {
        &self.cache_store
    }

    /// Consume the event trace of the cache.
    pub fn consume_event_trace(&self) -> EventTrace {
        self.cache_store.consume_event_trace()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{CachedRowGroupRef, LiquidCacheParquet};
    use crate::reader::FilterCandidateBuilder;
    use arrow::array::{Array, ArrayRef, Int32Array, StringViewArray};
    use arrow::buffer::BooleanBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Literal};
    use datafusion::physical_plan::expressions::Column;
    use liquid_cache::cache::AlwaysHydrate;
    use liquid_cache::cache::TranscodeEvict;
    use liquid_cache::cache_policies::LiquidPolicy;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use std::sync::Arc;

    async fn setup_cache(batch_size: usize, schema: SchemaRef) -> CachedRowGroupRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = LiquidCacheParquet::new(
            batch_size,
            usize::MAX,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeEvict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;
        let file = cache.register_or_get_file(
            ParquetFileIdentity::new(ObjectStoreUrl::local_filesystem(), "test".to_string()),
            schema,
        );
        file.create_row_group(0, vec![])
    }

    async fn setup_liquid_cache(
        batch_size: usize,
        schema: SchemaRef,
        max_memory_bytes: usize,
    ) -> CachedRowGroupRef {
        let tmp_dir = tempfile::tempdir().unwrap();
        let store = t4::mount(tmp_dir.path().join("liquid_cache.t4"))
            .await
            .unwrap();
        let cache = LiquidCacheParquet::new(
            batch_size,
            max_memory_bytes,
            usize::MAX,
            store,
            Box::new(LiquidPolicy::new()),
            Box::new(TranscodeEvict),
            Box::new(AlwaysHydrate::new()),
        )
        .await;
        cache
            .register_or_get_file(
                ParquetFileIdentity::new(ObjectStoreUrl::local_filesystem(), "test".to_string()),
                schema,
            )
            .create_row_group(0, vec![0, 1])
    }

    fn build_predicate(
        schema: &SchemaRef,
        arrays: Vec<ArrayRef>,
        expr: Arc<dyn PhysicalExpr>,
    ) -> LiquidPredicate {
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(schema), None).unwrap();
        writer
            .write(&RecordBatch::try_new(Arc::clone(schema), arrays).unwrap())
            .unwrap();
        writer.close().unwrap();
        let reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&reader, ArrowReaderOptions::new()).unwrap();
        let candidate = FilterCandidateBuilder::new(expr, Arc::clone(schema))
            .build(metadata.metadata())
            .unwrap()
            .unwrap();
        let projection = candidate.projection(metadata.metadata());
        LiquidPredicate::try_new(candidate, projection).unwrap()
    }

    fn equals(name: &str, index: usize, value: ScalarValue) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            Arc::new(Column::new(name, index)),
            Operator::Eq,
            Arc::new(Literal::new(value)),
        ))
    }

    fn dummy(len: usize) -> ArrayRef {
        Arc::new(Int32Array::from(vec![0; len]))
    }

    async fn insert_liquid_columns(
        row_group: &CachedRowGroupRef,
        batch_id: BatchID,
        arrays: [ArrayRef; 3],
    ) {
        for (column_id, array) in arrays.into_iter().enumerate() {
            row_group
                .get_column(column_id as u64)
                .unwrap()
                .insert(batch_id, array)
                .await
                .unwrap();
        }
        assert_eq!(row_group.cache_store.stats().memory_liquid_entries, 2);
    }

    #[tokio::test]
    async fn or_fast_path_evaluates_on_liquid() {
        let batch_size = 1024;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("dummy", DataType::Int32, false),
        ]));
        let a: ArrayRef = Arc::new(Int32Array::from_iter_values(
            (0..batch_size).map(|i| (i % 4) as i32 + 1),
        ));
        let b: ArrayRef = Arc::new(Int32Array::from_iter_values(
            (0..batch_size).map(|i| (i % 5) as i32 * 10),
        ));
        let budget = a.get_array_memory_size() + b.get_array_memory_size();
        let row_group = setup_liquid_cache(batch_size, schema.clone(), budget).await;
        let batch_id = BatchID::from_row_id(0, batch_size);
        insert_liquid_columns(&row_group, batch_id, [a.clone(), b.clone(), dummy(1)]).await;
        let expr = Arc::new(BinaryExpr::new(
            equals("a", 0, ScalarValue::Int32(Some(3))),
            Operator::Or,
            equals("b", 1, ScalarValue::Int32(Some(20))),
        ));
        let mut predicate = build_predicate(&schema, vec![a, b, dummy(batch_size)], expr);
        row_group.cache_store.stats();
        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();
        let expected = BooleanBuffer::collect_bool(batch_size, |i| i % 4 == 2 || i % 5 == 2);
        assert_eq!(result, BooleanArray::new(expected, None));
        assert!(row_group.cache_store.stats().runtime.try_read_liquid_calls >= 2);
    }

    #[tokio::test]
    async fn or_fast_path_on_strings() {
        let batch_size = 128;
        let schema = Arc::new(Schema::new(vec![
            Field::new("city", DataType::Utf8View, false),
            Field::new("name", DataType::Utf8View, false),
            Field::new("dummy", DataType::Int32, false),
        ]));
        let city: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..batch_size).map(|i| if i % 5 == 2 { "Tokyo" } else { "Paris" }),
        ));
        let name: ArrayRef = Arc::new(StringViewArray::from_iter_values(
            (0..batch_size).map(|i| if i % 4 == 1 { "Bob" } else { "Alice" }),
        ));
        let budget = city.get_array_memory_size() + name.get_array_memory_size();
        let row_group = setup_liquid_cache(batch_size, schema.clone(), budget).await;
        let batch_id = BatchID::from_row_id(0, batch_size);
        insert_liquid_columns(&row_group, batch_id, [city.clone(), name.clone(), dummy(1)]).await;
        let expr = Arc::new(BinaryExpr::new(
            equals("name", 1, ScalarValue::Utf8View(Some("Bob".into()))),
            Operator::Or,
            equals("city", 0, ScalarValue::Utf8View(Some("Tokyo".into()))),
        ));
        let mut predicate = build_predicate(&schema, vec![city, name, dummy(batch_size)], expr);
        row_group.cache_store.stats();
        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();
        let expected = BooleanBuffer::collect_bool(batch_size, |i| i % 4 == 1 || i % 5 == 2);
        assert_eq!(result, BooleanArray::new(expected, None));
        assert!(row_group.cache_store.stats().runtime.try_read_liquid_calls >= 2);
    }

    #[tokio::test]
    async fn evaluate_or_on_cached_columns() {
        let batch_size = 4;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let row_group = setup_cache(batch_size, schema.clone()).await;

        let col_a = row_group.get_column(0).unwrap();
        let col_b = row_group.get_column(1).unwrap();

        let batch_id = BatchID::from_row_id(0, batch_size);

        let array_a = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let array_b = Arc::new(Int32Array::from(vec![10, 20, 30, 40]));

        assert!(col_a.insert(batch_id, array_a.clone()).await.is_ok());
        assert!(col_b.insert(batch_id, array_b.clone()).await.is_ok());

        // build parquet metadata for predicate construction
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array_a, array_b]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        // expression a = 3 OR b = 20
        let expr_a: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));
        let expr_b: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(20)))),
        ));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(expr_a, Operator::Or, expr_b));

        let builder = FilterCandidateBuilder::new(expr, Arc::clone(&schema));
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();

        let expected = BooleanBuffer::collect_bool(batch_size, |i| i == 1 || i == 2).into();
        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn evaluate_three_column_or() {
        let batch_size = 8;

        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
        ]));

        let row_group = setup_cache(batch_size, schema.clone()).await;

        let col_a = row_group.get_column(0).unwrap();
        let col_b = row_group.get_column(1).unwrap();
        let col_c = row_group.get_column(2).unwrap();

        let batch_id = BatchID::from_row_id(0, batch_size);

        let array_a = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
        let array_b = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80]));
        let array_c = Arc::new(Int32Array::from(vec![
            100, 200, 300, 400, 500, 600, 700, 800,
        ]));

        assert!(col_a.insert(batch_id, array_a.clone()).await.is_ok());
        assert!(col_b.insert(batch_id, array_b.clone()).await.is_ok());
        assert!(col_c.insert(batch_id, array_c.clone()).await.is_ok());

        // build parquet metadata for predicate construction
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![array_a, array_b, array_c]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        // expression: a = 2 OR b = 40 OR c = 600
        let expr_a: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("a", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))),
        ));
        let expr_b: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("b", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(40)))),
        ));
        let expr_c: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("c", 2)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(600)))),
        ));

        // Build nested OR: (a = 2 OR b = 40) OR c = 600
        let expr_ab = Arc::new(BinaryExpr::new(expr_a, Operator::Or, expr_b));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(expr_ab, Operator::Or, expr_c));

        let builder = FilterCandidateBuilder::new(expr, Arc::clone(&schema));
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();

        // Expected: row 1 (a=2), row 3 (b=40), row 5 (c=600) -> indices 1, 3, 5
        let expected =
            BooleanBuffer::collect_bool(batch_size, |i| i == 1 || i == 3 || i == 5).into();
        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn evaluate_string_column_or() {
        let batch_size = 8;

        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8View, false),
            Field::new("city", DataType::Utf8View, false),
        ]));

        let row_group = setup_cache(batch_size, schema.clone()).await;

        let col_name = row_group.get_column(0).unwrap();
        let col_city = row_group.get_column(1).unwrap();

        let batch_id = BatchID::from_row_id(0, batch_size);

        let array_name = Arc::new(arrow::array::StringViewArray::from(vec![
            "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry",
        ]));
        let array_city = Arc::new(arrow::array::StringViewArray::from(vec![
            "New York", "London", "Paris", "Tokyo", "Berlin", "Sydney", "Madrid", "Rome",
        ]));

        assert!(col_name.insert(batch_id, array_name.clone()).await.is_ok());
        assert!(col_city.insert(batch_id, array_city.clone()).await.is_ok());

        // build parquet metadata for predicate construction
        let tmp_meta = tempfile::NamedTempFile::new().unwrap();
        let mut writer =
            ArrowWriter::try_new(tmp_meta.reopen().unwrap(), Arc::clone(&schema), None).unwrap();
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![array_name, array_city]).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let file_reader = std::fs::File::open(tmp_meta.path()).unwrap();
        let metadata = ArrowReaderMetadata::load(&file_reader, ArrowReaderOptions::new()).unwrap();

        // expression: name = "Bob" OR city = "Tokyo"
        let expr_name: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("name", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8View(Some("Bob".to_string())))),
        ));
        let expr_city: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("city", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8View(Some(
                "Tokyo".to_string(),
            )))),
        ));
        let expr: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(expr_name, Operator::Or, expr_city));

        let builder = FilterCandidateBuilder::new(expr, Arc::clone(&schema));
        let candidate = builder.build(metadata.metadata()).unwrap().unwrap();
        let projection = candidate.projection(metadata.metadata());
        let mut predicate = LiquidPredicate::try_new(candidate, projection).unwrap();

        let selection = BooleanBuffer::new_set(batch_size);
        let result = row_group
            .evaluate_selection_with_predicate(batch_id, &selection, &mut predicate)
            .await
            .unwrap()
            .unwrap();

        // Expected: row 1 (name="Bob"), row 3 (city="Tokyo") -> indices 1, 3
        let expected = BooleanBuffer::collect_bool(batch_size, |i| i == 1 || i == 3).into();
        assert_eq!(result, expected);
    }
}
