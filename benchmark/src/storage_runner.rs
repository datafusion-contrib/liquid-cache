#![cfg(target_os = "linux")]

/**
 * Benchmark to test the performance of io_uring runtime for clickbench queries. The queries are executed directly
 * on a LiquidCache instance to bypass datafusion, which is strongly coupled with tokio. The benchmark is based on
 * the arrow benchmark (https://github.com/apache/arrow-rs/blob/main/parquet/benches/arrow_reader_clickbench.rs#L729)
 */
use arrow::array::BooleanArray;
use arrow::buffer::BooleanBuffer;
use clap::Parser;
use datafusion::logical_expr::Operator;
use futures::future::join_all;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::expressions::{BinaryExpr, Column};
use datafusion::scalar::ScalarValue;
use liquid_cache_common::IoMode;
use liquid_cache_parquet::{SimpleIoContext, WorkStealingUringRuntime};
use liquid_cache_storage::cache::{
    CacheExpression, EntryID, LiquidCache, LiquidCacheBuilder, LiquidPolicy, NoHydration,
    TranscodeSqueezeEvict,
};
use logforth::filter::EnvFilter;
use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
use std::fs::create_dir_all;
use std::path::PathBuf;
use std::future::Future;
use std::sync::Arc;
use std::time::Instant;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

#[derive(Parser)]
#[command(name = "storage_runner")]
struct Args {
    /// ClickBench query index (0-based). Only queries with filters are supported (e.g. 1, 10, 19, 20).
    #[arg(long)]
    query_index: usize,

    /// Number of partitions (tasks per iteration).
    #[arg(long)]
    partitions: usize,

    /// Worker threads: io_uring work-stealing runtime size, or Tokio worker threads when `--io-mode std-blocking`.
    #[arg(long)]
    worker_threads: usize,

    #[arg(long)]
    iterations: usize,

    /// Path to hits.parquet. Default: benchmark/clickbench/data/hits.parquet
    #[arg(long, default_value = "benchmark/clickbench/data/hits.parquet")]
    parquet: PathBuf,

    /// Directory for the liquid-cache storage. Default: $TMPDIR/liquid_cache_storage_runner
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Directory to write flamegraph SVG files to (one per query iteration).
    #[arg(long = "flamegraph-dir")]
    flamegraph_dir: Option<PathBuf>,

    /// IO mode: uring-non-blocking (default) or std-blocking. With std-blocking, partition futures run on a multi-thread Tokio runtime (`worker_threads` workers).
    #[arg(long = "io-mode", default_value = "uring-non-blocking")]
    io_mode: IoMode,
}

/// Tracks process disk I/O (bytes read/written) between creation and stop().
struct DiskIoGuard {
    system: System,
    pid: sysinfo::Pid,
    start_read_total: u64,
    start_written_total: u64,
}

impl DiskIoGuard {
    fn new() -> Self {
        let mut system = System::new();
        let pid = sysinfo::get_current_pid().unwrap();
        system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[pid]),
            true,
            ProcessRefreshKind::nothing().with_disk_usage(),
        );
        let p = system.process(pid).unwrap();
        let du = p.disk_usage();
        Self {
            system,
            pid,
            start_read_total: du.total_read_bytes,
            start_written_total: du.total_written_bytes,
        }
    }

    fn stop(mut self) -> (u64, u64) {
        self.system.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]),
            true,
            ProcessRefreshKind::nothing().with_disk_usage(),
        );
        if let Some(p) = self.system.process(self.pid) {
            let du = p.disk_usage();
            (
                du.total_read_bytes.saturating_sub(self.start_read_total),
                du.total_written_bytes
                    .saturating_sub(self.start_written_total),
            )
        } else {
            (0, 0)
        }
    }
}

/// ClickBench query descriptor: filter column(s), optional projection column(s), and predicate expression(s).
/// Each predicate is evaluated on a single column (column index 0 in the cached array).
#[derive(Clone)]
struct FilterQuery {
    /// Column names to load and cache for filtering (in schema order).
    filter_columns: Vec<&'static str>,
    /// Column names to load when there are no predicates (projection-only / full-scan queries).
    projection_columns: Vec<&'static str>,
    /// One predicate per filter column; each expects Column(0) op Literal. Empty for projection-only.
    predicates: Vec<Arc<dyn PhysicalExpr>>,
    /// Number of expected rows in result
    expected_row_count: usize,
}

impl FilterQuery {
    /// Columns to load into cache: filter_columns when present, else projection_columns.
    fn columns_to_load(&self) -> &[&'static str] {
        if self.filter_columns.is_empty() {
            &self.projection_columns
        } else {
            &self.filter_columns
        }
    }
}

fn all_filter_queries() -> Vec<Option<FilterQuery>> {
    use datafusion::physical_plan::expressions::Literal as Lit;
    let col = || Arc::new(Column::new("col", 0)) as Arc<dyn PhysicalExpr>;

    let mut q: Vec<Option<FilterQuery>> = (0..43).map(|_| None).collect();

    // Q1: AdvEngineID <> 0
    q[1] = Some(FilterQuery {
        filter_columns: vec!["AdvEngineID"],
        projection_columns: vec![],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::UInt64(Some(0)))),
        ))],
        expected_row_count: 3312,
    });

    // Q10: MobilePhoneModel <> ''
    q[10] = Some(FilterQuery {
        filter_columns: vec!["MobilePhoneModel"],
        projection_columns: vec![],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 34276,
    });

    // Q12: SearchPhrase <> ''
    q[12] = Some(FilterQuery {
        filter_columns: vec!["SearchPhrase"],
        projection_columns: vec![],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 131559,
    });

    // Q19: UserID = 3233473875476175636 (value that exists in hits_1)
    q[19] = Some(FilterQuery {
        filter_columns: vec!["UserID"],
        projection_columns: vec![],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::Eq,
            Arc::new(Lit::new(ScalarValue::UInt64(Some(3233473875476175636)))),
        ))],
        expected_row_count: 4,
    });

    q[20] = Some(FilterQuery {
        filter_columns: vec!["URL"],
        projection_columns: vec![],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::LikeMatch,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::from("%google%")))))
        ))],
        expected_row_count: 99997497,
    });

    // Q27: URL <> ''
    q[27] = Some(FilterQuery {
        filter_columns: vec!["URL"],
        projection_columns: vec![],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 999978,
    });

    // Q28: Referer <> ''
    q[28] = Some(FilterQuery {
        filter_columns: vec!["Referer"],
        projection_columns: vec![],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 925813,
    });

    // Q30: SearchPhrase <> ''
    q[30] = Some(FilterQuery {
        filter_columns: vec!["SearchPhrase"],
        projection_columns: vec![],
        predicates: vec![Arc::new(BinaryExpr::new(
            col(),
            Operator::NotEq,
            Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
        ))],
        expected_row_count: 131559,
    });

    // Q36: CounterID = 62, DontCountHits = 0, IsRefresh = 0, URL <> ''
    q[36] = Some(FilterQuery {
        filter_columns: vec![
            "CounterID",
            "EventDate",
            "DontCountHits",
            "IsRefresh",
            "URL",
        ],
        projection_columns: vec![],
        predicates: vec![
            Arc::new(BinaryExpr::new(
                col(),
                Operator::Eq,
                Arc::new(Lit::new(ScalarValue::UInt32(Some(62)))),
            )),
            Arc::new(BinaryExpr::new(
                col(),
                Operator::Eq,
                Arc::new(Lit::new(ScalarValue::Int16(Some(0)))),
            )),
            Arc::new(BinaryExpr::new(
                col(),
                Operator::Eq,
                Arc::new(Lit::new(ScalarValue::UInt8(Some(0)))),
            )),
            Arc::new(BinaryExpr::new(
                col(),
                Operator::Eq,
                Arc::new(Lit::new(ScalarValue::UInt8(Some(0)))),
            )),
            Arc::new(BinaryExpr::new(
                col(),
                Operator::NotEq,
                Arc::new(Lit::new(ScalarValue::Utf8(Some(String::new())))),
            )),
        ],
        expected_row_count: 181198,
    });

    q
}

/// Partition futures run either on the work-stealing io_uring executor or, for `std-blocking` IO, on Tokio.
enum StorageBenchRuntime {
    Uring(WorkStealingUringRuntime),
    Tokio(tokio::runtime::Runtime),
}

impl StorageBenchRuntime {
    fn new(io_mode: IoMode, num_workers: usize) -> Self {
        let num_workers = num_workers.max(1);
        if io_mode == IoMode::StdBlocking {
            Self::Tokio(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(num_workers)
                    .thread_name("storage-bench-tokio")
                    .enable_all()
                    .build()
                    .expect("build tokio multi-thread runtime"),
            )
        } else {
            Self::Uring(WorkStealingUringRuntime::new(num_workers))
        }
    }

    fn run_to_completion<F, T>(&self, future: F) -> T
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        match self {
            Self::Uring(e) => e.run_to_completion(future),
            Self::Tokio(rt) => rt.block_on(future),
        }
    }

    fn total_runnable_wall_nanos(&self) -> Option<u64> {
        match self {
            Self::Uring(e) => Some(e.total_runnable_wall_nanos()),
            Self::Tokio(_) => None,
        }
    }
}

fn run_single_iter(
    num_batches: usize,
    num_partitions: usize,
    query: &FilterQuery,
    storage: Arc<LiquidCache>,
    entry_ids: &Vec<EntryID>,
    batch_lengths: &Vec<usize>,
    runtime: &StorageBenchRuntime,
) -> (std::time::Duration, usize) {
    // 2) Partition batch indices evenly across workers.
    let batches_per_partition = num_batches / num_partitions;
    let num_cols = query.columns_to_load().len();

    // 3) Create futures for every partition (only for partitions that have at least one batch)
    let mut futures = Vec::new();
    let mut start_batch_idx = 0;
    for p in 0..num_partitions {
        let batch_count = if p < num_batches % num_partitions {
            batches_per_partition + 1
        } else {
            batches_per_partition
        };
        let end = (start_batch_idx + batch_count).min(num_batches);
        if start_batch_idx >= end {
            continue;
        }
        let storage_clone = Arc::clone(&storage);
        let batch_range = start_batch_idx..end;
        start_batch_idx = end;
        let predicates = query.predicates.iter().map(Arc::clone).collect::<Vec<_>>();
        let entry_ids_clone = entry_ids.clone();
        let batch_lengths_clone = batch_lengths.clone();
        futures.push(run_partition(
            storage_clone,
            batch_range,
            num_cols,
            predicates,
            entry_ids_clone,
            batch_lengths_clone,
        ));
    }
    let num_tasks = futures.len();

    let start = Instant::now();
    let total_rows = match runtime {
        StorageBenchRuntime::Uring(executor) => {
            let receiver = executor.spawn_many(&mut futures);
            let mut tasks_completed = 0;
            let mut total_rows = 0;
            while tasks_completed < num_tasks {
                total_rows += receiver.recv().expect("Failed to receive result");
                tasks_completed += 1;
            }
            total_rows
        }
        StorageBenchRuntime::Tokio(rt) => {
            let handles: Vec<_> = futures.into_iter().map(|f| rt.spawn(f)).collect();
            rt.block_on(join_all(handles))
                .into_iter()
                .map(|r| r.expect("partition task failed"))
                .sum::<usize>()
        }
    };
    let elapsed = start.elapsed();
    if total_rows != query.expected_row_count {
        log::warn!(
            "Expected row count doesn't match. Actual: {}, expected: {}",
            total_rows,
            query.expected_row_count
        );
    }
    log::info!(
        "Partitions: {}, wall: {:.3}s, total rows: {}",
        num_partitions,
        elapsed.as_secs_f64(),
        total_rows
    );
    (elapsed, total_rows)
}

fn write_flamegraph(
    profiler: &pprof::ProfilerGuard<'_>,
    flamegraph_dir: &std::path::Path,
    query_index: usize,
    iteration: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let report = profiler.report().build()?;
    let mut svg_data = Vec::new();
    report.flamegraph(&mut svg_data)?;
    create_dir_all(flamegraph_dir)?;

    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
    let secs = now.as_secs();
    let hour = (secs / 3600) % 24;
    let minute = (secs / 60) % 60;
    let second = secs % 60;

    let filename =
        format!("{hour:02}h{minute:02}m{second:02}s_q{query_index:02}_i{iteration:02}.svg");
    let filepath = flamegraph_dir.join(filename);
    std::fs::write(&filepath, svg_data)?;
    log::info!("Flamegraph written to: {}", filepath.display());
    Ok(())
}

fn run_bench(
    cache_dir: PathBuf,
    parquet_path: PathBuf,
    query: &FilterQuery,
    query_index: usize,
    num_partitions: usize,
    num_iter: usize,
    num_workers: usize,
    flamegraph_dir: Option<PathBuf>,
    io_mode: IoMode,
) {
    let _ = std::fs::create_dir_all(&cache_dir);
    let fb_pool_size = if io_mode == IoMode::UringNonBlocking {
        4096
    } else {
        0
    };
    let io_context = Arc::new(SimpleIoContext::new(
        cache_dir.clone(),
        io_mode,
        fb_pool_size,
    ));
    let storage = LiquidCacheBuilder::new()
        .with_io_context(io_context)
        .with_cache_dir(cache_dir)
        .with_max_cache_bytes(256 * 1024 * 1024)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .with_hydration_policy(Box::new(NoHydration::new()))
        .with_squeeze_policy(Box::new(TranscodeSqueezeEvict))
        .build();

    let runtime = StorageBenchRuntime::new(io_mode, num_workers);
    let storage_clone = storage.clone();
    let query_owned = query.clone();
    let (num_batches, entry_ids, batch_lengths) = runtime.run_to_completion(async move {
        // 1) Load parquet into record batches (filter columns only) and insert into cache.
        let (entry_ids, batch_lengths) =
            load_and_insert(storage_clone.clone(), parquet_path, &query_owned).await;
        let num_cols_loaded = query_owned.columns_to_load().len();
        let num_batches = entry_ids.len() / num_cols_loaded;
        log::info!(
            "Populated cache: {} batches, {} columns, {} entries",
            num_batches,
            num_cols_loaded,
            entry_ids.len()
        );

        storage_clone.flush_all_to_disk().await;
        (num_batches, entry_ids, batch_lengths)
    });

    // Baseline after cache load so iteration deltas exclude setup work on the same workers.
    let mut prev_runnable_wall_total_ns = runtime.total_runnable_wall_nanos().unwrap_or(0);

    for i in 0..num_iter {
        liquid_cache_benchmarks::tracepoints::iteration_start(query_index as u32, i as u32);
        let io_guard = DiskIoGuard::new();
        let profiler_guard = if flamegraph_dir.is_some() {
            Some(
                pprof::ProfilerGuardBuilder::default()
                    .frequency(500)
                    .blocklist(&["libpthread.so.0", "libm.so.6", "libgcc_s.so.1"])
                    .build()
                    .expect("pprof ProfilerGuardBuilder::build"),
            )
        } else {
            None
        };

        let (iter_wall, _rows) = run_single_iter(
            num_batches,
            num_partitions,
            &query,
            storage.clone(),
            &entry_ids,
            &batch_lengths,
            &runtime,
        );

        let uring_runnable = match runtime.total_runnable_wall_nanos() {
            Some(total_now) => {
                let runnable_this_iter_ns =
                    total_now.saturating_sub(prev_runnable_wall_total_ns);
                prev_runnable_wall_total_ns = total_now;
                let wall_ns = iter_wall.as_nanos() as u64;
                // Summed across workers; can exceed wall clock when workers run in parallel.
                let wall_minus_runnable_sum_ns = wall_ns.saturating_sub(runnable_this_iter_ns);
                Some((
                    runnable_this_iter_ns as f64 / 1e6,
                    wall_minus_runnable_sum_ns as f64 / 1e6,
                ))
            }
            None => None,
        };

        let (disk_read, disk_written) = io_guard.stop();
        log::info!(
            "{}",
            liquid_cache_benchmarks::format_storage_iteration_metrics(
                i,
                iter_wall,
                disk_read,
                disk_written,
                uring_runnable,
            )
        );

        if let (Some(profiler), Some(dir)) = (profiler_guard, flamegraph_dir.as_ref()) {
            if let Err(e) = write_flamegraph(&profiler, dir, query_index, i as u32) {
                log::warn!("Failed to write flamegraph for iteration {}: {}", i, e);
            }
        }
    }
}

async fn run_partition(
    storage: Arc<LiquidCache>,
    batch_range: std::ops::Range<usize>,
    num_cols: usize,
    predicates: Vec<Arc<dyn PhysicalExpr>>,
    entry_ids: Vec<EntryID>,
    batch_lengths: Vec<usize>,
) -> usize {
    let mut total_matched = 0usize;

    if predicates.is_empty() {
        // No predicates: full scan, count all rows in the partition.
        for batch_idx in batch_range.clone() {
            let entry_idx = batch_idx * num_cols;
            let entry_id = &entry_ids[entry_idx];
            let _result = storage.get(entry_id).await;
            total_matched += batch_lengths[entry_idx];
        }
        return total_matched;
    }

    for batch_idx in batch_range {
        let mut combined_mask: Option<BooleanArray> = None;
        for (col_idx, pred) in predicates.iter().enumerate() {
            let entry_idx = batch_idx * num_cols + col_idx;
            let entry_id = &entry_ids[entry_idx];
            let len = batch_lengths[entry_idx];
            let selection = BooleanBuffer::new_set(len);
            let result = storage
                .eval_predicate(entry_id, pred)
                .with_selection(&selection) // Is this necessary?
                .await;
            match result {
                Some(Ok(mask)) => {
                    combined_mask = Some(match combined_mask.take() {
                        Some(prev) => arrow::compute::and(&prev, &mask).unwrap(),
                        None => mask,
                    });
                }
                Some(Err(_)) | None => {
                    // Predicate could not be evaluated in cache; treat as no match for this batch.
                    combined_mask = Some(BooleanArray::from(vec![false; len]));
                }
            }
        }
        if let Some(m) = combined_mask {
            total_matched += m.true_count();
        }
    }
    total_matched
}

/// Load parquet with projection = query.columns_to_load(), insert each (batch, column) into cache.
/// Returns (entry_ids in order batch0_col0, batch0_col1, ..., batch1_col0, ...), (length per entry).
async fn load_and_insert(
    storage: Arc<liquid_cache_storage::cache::LiquidCache>,
    parquet_path: PathBuf,
    query: &FilterQuery,
) -> (Vec<EntryID>, Vec<usize>) {
    let columns_to_load = query.columns_to_load();
    let needs_substring_fingerprints = query.predicates.iter().any(|predicate| {
        predicate
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .is_some_and(|expr| {
                matches!(
                    expr.op(),
                    Operator::LikeMatch
                        | Operator::NotLikeMatch
                )
            })
    });
    assert!(
        !columns_to_load.is_empty(),
        "query must have filter_columns or projection_columns"
    );

    let Ok(parquet_file) = std::fs::File::open(parquet_path.clone()) else {
        panic!("Failed to open {:?}", parquet_path.to_str());
    };

    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file).unwrap();
    let schema = builder.parquet_schema();
    let root_fields = schema.root_schema().get_fields();
    let projection_root_indices: Vec<usize> = columns_to_load
        .iter()
        .map(|name| {
            root_fields
                .iter()
                .position(|f| f.name() == *name)
                .unwrap_or_else(|| panic!("parquet schema has no column '{name}'"))
        })
        .collect();
    let projection_mask = ProjectionMask::roots(schema, projection_root_indices);

    let mut reader = builder
        .with_batch_size(8192)
        .with_projection(projection_mask)
        .build()
        .unwrap();

    let num_cols = columns_to_load.len();
    let mut entry_ids = Vec::new();
    let mut batch_lengths = Vec::new();
    let mut batch_idx = 0usize;

    while let Some(batch_res) = reader.next() {
        let batch = batch_res.expect("parquet read batch");
        let nrows = batch.num_rows();
        for col_idx in 0..num_cols {
            let entry_id = EntryID::from(batch_idx * num_cols + col_idx);
            let array = batch.column(col_idx).clone();
            if needs_substring_fingerprints {
                storage
                    .insert(entry_id, array)
                    .with_squeeze_hint(Arc::new(CacheExpression::SubstringSearch))
                    .await;
            } else {
                storage.insert(entry_id, array).await;
            }
            entry_ids.push(entry_id);
            batch_lengths.push(nrows);
        }
        batch_idx += 1;
    }

    (entry_ids, batch_lengths)
}

fn setup_logging() {
    let mut builder = logforth::builder();
    builder = builder.dispatch(|d| {
        d.filter(EnvFilter::from_default_env())
            .append(logforth::append::Stdout::default())
    });
    builder.apply();
}

fn main() {
    let args = Args::parse();
    setup_logging();

    let queries = all_filter_queries();
    let query = match args.query_index {
        i if i < queries.len() => match &queries[i] {
            Some(q) => q,
            None => {
                eprintln!(
                    "Query index {} has no filters. Only filter queries are supported. \
                     Try e.g. 1, 10, 12, 19, 27, 28, 30, 36.",
                    args.query_index
                );
                std::process::exit(1);
            }
        },
        _ => {
            eprintln!(
                "Query index {} out of range (0..{}).",
                args.query_index,
                queries.len()
            );
            std::process::exit(1);
        }
    };

    if args.partitions == 0 {
        eprintln!("partitions must be >= 1.");
        std::process::exit(1);
    }

    if !args.parquet.exists() {
        eprintln!(
            "Parquet file not found: {}. Download e.g. wget https://datasets.clickhouse.com/hits_compatible/athena/hits.parquet -O {}",
            args.parquet.display(),
            args.parquet.display()
        );
        std::process::exit(1);
    }
    let cache_dir = args
        .cache_dir
        .unwrap_or_else(|| std::env::temp_dir().join("lc_cache_dir"));
    run_bench(
        cache_dir,
        args.parquet,
        query,
        args.query_index,
        args.partitions,
        args.iterations,
        args.worker_threads,
        args.flamegraph_dir,
        args.io_mode,
    );
}
