use arrow_schema::{DataType, Field, Schema};
use liquid_cache::{
    cache::{CacheStats, Evict, EvictionPolicy, TranscodeEvict},
    cache_policies::LiquidPolicy,
};
use liquid_cache_datafusion::LiquidCacheParquetRef;
use std::{fmt, path::Path, sync::Arc};
use tempfile::TempDir;

use arrow::util::pretty::pretty_format_batches;
use datafusion::{
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTableUrl},
    },
    error::Result,
    physical_plan::{ExecutionPlan, collect, display::DisplayableExecutionPlan},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};

use crate::LiquidCacheLocalBuilder;
mod constant_conjunct;
mod date_optimizer;
mod filter_limit;
mod nested_filter;
mod variants;

const TEST_FILE: &str = "../../examples/nano_hits.parquet";
const OPENOBSERVE_FILE: &str = "../../dev/test_parquet/openobserve.parquet";

#[derive(Debug, Clone)]
struct QueryOutcome {
    values: String,
    plan: String,
    stats: CacheStatsSummary,
}

#[derive(Debug, Clone)]
struct CacheStatsSummary {
    stats: CacheStats,
    entries_after_first_run: usize,
}

impl CacheStatsSummary {
    fn from_stats(stats: CacheStats, entries_after_first_run: usize) -> Self {
        Self {
            stats,
            entries_after_first_run,
        }
    }

    fn has_cache_hits(&self) -> bool {
        let runtime = &self.stats.runtime;
        runtime.get_with_selection > 0
            || runtime.try_read_liquid_calls > 0
            || runtime.get > 0
            || runtime.eval_predicate > 0
    }

    fn entries_reused(&self) -> bool {
        self.stats.total_entries == self.entries_after_first_run
    }
}

impl fmt::Display for CacheStatsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "entries.total: {}", self.stats.total_entries)?;
        writeln!(
            f,
            "entries.after_first_run: {}",
            self.entries_after_first_run
        )?;
        writeln!(
            f,
            "entries.memory.arrow: {}",
            self.stats.memory_arrow_entries
        )?;
        writeln!(
            f,
            "entries.memory.liquid: {}",
            self.stats.memory_liquid_entries
        )?;
        writeln!(f, "entries.disk.liquid: {}", self.stats.disk_liquid_entries)?;
        writeln!(f, "entries.disk.arrow: {}", self.stats.disk_arrow_entries)?;
        writeln!(f, "usage.memory_bytes: {}", self.stats.memory_usage_bytes)?;
        writeln!(f, "usage.disk_bytes: {}", self.stats.disk_usage_bytes)?;
        // Use the Display implementation for runtime stats
        write!(f, "{}", self.stats.runtime)
    }
}

async fn create_session_context_with_liquid_cache(
    eviction_policy: Box<dyn EvictionPolicy>,
    cache_size_bytes: usize,
    cache_dir: &Path,
) -> Result<(SessionContext, LiquidCacheParquetRef)> {
    // These tests snapshot exact cache contents and counters. A repartitioned
    // file scan populates the cache concurrently, so insertion order (and, for
    // LIMIT queries, which partitions finish before cancellation) is not a
    // stable property to snapshot.
    let mut config = SessionConfig::new().with_repartition_file_scans(false);
    config.options_mut().execution.target_partitions = 4;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_prefetch(false)
        .with_max_memory_bytes(cache_size_bytes)
        .with_cache_dir(cache_dir.to_path_buf())
        .with_eviction_policy(eviction_policy)
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(config)
        .await?;

    // Register the test parquet file
    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();

    Ok((ctx, cache))
}

async fn get_physical_plan(sql: &str, ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    let df = ctx.sql(sql).await.unwrap();
    let (state, plan) = df.into_parts();
    state.create_physical_plan(&plan).await.unwrap()
}

async fn get_result(ctx: &SessionContext, sql: &str) -> String {
    let plan = get_physical_plan(sql, ctx).await;
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

async fn run_io_profile(prefetch: bool, cache_dir: &Path) -> (String, u64, u64, u64) {
    let config = SessionConfig::new().with_repartition_file_scans(false);
    let builder = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(64 * 1024 * 1024)
        .with_cache_dir(cache_dir.to_path_buf());
    let builder = if prefetch {
        builder
    } else {
        builder.with_prefetch(false)
    };
    let (ctx, cache) = builder.build(config).await.unwrap();
    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();
    let sql = r#"SELECT "WatchID" FROM hits WHERE "SearchPhrase" LIKE '%abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%'"#;

    let first = get_result(&ctx, sql).await;
    cache.flush_data().await.unwrap();
    cache.storage().stats();
    let second = get_result(&ctx, sql).await;
    let runtime = cache.storage().stats().runtime;
    assert_eq!(first, second);

    (
        second,
        runtime.read_io_count,
        runtime.get,
        runtime.eval_predicate,
    )
}

#[tokio::test]
async fn prefetch_matches_lazy_io() {
    let lazy_dir = TempDir::new().unwrap();
    let prefetch_dir = TempDir::new().unwrap();

    let lazy = run_io_profile(false, lazy_dir.path()).await;
    let prefetch = run_io_profile(true, prefetch_dir.path()).await;

    assert_eq!(lazy, prefetch);
}

async fn run_sql_with_cache(
    sql: &str,
    eviction_policy: Box<dyn EvictionPolicy>,
    cache_size_bytes: usize,
    cache_dir: &Path,
) -> QueryOutcome {
    let (ctx, cache) =
        create_session_context_with_liquid_cache(eviction_policy, cache_size_bytes, cache_dir)
            .await
            .unwrap();

    let plan = get_physical_plan(sql, &ctx).await;
    let displayable = DisplayableExecutionPlan::new(plan.as_ref());
    let plan_string = format!("{}", displayable.tree_render());

    // Clear any historical runtime counters before prefetching the cache.
    cache.storage().stats();

    let first_run = get_result(&ctx, sql).await;
    let entries_after_first_run = cache.storage().stats().total_entries;
    let second_run = get_result(&ctx, sql).await;

    assert_eq!(first_run, second_run);

    let stats_after_second_run = cache.storage().stats();
    let stats = CacheStatsSummary::from_stats(stats_after_second_run, entries_after_first_run);

    QueryOutcome {
        values: second_run,
        plan: plan_string,
        stats,
    }
}

async fn test_runner(sql: &str, reference: &str, cache_dir: &Path) {
    let cache_sizes = [10 * 1024, 1024 * 1024, usize::MAX]; // 10KB, 1MB, unlimited

    for cache_size in cache_sizes {
        let eviction_policies: Vec<Box<dyn EvictionPolicy>> =
            vec![Box::new(TranscodeEvict), Box::new(Evict)];
        for eviction_policy in eviction_policies {
            let QueryOutcome { values, .. } =
                run_sql_with_cache(sql, eviction_policy, cache_size, cache_dir).await;
            assert_eq!(
                values, reference,
                "Results differ, cache_size: {cache_size}"
            );
        }
    }
}

#[tokio::test]
async fn test_url_prefix_filtering() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select COUNT(*) from hits where "URL" like 'https://%'"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(sql, Box::new(TranscodeEvict), 1024 * 1024, cache_dir.path()).await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::with_settings!({ filters => vec![(r"usage\.(memory|disk)_bytes: \d+", "usage.${1}_bytes: [bytes]")] }, {
        insta::assert_snapshot!(format!(
            "plan: \n{}\nvalues: \n{}\nstats:\n{}",
            plan, values, stats
        ));
    });
    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_url_selection_and_ordering() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "URL" from hits where "URL" like '%tours%' order by "URL" desc"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(sql, Box::new(TranscodeEvict), 1024 * 300, cache_dir.path()).await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::with_settings!({ filters => vec![(r"usage\.(memory|disk)_bytes: \d+", "usage.${1}_bytes: [bytes]")] }, {
        insta::assert_snapshot!(format!(
            "plan: \n{}\nvalues: \n{}\nstats:\n{}",
            plan, values, stats
        ));
    });
    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_os_selection() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "OS" from hits where "URL" like '%tours%' order by "OS" desc"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(sql, Box::new(TranscodeEvict), 1024 * 1024, cache_dir.path()).await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::with_settings!({ filters => vec![(r"usage\.(memory|disk)_bytes: \d+", "usage.${1}_bytes: [bytes]")] }, {
        insta::assert_snapshot!(format!(
            "plan: \n{}\nvalues: \n{}\nstats:\n{}",
            plan, values, stats
        ));
    });

    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_referer_filtering() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "Referer" from hits where "Referer" <> '' AND "URL" like '%tours%' order by "Referer" desc"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(sql, Box::new(TranscodeEvict), 1024 * 1024, cache_dir.path()).await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::with_settings!({ filters => vec![(r"usage\.(memory|disk)_bytes: \d+", "usage.${1}_bytes: [bytes]")] }, {
        insta::assert_snapshot!(format!(
            "plan: \n{}\nvalues: \n{}\nstats:\n{}",
            plan, values, stats
        ));
    });

    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_single_column_filter_projection() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "WatchID" from hits where "WatchID" = 6978470580070504163"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(sql, Box::new(TranscodeEvict), 1024 * 1024, cache_dir.path()).await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::with_settings!({ filters => vec![(r"usage\.(memory|disk)_bytes: \d+", "usage.${1}_bytes: [bytes]")] }, {
        insta::assert_snapshot!(format!(
            "plan: \n{}\nvalues: \n{}\nstats:\n{}",
            plan, values, stats
        ));
    });

    test_runner(sql, &reference, cache_dir.path()).await;
}

#[tokio::test]
async fn test_provide_schema2() {
    use std::fmt::Write as _;

    let cache_dir = TempDir::new().unwrap();
    let df_ctx = SessionContext::new();
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 4;
    let (liquid_ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_prefetch(false)
        .with_cache_dir(cache_dir.path().to_path_buf())
        .with_max_memory_bytes(1024 * 1024)
        .with_eviction_policy(Box::new(TranscodeEvict))
        .build(config)
        .await
        .unwrap();

    let file_format = ParquetFormat::default().with_enable_pruning(true);
    let listing_options =
        ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");
    let table_path = ListingTableUrl::parse(OPENOBSERVE_FILE).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("_timestamp", DataType::Int64, false),
        Field::new("log", DataType::Utf8, true),
        Field::new("message", DataType::Utf8, true),
        Field::new("kubernetes_namespace_name", DataType::Utf8, false),
    ]));

    df_ctx
        .register_listing_table(
            "default",
            &table_path,
            listing_options.clone(),
            Some(schema.clone()),
            None,
        )
        .await
        .unwrap();
    liquid_ctx
        .register_listing_table("default", &table_path, listing_options, Some(schema), None)
        .await
        .unwrap();

    let queries = [
        "SELECT * from default where log like '%hhj%' order by _timestamp",
        "SELECT date_bin(interval '10 second', to_timestamp_micros(_timestamp), to_timestamp('2001-01-01T00:00:00')) AS zo_sql_key, count(*) AS zo_sql_num from default WHERE log like '%hhj%' or message like '%hhj%' GROUP BY zo_sql_key ORDER BY zo_sql_key",
        "SELECT _timestamp, kubernetes_namespace_name from default order by _timestamp desc limit 100",
    ];

    let mut snapshot = String::new();

    for (idx, sql) in queries.iter().enumerate() {
        let df_results = df_ctx.sql(sql).await.unwrap().collect().await.unwrap();

        let plan = get_physical_plan(sql, &liquid_ctx).await;
        let displayable = DisplayableExecutionPlan::new(plan.as_ref());
        let plan_string = format!("{}", displayable.tree_render());

        // Reset runtime counters so we measure hits from the prefetch run onwards.
        cache.storage().stats();

        let first_liquid_run = liquid_ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(df_results[0].columns(), first_liquid_run[0].columns());

        let entries_after_first_run = cache.storage().stats().total_entries;
        let second_liquid_run = liquid_ctx.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(df_results[0].columns(), second_liquid_run[0].columns());

        let stats = CacheStatsSummary::from_stats(cache.storage().stats(), entries_after_first_run);

        assert!(stats.has_cache_hits());
        assert!(stats.entries_reused());

        writeln!(snapshot, "query[{idx}]: {sql}").unwrap();
        writeln!(snapshot, "plan: \n{}", plan_string).unwrap();
        writeln!(snapshot, "stats:\n{}", stats).unwrap();

        if idx + 1 != queries.len() {
            snapshot.push('\n');
        }
    }

    insta::with_settings!({ filters => vec![(r"usage\.(memory|disk)_bytes: \d+", "usage.${1}_bytes: [bytes]")] }, {
        insta::assert_snapshot!(snapshot);
    });
}

#[tokio::test]
async fn test_provide_schema_with_filter() {
    let cache_dir = TempDir::new().unwrap();
    let sql = r#"select "WatchID", "OS", "EventTime" from hits where "OS" <> 2 order by "WatchID" desc limit 10"#;

    let QueryOutcome {
        values,
        plan,
        stats,
    } = run_sql_with_cache(sql, Box::new(TranscodeEvict), 1024 * 1024, cache_dir.path()).await;

    assert!(stats.has_cache_hits());
    assert!(stats.entries_reused());

    let reference = values.clone();

    insta::with_settings!({ filters => vec![(r"usage\.(memory|disk)_bytes: \d+", "usage.${1}_bytes: [bytes]")] }, {
        insta::assert_snapshot!(format!(
            "plan: \n{}\nvalues: \n{}\nstats:\n{}",
            plan, values, stats
        ));
    });

    let (ctx, _) = LiquidCacheLocalBuilder::new()
        .with_eviction_policy(Box::new(TranscodeEvict))
        .build(SessionConfig::new())
        .await
        .unwrap();

    let file_format = ParquetFormat::default().with_enable_pruning(true);
    let listing_options =
        ListingOptions::new(Arc::new(file_format)).with_file_extension(".parquet");

    let table_path = ListingTableUrl::parse("../../examples/nano_hits.parquet").unwrap();
    let schema = Schema::new(vec![
        Field::new("WatchID", DataType::Int64, true),
        Field::new("EventTime", DataType::Int64, true),
        Field::new("OS", DataType::Int16, true),
    ]);

    ctx.register_listing_table(
        "hits",
        &table_path,
        listing_options.clone(),
        Some(Arc::new(schema)),
        None,
    )
    .await
    .unwrap();

    let results = ctx.sql(sql).await.unwrap().collect().await.unwrap();

    let formatted_results = pretty_format_batches(&results).unwrap().to_string();
    if formatted_results != reference {
        println!("formatted_results: \n{formatted_results}");
        println!("reference: \n{reference}");
    }
    assert_eq!(formatted_results, reference);
}

#[tokio::test]
async fn test_repartitioned_file_scan_cache_correctness() {
    let reference_cache_dir = TempDir::new().unwrap();
    let parallel_cache_dir = TempDir::new().unwrap();
    let sql = r#"select "WatchID", "OS", "EventTime" from hits where "OS" <> 2 order by "WatchID" desc limit 10"#;

    let reference = run_sql_with_cache(
        sql,
        Box::new(TranscodeEvict),
        1024 * 1024,
        reference_cache_dir.path(),
    )
    .await
    .values;

    // DataFusion 55 lowered repartition_file_min_size from 10 MiB to 1 MiB,
    // which splits the 2.3 MiB fixture into four concurrent scan partitions.
    let mut config = SessionConfig::new();
    config.options_mut().execution.target_partitions = 4;
    let (ctx, cache) = LiquidCacheLocalBuilder::new()
        .with_max_memory_bytes(1024 * 1024)
        .with_cache_dir(parallel_cache_dir.path().to_path_buf())
        .with_eviction_policy(Box::new(TranscodeEvict))
        .with_cache_policy(Box::new(LiquidPolicy::new()))
        .build(config)
        .await
        .unwrap();
    ctx.register_parquet("hits", TEST_FILE, ParquetReadOptions::default())
        .await
        .unwrap();

    let plan = get_physical_plan(sql, &ctx).await;
    let plan = format!(
        "{}",
        DisplayableExecutionPlan::new(plan.as_ref()).tree_render()
    );
    assert!(
        plan.contains("files: 4"),
        "expected a repartitioned scan:\n{plan}"
    );

    assert_eq!(get_result(&ctx, sql).await, reference);
    let entries_after_first_run = cache.storage().stats().total_entries;
    assert_eq!(get_result(&ctx, sql).await, reference);

    let stats = cache.storage().stats();
    assert!(stats.runtime.get_with_selection > 0);
    assert!(stats.total_entries >= entries_after_first_run);
}
