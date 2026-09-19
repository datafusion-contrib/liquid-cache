//! Exercise the real Flight boundary against local mode and plain DataFusion.

use std::{
    fs::File,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use arrow::{
    array::{ArrayRef, BinaryArray, Date32Array, Int32Array, StringArray},
    compute::concat_batches,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use arrow_flight::flight_service_server::FlightServiceServer;
use datafusion::{
    common::tree_node::{TreeNode, TreeNodeRecursion},
    prelude::{SessionConfig, SessionContext},
};
use liquid_cache::cache::{
    AlwaysHydrate, CacheEntry, CacheExpression, Date32Field, EvictionPolicy, LiquidPolicy,
    TranscodeEvict, policies::EvictionOutcome,
};
use liquid_cache_datafusion::{
    LiquidCacheParquetRef, optimizers::LineageHints, register_variant_functions,
};
use liquid_cache_datafusion_client::{LiquidCacheClientBuilder, LiquidCacheClientExec};
use liquid_cache_datafusion_local::LiquidCacheLocalBuilder;
use parquet::{arrow::ArrowWriter, variant::json_to_variant};
use tokio::{net::TcpListener, task::JoinHandle};

use crate::LiquidCacheService;

type ObservedLineages = Arc<Mutex<Vec<(ArrayRef, Option<CacheExpression>)>>>;

#[derive(Debug)]
struct ObserveEviction(ObservedLineages);

impl EvictionPolicy for ObserveEviction {
    fn evict(&self, entry: &CacheEntry, lineage: Option<&CacheExpression>) -> EvictionOutcome {
        if let CacheEntry::MemoryArrow(array) = entry {
            self.0
                .lock()
                .unwrap()
                .push((array.clone(), lineage.cloned()));
        }
        TranscodeEvict.evict(entry, lineage)
    }
}

struct Fixture {
    contexts: [SessionContext; 3], // local, Flight client, plain DataFusion
    caches: [LiquidCacheParquetRef; 2],
    observed: [ObservedLineages; 2],
    server: JoinHandle<()>,
    _dir: tempfile::TempDir,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

impl Fixture {
    async fn new(memory_bytes: usize) -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("local")).unwrap();
        let observed: [ObservedLineages; 2] = Default::default();
        let mut config = SessionConfig::new().with_target_partitions(1);
        config
            .options_mut()
            .execution
            .parquet
            .schema_force_view_types = false;
        config.options_mut().execution.parquet.skip_metadata = false;
        let (local, local_cache) = LiquidCacheLocalBuilder::new()
            .with_cache_dir(dir.path().join("local"))
            .with_batch_size(8192 * 2)
            .with_max_memory_bytes(memory_bytes)
            .with_eviction_policy(Box::new(ObserveEviction(observed[0].clone())))
            .build(config.clone())
            .await
            .unwrap();
        let service = LiquidCacheService::new(
            LiquidCacheService::context().unwrap(),
            Some(memory_bytes),
            Some(dir.path().join("remote")),
            Box::new(LiquidPolicy::new()),
            Box::new(ObserveEviction(observed[1].clone())),
            Box::new(AlwaysHydrate::new()),
        )
        .await
        .unwrap();
        let remote_cache = service.cache().clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(FlightServiceServer::new(service))
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let client = LiquidCacheClientBuilder::new(format!("http://{address}"))
            .build(config.clone())
            .unwrap();
        let baseline = SessionContext::new_with_config(config);
        register_variant_functions(&baseline);
        let contexts = [local, client, baseline];
        for (table, offset) in [("a", 0), ("b", 400)] {
            let path = dir.path().join(format!("{table}.parquet"));
            write_fixture(&path, offset);
            for ctx in &contexts {
                ctx.register_parquet(table, path.to_str().unwrap(), Default::default())
                    .await
                    .unwrap();
            }
        }
        Self {
            contexts,
            caches: [local_cache, remote_cache],
            observed,
            server,
            _dir: dir,
        }
    }

    async fn assert_query(&self, sql: &str) {
        let expected = query(&self.contexts[2], sql).await;
        for (ctx, cache) in self.contexts[..2].iter().zip(&self.caches) {
            assert_eq!(query(ctx, sql).await, expected, "cold: {sql}");
            let entries = cache.storage().stats().total_entries;
            assert!(entries > 0, "query must exercise the cache: {sql}");
            assert_eq!(query(ctx, sql).await, expected, "warm: {sql}");
            let stats = cache.storage().stats();
            assert_eq!(stats.total_entries, entries);
            assert!(
                stats.runtime.get + stats.runtime.get_with_selection + stats.runtime.eval_predicate
                    > 0
            );
        }
    }

    fn assert_lineages(&self, data_type: &DataType, expected: &[Option<CacheExpression>]) {
        // Compare sets because a query can materialize and evict the same column repeatedly.
        let canonical = |values: Vec<Option<CacheExpression>>| {
            let mut values: Vec<_> = values
                .into_iter()
                .map(|expr| expr.map(|expr| expr.to_metadata_value()))
                .collect();
            values.sort();
            values.dedup();
            values
        };
        for observed in &self.observed {
            let values = observed
                .lock()
                .unwrap()
                .iter()
                .filter(|(array, _)| array.data_type() == data_type)
                .map(|(_, expr)| expr.clone())
                .collect();
            assert_eq!(canonical(values), canonical(expected.to_vec()));
        }
    }
}

async fn query(ctx: &SessionContext, sql: &str) -> RecordBatch {
    let batches = tokio::time::timeout(Duration::from_secs(30), async {
        ctx.sql(sql).await.unwrap().collect().await.unwrap()
    })
    .await
    .expect("query timed out");
    concat_batches(&batches[0].schema(), &batches).unwrap()
}

fn write_fixture(path: &Path, offset: i32) {
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        Some(r#"{"age":30,"name":"Alice"}"#),
        Some(r#"{"age":25,"name":"Bob"}"#),
        Some(r#"{"name":"Charlie"}"#),
        None,
    ]));
    let variant = json_to_variant(&json).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("date", DataType::Date32, true),
        variant.field("data").as_ref().clone(),
        Field::new("payload", DataType::Binary, true),
    ]));
    let dates = Date32Array::from(vec![
        Some(18_628 + offset),
        Some(19_024 + offset),
        Some(19_390 + offset),
        None,
    ]);
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(dates),
            ArrayRef::from(variant),
            Arc::new(BinaryArray::from(vec![
                Some(&[0xff, 0x00][..]),
                None,
                Some(&[0x80][..]),
                Some(&[][..]),
            ])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[tokio::test]
async fn client_projection_lineage_reaches_server_cache() {
    let fixture = Fixture::new(0).await;
    // The projection stays above the client-side join. Both components must travel.
    let sql = "SELECT a.id, EXTRACT(YEAR FROM a.date) AS y, EXTRACT(MONTH FROM a.date) AS m FROM a JOIN b ON a.id = b.id ORDER BY a.id";
    let plan = fixture.contexts[1]
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let mut remote_scans = 0;
    plan.apply(|node| {
        if node.is::<LiquidCacheClientExec>() {
            // The server cannot recover this lineage from the fragment alone.
            assert!(
                LineageHints::analyze(node.children()[0]).is_empty(),
                "{}",
                datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
            );
            remote_scans += 1;
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    assert_eq!(remote_scans, 2);
    fixture.assert_query(sql).await;
    fixture.assert_lineages(
        &DataType::Date32,
        &[CacheExpression::extract_date32_many([
            Date32Field::Year,
            Date32Field::Month,
        ])],
    );
    for cache in &fixture.caches {
        let trace = cache.consume_event_trace().to_string();
        assert!(trace.contains("expr=ExtractDate32[Year, Month]"), "{trace}");
    }
}

#[tokio::test]
async fn mixed_raw_and_derived_usage_keeps_full_column() {
    let fixture = Fixture::new(0).await;
    fixture
        .assert_query("SELECT id, date, payload, EXTRACT(YEAR FROM date) AS y FROM a ORDER BY id")
        .await;
    fixture.assert_lineages(&DataType::Date32, &[None]);
}

#[tokio::test]
async fn join_lineage_is_scoped_to_each_scan() {
    let fixture = Fixture::new(0).await;
    fixture.assert_query("SELECT a.id, EXTRACT(YEAR FROM a.date) AS y, EXTRACT(MONTH FROM b.date) AS m FROM a JOIN b ON a.id = b.id ORDER BY a.id").await;
    fixture.assert_lineages(
        &DataType::Date32,
        &[
            Some(CacheExpression::extract_date32(Date32Field::Year)),
            Some(CacheExpression::extract_date32(Date32Field::Month)),
        ],
    );
    for observed in &fixture.observed {
        for (array, expression) in observed.lock().unwrap().iter() {
            if let Some(dates) = array.as_any().downcast_ref::<Date32Array>() {
                let field = match dates.value(0) {
                    18_628 => Date32Field::Year,
                    19_028 => Date32Field::Month,
                    value => panic!("unexpected date from join input: {value}"),
                };
                assert_eq!(*expression, Some(CacheExpression::extract_date32(field)));
            }
        }
    }
}

#[tokio::test]
async fn variant_functions_and_lineage_cross_flight() {
    let fixture = Fixture::new(0).await;
    fixture.assert_query("SELECT id, variant_get(data, 'age', 'Int64') AS age, variant_get(data, 'name', 'Utf8') AS name FROM a ORDER BY id").await;
    let variant_type = fixture.contexts[2]
        .table("a")
        .await
        .unwrap()
        .schema()
        .field_with_unqualified_name("data")
        .unwrap()
        .data_type()
        .clone();
    fixture.assert_lineages(
        &variant_type,
        &[Some(CacheExpression::variant_get_many([
            ("age", DataType::Int64),
            ("name", DataType::Utf8),
        ]))],
    );
    // The predicate and aggregate execute on the server and require its UDF registry.
    fixture
        .assert_query("SELECT COUNT(*) FROM a WHERE variant_get(data, 'age', 'Int64') > 26")
        .await;
    fixture.assert_query("SELECT id, variant_pretty(data), variant_to_json(data), variant_to_json(variant_get(data, 'name')) FROM a ORDER BY id").await;
    fixture
        .assert_query("SELECT id, data FROM a ORDER BY id")
        .await;
}

#[tokio::test]
async fn later_raw_query_hydrates_cached_data() {
    let fixture = Fixture::new(usize::MAX).await;
    fixture.assert_query("SELECT id, EXTRACT(YEAR FROM date) AS y, variant_get(data, 'age', 'Int64') AS age FROM a ORDER BY id").await;
    for cache in &fixture.caches {
        cache.flush_data().await.unwrap();
        let stats = cache.storage().stats();
        assert!(stats.disk_arrow_entries > 0);
        assert_eq!(stats.memory_arrow_entries, 0);
    }
    let sql = "SELECT id, date, variant_to_json(data) FROM a ORDER BY id";
    let expected = query(&fixture.contexts[2], sql).await;
    for (ctx, cache) in fixture.contexts[..2].iter().zip(&fixture.caches) {
        assert_eq!(query(ctx, sql).await, expected);
        let stats = cache.storage().stats();
        assert!(stats.runtime.read_io_count > 0);
        assert!(stats.memory_arrow_entries > 0);
    }
}
