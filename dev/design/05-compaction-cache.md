# File compaction in LiquidCache

Status: direction agreed; implementation details remain TODO. DuckLake is the first integration. Step 2 focuses on compaction correctness; Step 3 adds cache reuse.

The cited context describes existing systems and constraints, not additional implementation decisions.

**Editing instructions:** Use simple, concise technical English. Organize agreed decisions, relevant context, cited research, and open questions under the corresponding step. Do not add undiscussed APIs, architecture, execution flows, algorithms, or implementation plans. Leave those as TODO; do not turn research findings or assumptions into agreed decisions.

## Step 0: Revisit the LiquidCache client/server implementation

Bring the LiquidCache client/server implementation to feature parity with local mode. In particular, query lineage must be pushed down to the server. This is a prerequisite for the DuckLake integration and compaction work.

### Implementation

The client analyzes lineage before splitting the physical plan into remote fragments. It sends each scan's expressions with the plan registration request. The server decodes these expressions and attaches them when rewriting Parquet scans to LiquidParquet scans.

Local, client, and server sessions register the same variant functions. Client and server Parquet settings preserve Arrow metadata, and the client no longer forces binary columns to strings. The server restores variant return-field metadata omitted by DataFusion's plan serialization, so nested variant functions work remotely.

Lineage analysis applies hash-join output projections when mapping columns back to their scans. The cache passes registered lineage to the eviction policy even when an incoming batch cannot fit in memory.

Flight integration tests compare client/server queries with local mode and plain DataFusion. They cover date extraction, multiple variant paths, nested variant functions, raw binary data, mixed raw and derived usage, joins with overlapping column names, cache reuse, and disk hydration for a later raw query. They also check the expressions received by the eviction policy and cache reads.

Validation: 200 library tests pass across the core, DataFusion integration, local, client, and server crates; one existing client test is ignored. `cargo check --workspace --all-targets` passes.

### Current limits

The current core retains lineage, but `TranscodeEvict` does not use it and there is no squeezed cache representation. The tests verify lineage propagation and ordinary disk hydration; squeeze-specific reuse and fallback cannot be tested until the core supports them.

Distributed dynamic filter pushdown remains disabled because runtime updates from client-side operators do not reach remote scans.

## Step 1: DuckLake–LiquidCache catalog provider

Add a DataFusion catalog provider that integrates DuckLake with LiquidCache. Users register the catalog and query its tables normally, without manually wiring DuckLake and LiquidCache together. This step makes LiquidCache ergonomic for DuckLake users.

Use [`datafusion-ducklake`](https://github.com/datafusion-contrib/datafusion-ducklake) for this integration. DuckLake is the first integration point; the core compaction feature should not depend on it. Other lakehouse formats could be supported through their own integrations.

### Catalog context

DuckLake defines its catalog through SQL tables and transactions. With PostgreSQL, clients connect directly to the database. [Catalog SQL](https://ducklake.select/docs/stable/specification/queries), [catalog backends](https://ducklake.select/docs/stable/duckdb/usage/choosing_a_catalog_database)

### Integration test setup

PostgreSQL is primarily a service for testing the catalog integration. In this test setup, it runs alongside LiquidCache and, once added, compaction. Compute runs remotely and uses the DuckLake–LiquidCache catalog provider. Object storage is separate.

The focus is how LiquidCache compacts files. Hosting PostgreSQL is not a core requirement of the compaction feature.

### TODO

Provider API and implementation details.

## Step 1.1: Direct LiquidParquet planning

Investigate whether the table providers exposed by our catalog provider can produce LiquidParquet plans directly. This could remove the need for optimizer rules that rewrite Parquet plans into LiquidParquet plans and make the integration simpler.

This is an open design question. Query lineage must still reach the server as required by Step 0.

### TODO

Determine whether direct planning can replace the scan-rewrite rules and how query lineage would be passed down.

## Step 2: Compaction correctness

LiquidCache will provide a compaction function, such as `merge_adjacent_files`. Users can call it from their existing maintenance jobs. Compaction runs on the LiquidCache server: the caller invokes the function, and the server performs the work.

Focus on correct compacted output and accurate reporting of the input and output files. LiquidCache returns which files were compacted and which new files were produced. The client decides how to update its lakehouse catalog. LiquidCache does not update the catalog.

PostgreSQL from Step 1 supports testing the catalog integration. The core compaction feature is independent of catalog hosting and scheduling. Cache reuse is outside this step's scope.

### Compaction context

Compaction exists across lakehouse formats:

| Format | Operation |
| --- | --- |
| DuckLake | [`merge_adjacent_files`](https://ducklake.select/docs/stable/duckdb/maintenance/merge_adjacent_files) merges files and can sort the output. |
| Iceberg | [`rewrite_data_files`](https://iceberg.apache.org/docs/latest/spark-procedures/#rewrite_data_files) supports bin-packing and sorting. |
| Delta Lake | [`OPTIMIZE`](https://docs.delta.io/optimizations-oss/) combines small files. |

In DuckLake's existing implementation, the client performs the Parquet rewrite and commits metadata changes; PostgreSQL stores the metadata. Compaction runs when a maintenance operation is invoked. The `auto_compact` option controls table eligibility, not automatic scheduling. [Catalog operations](https://ducklake.select/docs/stable/specification/queries), [compaction documentation](https://ducklake.select/docs/stable/duckdb/maintenance/merge_adjacent_files)

DuckLake tracks deletes and snapshot visibility separately from file contents. It delays physical file cleanup to protect active reads and retained history. These constraints matter for compaction correctness and later cache reuse. [Delete files](https://ducklake.select/docs/stable/specification/tables/ducklake_delete_file), [transactions](https://ducklake.select/docs/stable/duckdb/advanced_features/transactions), [file cleanup](https://ducklake.select/docs/stable/duckdb/maintenance/cleanup_of_files)

Related work: [AutoComp, SIGMOD 2025](https://arxiv.org/html/2504.04186v1) studies lakehouse compaction selection and scheduling.

### TODO

- Compaction flow.
- Function API and integration interface.
- Implementation details.

## Step 3: Cache reuse across compaction

Add cache reuse across compaction after Step 2 establishes correctness. Queries must use LiquidCache to benefit from it.

### Motivation

LiquidCache works well when Parquet files stay unchanged. In a lakehouse, files are added, rows are deleted, and files are compacted. Compaction is especially problematic:

```text
a.parquet + b.parquet → c.parquet
```

The data may be unchanged, but its location and organization change. Cached data from `a.parquet` and `b.parquet` cannot currently be reused through the new file identity. The compaction operations in Step 2 can create this problem across lakehouse formats.

LiquidCache stores column batches, typically 8,192 rows. Its current DataFusion cache key is `(file ID, column ID, row group ID, batch ID)`. [Cache keys](../../src/datafusion/src/cache/id.rs), [batch size](../../src/core/src/cache/builders.rs)

The original idea is to track lineage so that data moved by compaction can be remapped to existing cached data. How to do this remains open.

### Research context

DuckLake row IDs survive both compaction and updates. Equal row IDs therefore do not prove equal values. Row IDs can be derived from a file's starting ID or stored explicitly inside Parquet. Catalog metadata alone does not describe every possible row reordering. [Row lineage](https://ducklake.select/docs/stable/duckdb/advanced_features/row_lineage), [file metadata](https://ducklake.select/docs/stable/specification/tables/ducklake_data_file)

Merging files can split or combine existing cache batches. Sorting can also change row order. Unchanged logical data therefore does not imply that an output batch matches a cached source batch. [Sorted compaction](https://ducklake.select/docs/stable/duckdb/maintenance/merge_adjacent_files)

We have not established whether `datafusion-ducklake`'s current APIs expose the information needed for cache reuse.

### Related work

| Work | Relevance |
| --- | --- |
| [Lance stable row IDs and remap separation](https://github.com/lance-format/lance/discussions/3694) | Discusses keeping indexes useful after compaction through stable IDs or row-address mappings. The linked discussion is a design proposal. |
| [Iceberg v3 row lineage](https://iceberg.apache.org/spec/#row-lineage) | Separates row identity from the sequence number of its last update. |
| [Delta Lake row tracking](https://docs.delta.io/delta-row-tracking/) | Tracks stable row IDs and row commit versions. |
| [dLSM, 2016](https://arxiv.org/abs/1606.02015) | Studies cache invalidation caused by compaction moving data in LSM trees. |
| [Databricks disk cache](https://docs.databricks.com/aws/en/optimizations/disk-cache) | Provides a baseline for invalidating cached Parquet data after file changes. The cited docs do not establish reuse across compaction. |

### TODO

Lineage tracking and the cache-reuse implementation.
