//! Physical-plan lineage analysis for lineage expressions.
//!
//! This replaces the old logical [`LineageOptimizer`](super) + global
//! `Arc::as_ptr` registry + field-metadata-string machinery. We analyze the
//! *physical* plan directly: for every parquet scan we look at how each of its
//! output columns is consumed by the operators above it (and by the scan's own
//! pushed-down projection/filter), and we derive a typed [`CacheExpression`]
//! per file column describing the safe cache expression.
//!
//! Working on the physical plan buys two things over the previous logical
//! approach:
//!
//! * physical columns are positional, so two join inputs that both expose a
//!   column named `date` can never collide (the old name-keyed metadata map
//!   did); and
//! * the result is a plain value we can hand straight to the scan's
//!   [`LiquidParquetSource`](crate::LiquidParquetSource) — no out-of-band
//!   registry, no stringly schema metadata, and the same code runs in local
//!   mode (full plan) and on the cache server (deserialized fragment).
//!
//! The analysis is deliberately *conservative*: any column that reaches an
//! operator we do not understand is treated as "used raw", which suppresses its
//! hint. Emitting no hint only costs a missed optimization; emitting a wrong
//! hint would let the cache drop data the query still needs.

use std::collections::HashMap;
use std::str::FromStr;

use arrow_schema::DataType;
use datafusion::common::ScalarValue;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::logical_expr::JoinType;
use datafusion::physical_expr::Partitioning;
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::expressions::{CastExpr, Column, Literal};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use liquid_cache::cache::{CacheExpression, Date32Field};

use crate::cache::ColumnLineages;

/// Stable identity of a node within a single analysis pass: the data address of
/// its `Arc`. Valid only for the lifetime of one [`HintAnalyzer::analyze`] call
/// (no clones happen between analysis and the rewrite that consumes it), unlike
/// the old cross-process `Arc::as_ptr` registry.
type NodePtr = usize;

fn node_ptr(plan: &std::sync::Arc<dyn ExecutionPlan>) -> NodePtr {
    std::sync::Arc::as_ptr(plan) as *const () as usize
}

/// One operation applied to a base column on its way up the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Op {
    Extract(Date32Field),
    Variant {
        path: String,
        data_type: Option<DataType>,
    },
    /// Any other consumption (arithmetic, comparison, cast, unknown op, …).
    Other,
}

/// Lineage of one value: which scan column it ultimately came from, and the
/// chain of operations applied to it (innermost first).
#[derive(Debug, Clone)]
struct Usage {
    scan: NodePtr,
    col: usize,
    ops: Vec<Op>,
}

/// Per-node lineage: for each output column, the set of base-column usages that
/// flow into it. Indexed by the node's output schema position.
type LineageMap = Vec<Vec<Usage>>;

/// Accumulated op chains observed for a single (scan, file column).
#[derive(Default)]
struct ColumnStats {
    usages: Vec<Vec<Op>>,
}

/// Analyzes a physical plan and produces, per parquet scan, the typed lineage
/// hint for each of its file columns.
#[derive(Default)]
pub(crate) struct HintAnalyzer {
    /// (scan ptr, file column index) -> observed op chains.
    stats: HashMap<(NodePtr, usize), ColumnStats>,
    /// scan ptr -> file column names and types, for resolving hints at the end.
    scan_columns: HashMap<NodePtr, Vec<(String, DataType)>>,
}

/// Lineage expressions derived from a physical plan, keyed by the analyzed scan
/// nodes. Valid only for the plan it was analyzed from (keyed by `Arc` identity).
///
/// Local mode consumes this directly during the parquet-scan rewrite; the Flight
/// client uses [`Self::for_fragment`] to extract the hints for the single scan
/// inside each pushed-down fragment so they can be shipped to the cache server.
pub struct LineageHints {
    per_scan: HashMap<NodePtr, ColumnLineages>,
}

impl LineageHints {
    /// Analyze a physical plan and derive per-scan lineage expressions.
    pub fn analyze(plan: &std::sync::Arc<dyn ExecutionPlan>) -> Self {
        Self {
            per_scan: HintAnalyzer::analyze(plan),
        }
    }

    /// Whether any scan in the analyzed plan produced a hint.
    pub fn is_empty(&self) -> bool {
        self.per_scan.is_empty()
    }

    /// Merge the hints for every parquet scan reachable under `fragment`.
    ///
    /// Pushed-down fragments are single-scan, so this returns that scan's hints;
    /// `fragment` must be a node from the same plan this map was analyzed from.
    pub fn for_fragment(&self, fragment: &std::sync::Arc<dyn ExecutionPlan>) -> ColumnLineages {
        let mut merged = ColumnLineages::default();
        fragment
            .apply(|node| {
                if let Some(hints) = self.per_scan.get(&node_ptr(node)) {
                    for (name, expr) in hints {
                        merged.insert(name.clone(), expr.clone());
                    }
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
        merged
    }
}

impl HintAnalyzer {
    /// Analyze `plan` and return, keyed by scan node pointer, the lineage expressions
    /// for that scan's file columns.
    pub(crate) fn analyze(
        plan: &std::sync::Arc<dyn ExecutionPlan>,
    ) -> HashMap<NodePtr, ColumnLineages> {
        let mut analyzer = HintAnalyzer::default();
        let root = analyzer.visit(plan);
        // Columns that escape the top of the analyzed plan (returned to the
        // caller / user) are, by definition, consumed as-is.
        analyzer.record_escape(&root);
        analyzer.finish()
    }

    fn record(&mut self, usages: &[Usage]) {
        for usage in usages {
            self.stats
                .entry((usage.scan, usage.col))
                .or_default()
                .usages
                .push(usage.ops.clone());
        }
    }

    /// Record `usages` as escaping (used exactly as-is, no extra op).
    fn record_escape(&mut self, map: &LineageMap) {
        for column in map {
            self.record(column);
        }
    }

    /// Record every base column reachable through `map` as used opaquely, so no
    /// hint survives for a column flowing into an operator we don't model.
    fn record_opaque(&mut self, map: &LineageMap) {
        for column in map {
            for usage in column {
                let mut ops = usage.ops.clone();
                ops.push(Op::Other);
                self.stats
                    .entry((usage.scan, usage.col))
                    .or_default()
                    .usages
                    .push(ops);
            }
        }
    }

    fn visit(&mut self, plan: &std::sync::Arc<dyn ExecutionPlan>) -> LineageMap {
        if let Some(dse) = plan.downcast_ref::<DataSourceExec>() {
            if let Some((cfg, parquet)) = dse.downcast_to_file_source::<ParquetSource>() {
                return self.visit_scan(plan, parquet, cfg);
            }
            return opaque(plan);
        }

        if let Some(proj) = plan.downcast_ref::<ProjectionExec>() {
            let child = self.visit(proj.input());
            return proj
                .expr()
                .iter()
                .map(|pe| lineage_for_expr(&pe.expr, &child))
                .collect();
        }

        if let Some(filter) = plan.downcast_ref::<FilterExec>() {
            let child = self.visit(filter.input());
            let usages = lineage_for_expr(filter.predicate(), &child);
            self.record(&usages);
            return child;
        }

        if let Some(agg) = plan.downcast_ref::<AggregateExec>() {
            let child = self.visit(agg.input());
            for expr in agg.group_expr().input_exprs() {
                let usages = lineage_for_expr(&expr, &child);
                self.record(&usages);
            }
            for (aggr, filter) in agg.aggr_expr().iter().zip(agg.filter_expr()) {
                for expr in aggr.expressions() {
                    let usages = lineage_for_expr(&expr, &child);
                    self.record(&usages);
                }
                for order_by in aggr.order_bys() {
                    let usages = lineage_for_expr(&order_by.expr, &child);
                    self.record(&usages);
                }
                if let Some(filter) = filter {
                    let usages = lineage_for_expr(filter, &child);
                    self.record(&usages);
                }
            }
            return opaque(plan);
        }

        if let Some(sort) = plan.downcast_ref::<SortExec>() {
            let child = self.visit(sort.input());
            for sort_expr in sort.expr().iter() {
                let usages = lineage_for_expr(&sort_expr.expr, &child);
                self.record(&usages);
            }
            return child;
        }

        if let Some(repart) = plan.downcast_ref::<RepartitionExec>() {
            let child = self.visit(repart.input());
            if let Partitioning::Hash(exprs, _) = repart.partitioning() {
                for expr in exprs {
                    let usages = lineage_for_expr(expr, &child);
                    self.record(&usages);
                }
            }
            return child;
        }

        // Schema-preserving passthroughs: the child lineage flows up unchanged.
        // (Batch coalescing is handled inside arrow-rs in DF 54, so there is no
        // dedicated CoalesceBatchesExec to model here.)
        if plan.downcast_ref::<CoalescePartitionsExec>().is_some()
            || plan.downcast_ref::<GlobalLimitExec>().is_some()
            || plan.downcast_ref::<LocalLimitExec>().is_some()
            || plan.downcast_ref::<SortPreservingMergeExec>().is_some()
        {
            let children = plan.children();
            if children.len() == 1 {
                return self.visit(children[0]);
            }
        }

        if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
            return self.visit_hash_join(plan, join);
        }

        // Unknown operator: analyze children but treat every column they expose
        // as used opaquely, then expose nothing upward.
        for child in plan.children() {
            let child_map = self.visit(child);
            self.record_opaque(&child_map);
        }
        opaque(plan)
    }

    fn visit_scan(
        &mut self,
        plan: &std::sync::Arc<dyn ExecutionPlan>,
        parquet: &ParquetSource,
        _cfg: &FileScanConfig,
    ) -> LineageMap {
        let ptr = node_ptr(plan);
        let table_schema = parquet.table_schema();
        let file_schema = table_schema.file_schema();
        let file_field_count = file_schema.fields().len();
        let table_field_count = table_schema.table_schema().fields().len();

        self.scan_columns.insert(
            ptr,
            file_schema
                .fields()
                .iter()
                .map(|f| (f.name().to_string(), f.data_type().clone()))
                .collect(),
        );

        // Base lineage in terms of the *table* schema (file columns first, then
        // partition columns). Partition columns are not cached, so they carry
        // no base usage.
        let base: LineageMap = (0..table_field_count)
            .map(|i| {
                if i < file_field_count {
                    vec![Usage {
                        scan: ptr,
                        col: i,
                        ops: Vec::new(),
                    }]
                } else {
                    Vec::new()
                }
            })
            .collect();

        // A pushed-down filter consumes columns directly at the scan (with
        // filter pushdown enabled, `WHERE col LIKE '%x%'` lives here rather than
        // in a FilterExec above). Record those usages so substring searches are
        // detected and columns used in other predicates are not given a partial representation.
        if let Some(predicate) = parquet.filter() {
            let usages = lineage_for_expr(&predicate, &base);
            self.record(&usages);
        }

        // Apply the scan's pushed-down projection (which may itself contain
        // date_part/variant_get), if any.
        let out = match parquet.projection() {
            Some(projection) => projection
                .as_ref()
                .iter()
                .map(|pe| lineage_for_expr(&pe.expr, &base))
                .collect::<LineageMap>(),
            None => base,
        };

        // Guard: the lineage map must align with the node's output schema, or
        // parent operators would index the wrong columns. If it doesn't, fall
        // back to opaque (no hints) rather than risk a wrong attribution.
        if out.len() != plan.schema().fields().len() {
            return opaque(plan);
        }
        out
    }

    fn visit_hash_join(
        &mut self,
        plan: &std::sync::Arc<dyn ExecutionPlan>,
        join: &HashJoinExec,
    ) -> LineageMap {
        let left = self.visit(join.left());
        let right = self.visit(join.right());

        // Join keys consume their columns as-is.
        for (l, r) in join.on() {
            let lu = lineage_for_expr(l, &left);
            self.record(&lu);
            let ru = lineage_for_expr(r, &right);
            self.record(&ru);
        }

        // Equi-joins without a residual filter pass lineage through, applying
        // their output projection to the concatenated input columns. Any
        // other shape (semi/anti/mark joins, a residual filter we don't map)
        // is treated opaquely.
        let passthrough = join.filter().is_none()
            && matches!(
                join.join_type(),
                JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
            );

        if passthrough {
            let mut out = left;
            out.extend(right);
            if let Some(projection) = &join.projection {
                out = projection.iter().map(|&index| out[index].clone()).collect();
            }
            if out.len() == plan.schema().fields().len() {
                return out;
            }
            // Width mismatch: fall through to opaque handling below.
            self.record_opaque(&out);
            return opaque(plan);
        }

        self.record_opaque(&left);
        self.record_opaque(&right);
        opaque(plan)
    }

    fn finish(self) -> HashMap<NodePtr, ColumnLineages> {
        let mut per_scan: HashMap<NodePtr, ColumnLineages> = HashMap::new();

        for ((scan, col), stats) in &self.stats {
            let Some(columns) = self.scan_columns.get(scan) else {
                continue;
            };
            let Some((name, data_type)) = columns.get(*col) else {
                continue;
            };
            if let Some(expr) = derive_hint(data_type, &stats.usages) {
                per_scan
                    .entry(*scan)
                    .or_default()
                    .insert(name.clone(), std::sync::Arc::new(expr));
            }
        }

        per_scan
    }
}

/// An output column with no derivable base lineage (each entry empty).
fn opaque(plan: &std::sync::Arc<dyn ExecutionPlan>) -> LineageMap {
    vec![Vec::new(); plan.schema().fields().len()]
}

/// Compute the lineage of a physical expression against the lineage of its
/// input columns. Mirrors the logical analyzer's per-expression rules.
fn lineage_for_expr(expr: &std::sync::Arc<dyn PhysicalExpr>, input: &LineageMap) -> Vec<Usage> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        return input.get(column.index()).cloned().unwrap_or_default();
    }

    if let Some(sf) = expr.downcast_ref::<ScalarFunctionExpr>() {
        let name = sf.fun().name();
        let args = sf.args();
        if name.eq_ignore_ascii_case("date_part")
            && args.len() == 2
            && let Some(field) = literal_date_field(&args[0])
        {
            let mut usages = lineage_for_expr(&args[1], input);
            for usage in &mut usages {
                usage.ops.push(Op::Extract(field));
            }
            return usages;
        }
        if name.eq_ignore_ascii_case("variant_get")
            && (args.len() == 2 || args.len() == 3)
            && let Some(path) = literal_utf8(&args[1])
        {
            let data_type = args.get(2).and_then(literal_data_type);
            let mut usages = lineage_for_expr(&args[0], input);
            for usage in &mut usages {
                usage.ops.push(Op::Variant {
                    path: path.clone(),
                    data_type: data_type.clone(),
                });
            }
            return usages;
        }
        return propagate_other(expr, input);
    }

    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let mut usages = lineage_for_expr(cast.expr(), input);
        for usage in &mut usages {
            usage.ops.push(Op::Other);
        }
        return usages;
    }

    if expr.downcast_ref::<Literal>().is_some() {
        return Vec::new();
    }

    propagate_other(expr, input)
}

/// Default propagation: any base column reached through `expr`'s children is
/// consumed via some unmodelled operation, recorded as [`Op::Other`].
fn propagate_other(expr: &std::sync::Arc<dyn PhysicalExpr>, input: &LineageMap) -> Vec<Usage> {
    let mut combined = Vec::new();
    for child in expr.children() {
        let mut usages = lineage_for_expr(child, input);
        for usage in &mut usages {
            usage.ops.push(Op::Other);
        }
        combined.extend(usages);
    }
    combined
}

/// Decide the lineage expression for one file column from its observed op chains.
fn derive_hint(data_type: &DataType, usages: &[Vec<Op>]) -> Option<CacheExpression> {
    if usages.is_empty() {
        return None;
    }
    if let Some(expr) = derive_date(data_type, usages) {
        return Some(expr);
    }
    if let Some(expr) = derive_variant(usages) {
        return Some(expr);
    }
    None
}

fn derive_date(data_type: &DataType, usages: &[Vec<Op>]) -> Option<CacheExpression> {
    if !is_date_part_type(data_type) {
        return None;
    }
    let mut fields = Vec::new();
    for chain in usages {
        // Take the leading run of Extract ops; the column must never be used
        // any other way (including raw passthrough).
        let mut leading = Vec::new();
        for op in chain {
            match op {
                Op::Extract(field) => leading.push(*field),
                _ => break,
            }
        }
        if leading.is_empty() {
            return None;
        }
        fields.extend(leading);
    }
    CacheExpression::extract_date32_many(fields)
}

fn derive_variant(usages: &[Vec<Op>]) -> Option<CacheExpression> {
    let mut requests: Vec<(String, DataType)> = Vec::new();
    let mut seen: HashMap<String, Option<DataType>> = HashMap::new();
    let mut saw_variant = false;

    for chain in usages {
        match chain.first() {
            Some(Op::Variant { path, data_type }) => {
                saw_variant = true;
                match seen.get(path) {
                    Some(existing) => {
                        // Conflicting requested types for the same path: bail.
                        if existing != data_type {
                            return None;
                        }
                    }
                    None => {
                        seen.insert(path.clone(), data_type.clone());
                        // A variant_get without an explicit type hint cannot be
                        // represented as a typed column; only record typed paths.
                        if let Some(dt) = data_type {
                            requests.push((path.clone(), dt.clone()));
                        }
                    }
                }
            }
            // Raw passthrough of a variant column does not invalidate the hint:
            // the partial representation keeps a disk backing for full reads.
            None => continue,
            _ => return None,
        }
    }

    if saw_variant && !requests.is_empty() {
        Some(CacheExpression::variant_get_many(requests))
    } else {
        None
    }
}

fn literal_utf8(expr: &std::sync::Arc<dyn PhysicalExpr>) -> Option<String> {
    let literal = expr.downcast_ref::<Literal>()?;
    match literal.value() {
        ScalarValue::Utf8(Some(v))
        | ScalarValue::LargeUtf8(Some(v))
        | ScalarValue::Utf8View(Some(v)) => Some(v.clone()),
        _ => None,
    }
}

fn literal_data_type(expr: &std::sync::Arc<dyn PhysicalExpr>) -> Option<DataType> {
    literal_utf8(expr).and_then(|spec| DataType::from_str(&spec).ok())
}

fn literal_date_field(expr: &std::sync::Arc<dyn PhysicalExpr>) -> Option<Date32Field> {
    let text = literal_utf8(expr)?;
    let lowered = text.to_ascii_lowercase();
    match lowered.as_str() {
        "dow" | "dayofweek" | "day_of_week" => return Some(Date32Field::DayOfWeek),
        _ => {}
    }
    match lowered.as_str() {
        "year" => Some(Date32Field::Year),
        "month" => Some(Date32Field::Month),
        "day" => Some(Date32Field::Day),
        _ => None,
    }
}

fn is_date_part_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Date32 | DataType::Timestamp(_, _))
}

/// Converts one parquet scan node, given its derived hints, into its
/// liquid-cache-backed equivalent. Returns `None` for non-parquet nodes.
pub(crate) type ScanConverter<'a> = dyn FnMut(
        &std::sync::Arc<dyn ExecutionPlan>,
        ColumnLineages,
    ) -> Option<std::sync::Arc<dyn ExecutionPlan>>
    + 'a;

/// Rewrite every parquet scan in `plan`, attaching the lineage expressions derived for
/// it. `hints` resolves a scan's node pointer to its hints; scans absent from
/// the map get [`ColumnLineages::default`].
pub(crate) fn rewrite_with_hints(
    plan: std::sync::Arc<dyn ExecutionPlan>,
    convert: &mut ScanConverter<'_>,
    hints: &HashMap<NodePtr, ColumnLineages>,
) -> std::sync::Arc<dyn ExecutionPlan> {
    plan.transform_up(|node| {
        let ptr = node_ptr(&node);
        let scan_hints = hints.get(&ptr).cloned().unwrap_or_default();
        if let Some(new_node) = convert(&node, scan_hints) {
            Ok(Transformed::new(
                new_node,
                true,
                TreeNodeRecursion::Continue,
            ))
        } else {
            Ok(Transformed::no(node))
        }
    })
    .unwrap()
    .data
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Date32Array, StringArray, TimestampMicrosecondArray};
    use arrow_schema::{Field, Schema, TimeUnit};
    use datafusion::prelude::{SessionConfig, SessionContext};
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    fn write_fixture(path: &std::path::Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_ts",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("date", DataType::Date32, false),
            Field::new("url", DataType::Utf8, false),
        ]));
        let timestamps: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            1_609_459_200_000_000,
            1_640_995_200_000_000,
        ]));
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![18_900, 19_000]));
        let urls: ArrayRef = Arc::new(StringArray::from(vec![
            "https://example.com/a",
            "https://example.com/b",
        ]));
        let batch = arrow::record_batch::RecordBatch::try_new(
            Arc::clone(&schema),
            vec![timestamps, dates, urls],
        )
        .unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    async fn hints_for(sql: &str) -> ColumnLineages {
        let mut config = SessionConfig::new();
        // Mirror liquid cache: predicates are pushed into the parquet scan.
        config.options_mut().execution.parquet.pushdown_filters = true;
        let ctx = SessionContext::new_with_config(config);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fixture.parquet");
        write_fixture(&path);
        ctx.register_parquet("t", path.to_str().unwrap(), Default::default())
            .await
            .unwrap();

        let df = ctx.sql(sql).await.unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let map = LineageHints::analyze(&plan);
        // Single-table queries: one scan, so the merged fragment hints are it.
        map.for_fragment(&plan)
    }

    fn date(fields: &[Date32Field]) -> Arc<CacheExpression> {
        Arc::new(CacheExpression::extract_date32_many(fields.iter().copied()).unwrap())
    }

    #[tokio::test]
    async fn extract_single_component() {
        let hints = hints_for("SELECT EXTRACT(YEAR FROM date) AS y FROM t").await;
        assert_eq!(hints.get("date"), Some(&date(&[Date32Field::Year])));
    }

    #[tokio::test]
    async fn extract_multiple_components_are_unioned() {
        let hints =
            hints_for("SELECT EXTRACT(DAY FROM date) AS d, EXTRACT(MONTH FROM date) AS m FROM t")
                .await;
        assert_eq!(
            hints.get("date"),
            Some(&date(&[Date32Field::Month, Date32Field::Day]))
        );
    }

    #[tokio::test]
    async fn extract_from_timestamp_column() {
        let hints = hints_for("SELECT EXTRACT(YEAR FROM event_ts) AS y FROM t").await;
        assert_eq!(hints.get("event_ts"), Some(&date(&[Date32Field::Year])));
    }

    #[tokio::test]
    async fn raw_column_gets_no_hint() {
        let hints = hints_for("SELECT date FROM t").await;
        assert_eq!(hints.get("date"), None);
    }

    #[tokio::test]
    async fn mixed_raw_and_extract_gets_no_hint() {
        // `date` escapes raw in the projection, so it must remain available in full.
        let hints = hints_for("SELECT date, EXTRACT(YEAR FROM date) AS y FROM t").await;
        assert_eq!(hints.get("date"), None);
    }

    #[tokio::test]
    async fn aggregate_filter_records_raw_column_use() {
        let hints = hints_for(
            "SELECT AVG(EXTRACT(YEAR FROM date)) \
             FILTER (WHERE date = DATE '2021-01-01') FROM t",
        )
        .await;

        // The aggregate argument needs only YEAR, but its filter needs the
        // exact date, so retaining only YEAR would change the result.
        assert_eq!(hints.get("date"), None);
    }

    #[tokio::test]
    async fn aggregate_order_by_records_raw_column_use() {
        let hints =
            hints_for("SELECT FIRST_VALUE(EXTRACT(MONTH FROM date) ORDER BY date DESC) FROM t")
                .await;

        // The aggregate value needs only MONTH, but chronological ordering
        // needs the exact date.
        assert_eq!(hints.get("date"), None);
    }
}
