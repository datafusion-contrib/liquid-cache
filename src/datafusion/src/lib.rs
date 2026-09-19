#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod io;
pub mod optimizers;
mod reader;
mod sync;
pub(crate) mod utils;

pub mod cache;
pub use cache::{LiquidCacheParquet, LiquidCacheParquetRef};
pub use liquid_cache as storage;
pub use liquid_cache_common as common;
pub use reader::variant_udf::{VariantGetUdf, VariantPretty, VariantToJsonUdf};
pub use reader::{FilterCandidateBuilder, LiquidParquetSource, LiquidPredicate, LiquidRowFilter};
pub use utils::{boolean_buffer_and_then, extract_execution_metrics};

/// Register the variant functions used by LiquidCache in a query session.
pub fn register_variant_functions(ctx: &datafusion::prelude::SessionContext) {
    use datafusion::logical_expr::ScalarUDF;

    ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::default()));
    ctx.register_udf(ScalarUDF::new_from_impl(VariantPretty::default()));
    ctx.register_udf(ScalarUDF::new_from_impl(VariantToJsonUdf::default()));
}
