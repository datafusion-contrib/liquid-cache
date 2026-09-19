#![warn(missing_docs)]
#![cfg_attr(not(doctest), doc = include_str!(concat!("../", std::env!("CARGO_PKG_README"))))]
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
mod client_exec;
mod metrics;
mod optimizer;
pub use client_exec::LiquidCacheClientExec;
use datafusion::{
    common::config::ConfigNonZeroUsize,
    error::{DataFusionError, Result},
    execution::{SessionStateBuilder, object_store::ObjectStoreUrl, runtime_env::RuntimeEnv},
    prelude::*,
};
use fastrace_tonic::FastraceClientService;
pub use optimizer::PushdownOptimizer;
use tonic::transport::Channel;

pub use liquid_cache_common as common;

#[cfg(test)]
mod tests;

/// The builder for LiquidCache client state.
///
/// # Example
///
/// ```ignore
/// use datafusion::execution::object_store::ObjectStoreUrl;
/// use datafusion::prelude::SessionConfig;
/// use liquid_cache_datafusion_client::LiquidCacheClientBuilder;
/// use std::collections::HashMap;
///
/// let mut s3_options = HashMap::new();
/// s3_options.insert("access_key_id".to_string(), "your-access-key".to_string());
/// s3_options.insert("secret_access_key".to_string(), "your-secret-key".to_string());
/// s3_options.insert("region".to_string(), "us-east-1".to_string());
///
/// let ctx = LiquidCacheClientBuilder::new("localhost:15214")
///     .with_object_store(ObjectStoreUrl::parse("s3://my_bucket").unwrap(), Some(s3_options))
///     .build(SessionConfig::from_env().unwrap())
///     .unwrap();
///
/// ctx.register_parquet("my_table", "s3://my_bucket/my_table.parquet", Default::default())
///     .await?;
/// let df = ctx.sql("SELECT * FROM my_table").await?.show().await?;
/// println!("{:?}", df);
/// ```
pub struct LiquidCacheClientBuilder {
    object_stores: Vec<(ObjectStoreUrl, HashMap<String, String>)>,
    cache_server: String,
}

impl LiquidCacheClientBuilder {
    /// Create a new builder for LiquidCache client state.
    pub fn new(cache_server: impl AsRef<str>) -> Self {
        Self {
            object_stores: vec![],
            cache_server: cache_server.as_ref().to_string(),
        }
    }

    /// Add an object store to the builder.
    /// Checkout <https://docs.rs/object_store/latest/object_store/fn.parse_url_opts.html> for available options.
    pub fn with_object_store(
        mut self,
        url: ObjectStoreUrl,
        object_store_options: Option<HashMap<String, String>>,
    ) -> Self {
        self.object_stores
            .push((url, object_store_options.unwrap_or_default()));
        self
    }

    /// Build the [SessionContext].
    pub fn build(self, config: SessionConfig) -> Result<SessionContext> {
        let mut session_config = config;
        session_config
            .options_mut()
            .execution
            .parquet
            .pushdown_filters = true;
        session_config
            .options_mut()
            .execution
            .parquet
            .schema_force_view_types = false;
        session_config
            .options_mut()
            .execution
            .parquet
            .skip_arrow_metadata = false;
        session_config.options_mut().execution.parquet.skip_metadata = false;
        session_config.options_mut().execution.batch_size = ConfigNonZeroUsize::try_new(8192 * 2)?;
        // Dynamic filters (e.g. a hash join's runtime build-side filter) are pushed
        // into scan predicates by DataFusion. In distributed mode those scans are
        // serialized and executed on a remote server that can never receive the
        // join's runtime updates, and the serialized `DynamicFilterPhysicalExpr`
        // carries column indices relative to the join schema rather than the scan,
        // so the server fails to decode it. Disable the optimization on the client.
        // The master `enable_dynamic_filter_pushdown` only cascades to the
        // sub-options when set via the string API, so set each one explicitly.
        let optimizer_opts = &mut session_config.options_mut().optimizer;
        optimizer_opts.enable_dynamic_filter_pushdown = false;
        optimizer_opts.enable_join_dynamic_filter_pushdown = false;
        optimizer_opts.enable_topk_dynamic_filter_pushdown = false;
        optimizer_opts.enable_aggregate_dynamic_filter_pushdown = false;

        let runtime_env = Arc::new(RuntimeEnv::default());

        // Register object stores
        for (object_store_url, options) in &self.object_stores {
            let (object_store, _path) =
                object_store::parse_url_opts(object_store_url.as_ref(), options.clone())
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            runtime_env.register_object_store(object_store_url.as_ref(), Arc::new(object_store));
        }

        let session_state = SessionStateBuilder::new()
            .with_config(session_config)
            .with_runtime_env(runtime_env)
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(PushdownOptimizer::new(
                self.cache_server.clone(),
                self.object_stores.clone(),
            )))
            .build();
        let ctx = SessionContext::new_with_state(session_state);
        liquid_cache_datafusion::register_variant_functions(&ctx);
        Ok(ctx)
    }
}

pub(crate) fn to_df_err<E: Error + Send + Sync + 'static>(err: E) -> DataFusionError {
    DataFusionError::External(Box::new(err))
}

pub(crate) async fn flight_channel(
    source: impl Into<String>,
) -> Result<FastraceClientService<Channel>> {
    use fastrace_tonic::FastraceClientLayer;
    use tower::ServiceBuilder;

    // No tls here, to avoid the overhead of TLS
    // we assume both server and client are running on the trusted network.
    let endpoint = Channel::from_shared(source.into())
        .map_err(to_df_err)?
        .tcp_keepalive(Some(Duration::from_secs(10)));

    let channel = endpoint.connect().await.map_err(to_df_err)?;
    let channel = ServiceBuilder::new()
        .layer(FastraceClientLayer)
        .service(channel);
    Ok(channel)
}
