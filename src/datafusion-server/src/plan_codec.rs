//! Restore variant return fields that DataFusion's scalar-UDF protobuf omits.

use std::sync::Arc;

use arrow::datatypes::Schema;
use datafusion::{
    error::Result,
    execution::TaskContext,
    physical_expr::{PhysicalExpr, ScalarFunctionExpr},
    physical_plan::ExecutionPlan,
};
use datafusion_proto::{
    bytes::physical_plan_from_bytes_with_proto_converter,
    physical_plan::{
        DefaultPhysicalExtensionCodec, DefaultPhysicalProtoConverter, PhysicalExtensionCodec,
        PhysicalPlanDecodeContext, PhysicalProtoConverterExtension,
    },
    protobuf,
};
use liquid_cache_datafusion::VariantGetUdf;

pub(crate) fn decode_plan(bytes: &[u8], ctx: &TaskContext) -> Result<Arc<dyn ExecutionPlan>> {
    physical_plan_from_bytes_with_proto_converter(
        bytes,
        ctx,
        &DefaultPhysicalExtensionCodec {},
        &VariantFieldConverter,
    )
}

struct VariantFieldConverter;

impl PhysicalProtoConverterExtension for VariantFieldConverter {
    fn proto_to_execution_plan(
        &self,
        proto: &protobuf::PhysicalPlanNode,
        ctx: &PhysicalPlanDecodeContext<'_>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.default_proto_to_execution_plan(proto, ctx)
    }

    fn proto_to_physical_expr(
        &self,
        proto: &protobuf::PhysicalExprNode,
        input_schema: &Schema,
        ctx: &PhysicalPlanDecodeContext<'_>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let expr = self.default_proto_to_physical_expr(proto, input_schema, ctx)?;
        let Some(function) = ScalarFunctionExpr::try_downcast_func::<VariantGetUdf>(expr.as_ref())
        else {
            return Ok(expr);
        };
        // The wire format carries the return data type, but not its extension metadata.
        // Re-infer the field before a parent expression uses it, including nested UDFs.
        Ok(Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(function.fun().clone()),
                function.args().to_vec(),
                input_schema,
                ctx.task_ctx().session_config().options().clone(),
            )?
            .with_nullable(function.nullable()),
        ))
    }

    fn execution_plan_to_proto(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        codec: &dyn PhysicalExtensionCodec,
    ) -> Result<protobuf::PhysicalPlanNode> {
        DefaultPhysicalProtoConverter {}.execution_plan_to_proto(plan, codec)
    }

    fn physical_expr_to_proto(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        codec: &dyn PhysicalExtensionCodec,
    ) -> Result<protobuf::PhysicalExprNode> {
        DefaultPhysicalProtoConverter {}.physical_expr_to_proto(expr, codec)
    }
}
