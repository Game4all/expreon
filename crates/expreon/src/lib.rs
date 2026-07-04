pub mod gp;

pub mod ops {
    pub use expreon_eval::ops::{
        Arity, Operation, OperationSet, OperationTable, OperationTableBuilder, builtin,
    };
}

pub mod prelude {
    pub use expreon_ast::{
        ExprArena, ExprNode, ExprNodeIter, NodeId, NodeKind, OperationId, ParameterId, RootId,
        Scalar, VariableId,
    };
    pub use expreon_eval::{
        eval::ExprEvalContext,
        ops::{Arity, Operation, OperationSet, OperationTable, OperationTableBuilder},
    };
}
