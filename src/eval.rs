use crate::ast::{ExprArena, NodeKind};
use crate::ops::OperationTable;
use crate::types::{NodeId, Scalar};

/// Evaluation context for expressions
pub struct EvalContext<'a, 'b, Tag> {
    pub arena: &'a ExprArena<Tag>,
    pub ops: &'b OperationTable,
}

impl<'a, 'b, Tag> EvalContext<'a, 'b, Tag> {
    pub const fn new(arena: &'a ExprArena<Tag>, ops: &'b OperationTable) -> Self {
        Self { arena, ops }
    }

    /// Evaluates the expression with the given node ID using provided parameters and inputs
    pub fn eval(&self, node_id: NodeId, inputs: &[Scalar], parameters: &[Scalar]) -> Scalar {
        let node = self
            .arena
            .get_node(node_id)
            .expect("node_id not present in arena");

        match node.kind {
            NodeKind::Variable(var_id) => inputs[*var_id as usize],
            NodeKind::Parameter(param_id) => parameters[*param_id as usize],
            NodeKind::Unary { value, op } => {
                let val = self.eval(value, inputs, parameters);
                let meta = self.ops.get_meta_from_id(op).expect("op not found");
                meta.call(&[val])
            }
            NodeKind::Binary { left, right, op } => {
                let l = self.eval(left, inputs, parameters);
                let r = self.eval(right, inputs, parameters);
                let meta = self.ops.get_meta_from_id(op).expect("op not found");
                meta.call(&[l, r])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{ExprArena, ExprNode, NodeKind};
    use crate::eval::EvalContext;
    use crate::ops::{Arity, Operation, OperationTableBuilder};
    use crate::types::{OperationId, ParameterId, Scalar, VariableId};

    struct Add;
    impl Operation for Add {
        const NAME: &'static str = "add";
        const ID: &'static str = "add";
        const ARITY: Arity = Arity::Binary;
        fn forward(input: &[Scalar]) -> Scalar {
            input[0] + input[1]
        }
    }

    struct Neg;
    impl Operation for Neg {
        const NAME: &'static str = "neg";
        const ID: &'static str = "neg";
        const ARITY: Arity = Arity::Unary;
        fn forward(input: &[Scalar]) -> Scalar {
            -input[0]
        }
    }

    fn build_ops_test_table() -> crate::ops::OperationTable {
        let mut b = OperationTableBuilder::new();
        b.register::<Add>();
        b.register::<Neg>();
        b.build()
    }

    #[test]
    fn test_eval_parameter() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let p = arena.add(ExprNode::new(NodeKind::Parameter(ParameterId::from(1)), ()));
        let ctx = EvalContext::new(&arena, &ops);

        assert_eq!(ctx.eval(p, &[], &[10.0, 42.0]), 42.0);
    }

    #[test]
    fn test_eval_variable() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let v = arena.add(ExprNode::new(NodeKind::Variable(VariableId::from(0)), ()));
        let ctx = EvalContext::new(&arena, &ops);
        assert_eq!(ctx.eval(v, &[7.0], &[]), 7.0);
    }

    #[test]
    fn test_eval_binary_add() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let a = arena.add(ExprNode::new(NodeKind::Parameter(ParameterId::from(0)), ()));
        let b = arena.add(ExprNode::new(NodeKind::Parameter(ParameterId::from(1)), ()));
        let add = arena.add(ExprNode::new(
            NodeKind::Binary {
                left: a,
                right: b,
                op: OperationId::from(0),
            },
            (),
        ));

        let ctx = EvalContext::new(&arena, &ops);

        assert_eq!(ctx.eval(add, &[], &[3.0, 4.0]), 7.0);
    }

    #[test]
    fn test_eval_unary_neg() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let v = arena.add(ExprNode::new(NodeKind::Variable(VariableId::from(0)), ()));
        let neg = arena.add(ExprNode::new(
            NodeKind::Unary {
                value: v,
                op: OperationId::from(1),
            },
            (),
        ));
        let ops = build_ops_test_table();
        let ctx = EvalContext::new(&arena, &ops);
        assert_eq!(ctx.eval(neg, &[5.0], &[]), -5.0);
    }

    #[test]
    fn test_eval_sum_five_variables() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let vars: Vec<_> = (0..5)
            .map(|i| arena.add(ExprNode::new(NodeKind::Variable(VariableId::from(i)), ())))
            .collect();

        let sum = vars[1..].iter().fold(vars[0], |acc, &v| {
            arena.add(ExprNode::new(
                NodeKind::Binary {
                    left: acc,
                    right: v,
                    op: OperationId::from(0),
                },
                (),
            ))
        });

        let ctx = EvalContext::new(&arena, &ops);
        assert_eq!(ctx.eval(sum, &[1.0, 2.0, 3.0, 4.0, 5.0], &[]), 15.0);
    }
}
