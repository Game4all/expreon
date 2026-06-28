use ndarray::{Array1, ArrayView1, ArrayView2, Zip};

use crate::ast::{ExprArena, NodeKind};
use crate::ops::OperationTable;
use crate::types::{NodeId, Scalar};

/// Evaluation context for expressions
pub struct EvalContext<'a, 'b, Tag: Clone> {
    pub arena: &'a ExprArena<Tag>,
    pub ops: &'b OperationTable,
}

impl<'a, 'b, Tag: Clone> EvalContext<'a, 'b, Tag> {
    pub const fn new(arena: &'a ExprArena<Tag>, ops: &'b OperationTable) -> Self {
        Self { arena, ops }
    }

    /// Evaluates the expression with the given node ID using provided parameters and inputs
    /// Returns a single output.
    pub fn eval(
        &self,
        node_id: NodeId,
        inputs: ArrayView1<Scalar>,
        parameters: ArrayView1<Scalar>,
    ) -> Scalar {
        let node = self
            .arena
            .get_node(node_id)
            .expect("node_id not present in arena");

        match node.kind {
            NodeKind::Variable(var_id) => {
                let idx = *var_id as usize;
                assert!(
                    idx < inputs.len(),
                    "variable {var_id} out of range for input of length {}",
                    inputs.len()
                );
                inputs[idx]
            }
            NodeKind::Parameter(param_id) => parameters[*param_id as usize],
            NodeKind::Unary { value, op } => {
                let val = self.eval(value, inputs, parameters);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                meta.call(&[val])
            }
            NodeKind::Binary { left, right, op } => {
                let l = self.eval(left, inputs, parameters);
                let r = self.eval(right, inputs, parameters);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                meta.call(&[l, r])
            }
        }
    }

    /// Evaluates the expression over a batch of inputs and parameters.
    /// Returns one output per sample.
    ///
    /// ## Notes
    /// - `inputs` is expected to have shape `[batch_size, n_variables]`
    /// - `parameters` is expected to have shape `[batch_size, n_parameters]`.
    pub fn eval_batch(
        &self,
        node_id: NodeId,
        inputs: ArrayView2<Scalar>,
        parameters: ArrayView2<Scalar>,
    ) -> Array1<Scalar> {
        let node = self
            .arena
            .get_node(node_id)
            .expect("node_id not present in arena");

        match node.kind {
            NodeKind::Variable(var_id) => {
                let idx = *var_id as usize;
                assert!(
                    idx < inputs.ncols(),
                    "variable {var_id} out of range for input of length {}",
                    inputs.ncols()
                );
                inputs.column(idx).to_owned()
            }
            NodeKind::Parameter(param_id) => parameters.column(*param_id as usize).to_owned(),
            NodeKind::Unary { value, op } => {
                let val = self.eval_batch(value, inputs, parameters);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                val.mapv(|x| meta.call(&[x]))
            }
            NodeKind::Binary { left, right, op } => {
                let l = self.eval_batch(left, inputs, parameters);
                let r = self.eval_batch(right, inputs, parameters);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                Zip::from(&l)
                    .and(&r)
                    .map_collect(|&lv, &rv| meta.call(&[lv, rv]))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use crate::ast::{ExprArena, ExprNode};
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

        let p = arena.add(ExprNode::new_parameter(ParameterId::from(1), ()));
        let ctx = EvalContext::new(&arena, &ops);

        assert_eq!(
            ctx.eval(p, arr1(&[]).view(), arr1(&[10.0, 42.0]).view()),
            42.0
        );
    }

    #[test]
    fn test_eval_variable() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let v = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let ctx = EvalContext::new(&arena, &ops);
        assert_eq!(ctx.eval(v, arr1(&[7.0]).view(), arr1(&[]).view()), 7.0);
    }

    #[test]
    fn test_eval_binary_add() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let a = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let b = arena.add(ExprNode::new_parameter(ParameterId::from(1), ()));
        let add = arena.add(ExprNode::new_binary(a, b, OperationId::from(0), ()));

        let ctx = EvalContext::new(&arena, &ops);

        assert_eq!(
            ctx.eval(add, arr1(&[]).view(), arr1(&[3.0, 4.0]).view()),
            7.0
        );
    }

    #[test]
    fn test_eval_unary_neg() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let v = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let neg = arena.add(ExprNode::new_unary(v, OperationId::from(1), ()));

        let ops = build_ops_test_table();
        let ctx = EvalContext::new(&arena, &ops);
        assert_eq!(ctx.eval(neg, arr1(&[5.0]).view(), arr1(&[]).view()), -5.0);
    }

    #[test]
    fn test_eval_sum_five_variables() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let vars: Vec<_> = (0..5)
            .map(|i| arena.add(ExprNode::new_variable(VariableId::from(i), ())))
            .collect();

        let sum = vars[1..].iter().fold(vars[0], |acc, &v| {
            arena.add(ExprNode::new_binary(acc, v, OperationId::from(0), ()))
        });

        let ctx = EvalContext::new(&arena, &ops);
        assert_eq!(
            ctx.eval(
                sum,
                arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]).view(),
                arr1(&[]).view()
            ),
            15.0
        );
    }

    #[test]
    fn test_eval_batch_variable() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let v = arena.add(ExprNode::new_variable(VariableId::from(1), ()));
        let ctx = EvalContext::new(&arena, &ops);

        // 3 samples, 2 variables each; we read variable index 1
        let inputs = arr2(&[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]);
        let params = arr2(&[[], [], []]);
        let result = ctx.eval_batch(v, inputs.view(), params.view());

        assert_eq!(result, arr1(&[10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_eval_batch_parameter() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let p = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let ctx = EvalContext::new(&arena, &ops);

        // 3 samples, each with its own parameter value
        let inputs = arr2(&[[], [], []]);
        let params = arr2(&[[5.0], [6.0], [7.0]]);
        let result = ctx.eval_batch(p, inputs.view(), params.view());

        assert_eq!(result, arr1(&[5.0, 6.0, 7.0]));
    }

    #[test]
    fn test_eval_batch_binary_add() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let var = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let param = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let add = arena.add(ExprNode::new_binary(var, param, OperationId::from(0), ()));

        let ctx = EvalContext::new(&arena, &ops);

        // 3 samples: var=1,2,3 + param=10,20,30 → 11,22,33
        let inputs = arr2(&[[1.0], [2.0], [3.0]]);
        let params = arr2(&[[10.0], [20.0], [30.0]]);
        let result = ctx.eval_batch(add, inputs.view(), params.view());

        assert_eq!(result, arr1(&[11.0, 22.0, 33.0]));
    }
}
