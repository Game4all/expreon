use ndarray::{Array1, ArrayView2};

use crate::ast::{ExprArena, NodeKind};
use crate::ops::OperationTable;
use crate::types::{NodeId, Scalar};

/// An owned, batch-sized scratch buffer leased from an [`EvalBufferStack`].
///
/// Give it back with [`EvalBufferStack::reclaim`] once you're done with it so
/// it can be reused, otherwise it's simply dropped like any other value
/// and vectorized evaluations may allocate memory again.
pub type Buffer = Array1<Scalar>;

/// A reusable pool of batch-sized scratch buffers for evaluation of expressions.
///
/// - [`EvalBufferStack::acquire`] hands out an owned [`Buffer`], pooled or freshly allocated.
/// - [`EvalBufferStack::reclaim`] returns a [`Buffer`] to the pool for reuse.
pub struct EvalBufferStack {
    batch: usize,
    free: Vec<Buffer>,
}

impl EvalBufferStack {
    /// Creates an empty stack sized for batches of `batch` elements.
    pub const fn new(batch: usize) -> Self {
        Self {
            batch,
            free: Vec::new(),
        }
    }

    /// The batch size every buffer in this stack is sized for.
    pub const fn batch_size(&self) -> usize {
        self.batch
    }

    /// Number of buffers currently idle in the pool. Once every acquired
    /// buffer has been reclaimed, this converges to the peak number of
    /// buffers that were ever concurrently in use.
    pub fn len(&self) -> usize {
        self.free.len()
    }

    /// `true` if this stack has no idle buffers pooled right now.
    pub fn is_empty(&self) -> bool {
        self.free.is_empty()
    }

    /// Returns an owned buffer, taken from the pool or freshly allocated if
    /// none is idle. The returned buffer isn't zeroed
    pub fn acquire(&mut self) -> Buffer {
        self.free.pop().unwrap_or_else(|| Array1::zeros(self.batch))
    }

    /// Reclaims and returns a buffer to the pool.
    /// Subsequent calls to [`EvalBufferStack::acquire()`] may reuse the returned buffer
    pub fn reclaim(&mut self, buf: Buffer) {
        assert_eq!(
            buf.len(),
            self.batch,
            "reclaimed buffer of len {} into a stack sized for batches of {}",
            buf.len(),
            self.batch
        );
        self.free.push(buf);
    }
}

/// Vectorized evaluation context for expressions.
///
/// Mirrors [`crate::eval::EagerEvalContext`], but evaluates a whole node in one
/// dispatch (via [`crate::ops::Operation::forward_vectorized`]) instead of
/// once per batch element, and draws its intermediate buffers from a
/// caller-supplied [`BufferStack`] instead of allocating a fresh array per
/// node.
pub struct VectorizedEvalContext<'a, 'b, Tag: Clone> {
    pub arena: &'a ExprArena<Tag>,
    pub ops: &'b OperationTable,
}

impl<'a, 'b, Tag: Clone> VectorizedEvalContext<'a, 'b, Tag> {
    pub const fn new(arena: &'a ExprArena<Tag>, ops: &'b OperationTable) -> Self {
        Self { arena, ops }
    }

    /// Evaluates the expression over a batch of inputs and parameters,
    /// reusing scratch buffers from `stack`. Returns an owned
    /// result buffer; Use [`EvalBufferStack::reclaim`] on it once done to return the allocation to the pool
    ///
    /// ## Notes
    /// - `inputs` is expected to have shape `[batch_size, n_variables]`
    /// - `parameters` is expected to have shape `[batch_size, n_parameters]`.
    pub fn eval_batch(
        &self,
        node_id: NodeId,
        inputs: ArrayView2<Scalar>,
        parameters: ArrayView2<Scalar>,
        stack: &mut EvalBufferStack,
    ) -> Buffer {
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
                let mut out = stack.acquire();
                out.assign(&inputs.column(idx));
                out
            }
            NodeKind::Parameter(param_id) => {
                let mut out = stack.acquire();
                out.assign(&parameters.column(*param_id as usize));
                out
            }
            NodeKind::Unary { value, op } => {
                let val = self.eval_batch(value, inputs, parameters, stack);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                let mut out = stack.acquire();
                meta.call_vectorized(&[val.view()], out.view_mut());
                stack.reclaim(val);
                out
            }
            NodeKind::Binary { left, right, op } => {
                let l = self.eval_batch(left, inputs, parameters, stack);
                let r = self.eval_batch(right, inputs, parameters, stack);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                let mut out = stack.acquire();
                meta.call_vectorized(&[l.view(), r.view()], out.view_mut());
                stack.reclaim(l);
                stack.reclaim(r);
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};

    use crate::ast::{ExprArena, ExprNode};
    use crate::eval::EagerEvalContext;
    use crate::ops::{Arity, Operation, OperationTableBuilder};
    use crate::types::{OperationId, ParameterId, Scalar, VariableId};
    use crate::vectorized::{EvalBufferStack, VectorizedEvalContext};

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
    fn test_eval_batch_variable() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let v = arena.add(ExprNode::new_variable(VariableId::from(1), ()));
        let ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(3);

        let inputs = arr2(&[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]);
        let params = arr2(&[[], [], []]);
        let result = ctx.eval_batch(v, inputs.view(), params.view(), &mut stack);

        assert_eq!(result, arr1(&[10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_eval_batch_parameter() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let p = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(3);

        let inputs = arr2(&[[], [], []]);
        let params = arr2(&[[5.0], [6.0], [7.0]]);
        let result = ctx.eval_batch(p, inputs.view(), params.view(), &mut stack);

        assert_eq!(result, arr1(&[5.0, 6.0, 7.0]));
    }

    #[test]
    fn test_eval_batch_unary_neg() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let v = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let neg = arena.add(ExprNode::new_unary(v, OperationId::from(1), ()));

        let ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(3);

        let inputs = arr2(&[[1.0], [2.0], [3.0]]);
        let params = arr2(&[[], [], []]);
        let result = ctx.eval_batch(neg, inputs.view(), params.view(), &mut stack);

        assert_eq!(result, arr1(&[-1.0, -2.0, -3.0]));
    }

    #[test]
    fn test_eval_batch_binary_add() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let var = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let param = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let add = arena.add(ExprNode::new_binary(var, param, OperationId::from(0), ()));

        let ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(3);

        let inputs = arr2(&[[1.0], [2.0], [3.0]]);
        let params = arr2(&[[10.0], [20.0], [30.0]]);
        let result = ctx.eval_batch(add, inputs.view(), params.view(), &mut stack);

        assert_eq!(result, arr1(&[11.0, 22.0, 33.0]));
    }

    /// Cross-checks the vectorized path against the existing scalar
    /// [`EagerEvalContext::eval_batch`] on a deeper nested expression:
    /// `neg(var0 + param0) + param1`.
    #[test]
    fn vectorized_matches_scalar_eval_batch_on_nested_expr() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let var = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1), ()));
        let add1 = arena.add(ExprNode::new_binary(var, p0, OperationId::from(0), ()));
        let neg = arena.add(ExprNode::new_unary(add1, OperationId::from(1), ()));
        let root = arena.add(ExprNode::new_binary(neg, p1, OperationId::from(0), ()));

        let inputs = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
        let params = arr2(&[[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]]);

        let scalar_ctx = EagerEvalContext::new(&arena, &ops);
        let expected = scalar_ctx.eval_batch(root, inputs.view(), params.view());

        let vec_ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(4);
        let actual = vec_ctx.eval_batch(root, inputs.view(), params.view(), &mut stack);

        assert_eq!(actual, expected);
    }

    /// Running several evaluations through the same stack, reclaiming each
    /// result buffer before the next call, must keep producing correct
    /// results (proving recycled buffers are fully overwritten, not stale)
    /// and must not keep growing the number of allocated buffers.
    #[test]
    fn buffer_reuse_is_correct_and_bounded() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let var = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let param = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let add = arena.add(ExprNode::new_binary(var, param, OperationId::from(0), ()));

        let ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(2);

        for i in 0..10 {
            let inputs = arr2(&[[i as Scalar], [i as Scalar + 1.0]]);
            let params = arr2(&[[100.0], [200.0]]);
            let result = ctx.eval_batch(add, inputs.view(), params.view(), &mut stack);
            assert_eq!(result, arr1(&[i as Scalar + 100.0, i as Scalar + 201.0]));
            stack.reclaim(result);
        }

        // Two leaf reads + one add output = at most 3 live buffers per call;
        // the pool must stabilize instead of growing every iteration.
        assert!(
            stack.len() <= 3,
            "expected bounded pool, got {}",
            stack.len()
        );
    }
}
