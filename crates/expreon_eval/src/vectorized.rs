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

/// A reusable pool of scratch buffers for evaluation of expressions, able to
/// serve any number of distinct batch sizes from the same pool — evaluating
/// training data of one size and test data of another needs only one stack.
///
/// - [`EvalBufferStack::acquire`] hands out an owned [`Buffer`] of a requested
///   length, pooled or freshly allocated.
/// - [`EvalBufferStack::reclaim`] returns a [`Buffer`] to the pool for reuse.
pub struct EvalBufferStack {
    batch: usize,
    free: Vec<Buffer>,
}

impl EvalBufferStack {
    /// Creates an empty stack, pre-sized as a hint for batches of `batch`
    /// elements — see [`Self::batch_size`]. The stack isn't restricted to
    /// that size: [`Self::acquire`] serves any length asked of it.
    pub const fn new(batch: usize) -> Self {
        Self {
            batch,
            free: Vec::new(),
        }
    }

    /// The batch size this stack was hinted to size for at construction.
    /// Purely informational — [`Self::acquire`] isn't restricted to it.
    pub const fn batch_size(&self) -> usize {
        self.batch
    }

    /// Number of buffers currently idle in the pool, of any length. Once
    /// every acquired buffer has been reclaimed, this converges to the peak
    /// number of buffers that were ever concurrently in use.
    pub fn len(&self) -> usize {
        self.free.len()
    }

    /// `true` if this stack has no idle buffers pooled right now.
    pub fn is_empty(&self) -> bool {
        self.free.is_empty()
    }

    /// Returns an owned buffer of length `len`, taken from the pool if one of
    /// that exact length is idle, or freshly allocated otherwise. The
    /// returned buffer isn't zeroed.
    pub fn acquire(&mut self, len: usize) -> Buffer {
        match self.free.iter().position(|b| b.len() == len) {
            Some(pos) => self.free.swap_remove(pos),
            None => Array1::zeros(len),
        }
    }

    /// Reclaims and returns a buffer to the pool, whatever its length.
    /// Subsequent calls to [`EvalBufferStack::acquire`] for that same length
    /// may reuse it.
    pub fn reclaim(&mut self, buf: Buffer) {
        self.free.push(buf);
    }
}

/// Vectorized evaluation context for expressions.
///
/// Mirrors [`crate::eval::EagerEvalContext`], but evaluates a whole node in one
/// dispatch (via [`crate::ops::OpMetadata::call_vectorized`]) instead of
/// once per batch element, and draws its intermediate buffers from a
/// caller-supplied [`EvalBufferStack`] instead of allocating a fresh array per
/// node.
pub struct VectorizedEvalContext<'a, 'b, Tag: Clone> {
    pub arena: &'a ExprArena<Tag>,
    pub ops: &'b OperationTable,
}

impl<'a, 'b, Tag: Clone> VectorizedEvalContext<'a, 'b, Tag> {
    pub const fn new(arena: &'a ExprArena<Tag>, ops: &'b OperationTable) -> Self {
        Self { arena, ops }
    }

    /// Evaluates the expression over a batch of inputs against one shared set
    /// of parameters, reusing scratch buffers from `stack`. Returns an owned
    /// result buffer; use [`EvalBufferStack::reclaim`] on it once done to
    /// return the allocation to the pool.
    ///
    /// ## Notes
    /// - `inputs` is expected to have shape `[batch_size, n_variables]`.
    /// - `parameters` holds one scalar per parameter slot — the same values
    ///   for every sample in the batch, as is the case for an individual's
    ///   constants in symbolic regression. There is no per-sample variant of
    ///   this method.
    pub fn eval_batch(
        &self,
        node_id: NodeId,
        inputs: ArrayView2<Scalar>,
        parameters: &[Scalar],
        stack: &mut EvalBufferStack,
    ) -> Buffer {
        let node = self
            .arena
            .get_node(node_id)
            .expect("node_id not present in arena");
        let batch = inputs.nrows();

        match node.kind {
            NodeKind::Variable(var_id) => {
                let idx = *var_id as usize;
                assert!(
                    idx < inputs.ncols(),
                    "variable {var_id} out of range for input of length {}",
                    inputs.ncols()
                );
                let mut out = stack.acquire(batch);
                out.assign(&inputs.column(idx));
                out
            }
            NodeKind::Parameter(param_id) => {
                let mut out = stack.acquire(batch);
                out.fill(parameters[*param_id as usize]);
                out
            }
            NodeKind::Unary { value, op } => {
                let val = self.eval_batch(value, inputs, parameters, stack);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                let mut out = stack.acquire(batch);
                meta.call_vectorized(&[val.view()], out.view_mut());
                stack.reclaim(val);
                out
            }
            NodeKind::Binary { left, right, op } => {
                let l = self.eval_batch(left, inputs, parameters, stack);
                let r = self.eval_batch(right, inputs, parameters, stack);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                let mut out = stack.acquire(batch);
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
        let result = ctx.eval_batch(v, inputs.view(), &[], &mut stack);

        assert_eq!(result, arr1(&[10.0, 20.0, 30.0]));
    }

    #[test]
    fn test_eval_batch_parameter_broadcasts_across_samples() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let p = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(3);

        let inputs = arr2(&[[], [], []]);
        let result = ctx.eval_batch(p, inputs.view(), &[5.0], &mut stack);

        assert_eq!(result, arr1(&[5.0, 5.0, 5.0]));
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
        let result = ctx.eval_batch(neg, inputs.view(), &[], &mut stack);

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

        // 3 samples: var=1,2,3 + the broadcast param (10) → 11,12,13
        let inputs = arr2(&[[1.0], [2.0], [3.0]]);
        let result = ctx.eval_batch(add, inputs.view(), &[10.0], &mut stack);

        assert_eq!(result, arr1(&[11.0, 12.0, 13.0]));
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
        let params = [10.0, 100.0];

        let scalar_ctx = EagerEvalContext::new(&arena, &ops);
        let expected = scalar_ctx.eval_batch(root, inputs.view(), &params);

        let vec_ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(4);
        let actual = vec_ctx.eval_batch(root, inputs.view(), &params, &mut stack);

        assert_eq!(actual, expected);
    }

    /// Cross-checks the vectorized path against the scalar
    /// [`EagerEvalContext::eval_batch`] on a parameter-free tree — the case
    /// the old `[batch, n_params]` layout needed a `.max(1)` column-count
    /// workaround for; an empty `parameters` slice needs no such thing.
    #[test]
    fn eval_batch_agrees_on_parameter_free_tree() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let var = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let root = arena.add(ExprNode::new_unary(var, OperationId::from(1), ()));

        let inputs = arr2(&[[1.0], [2.0], [3.0]]);

        let scalar_ctx = EagerEvalContext::new(&arena, &ops);
        let expected = scalar_ctx.eval_batch(root, inputs.view(), &[]);

        let vec_ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(3);
        let actual = vec_ctx.eval_batch(root, inputs.view(), &[], &mut stack);

        assert_eq!(actual, expected);
    }

    /// One stack must serve batches of different sizes correctly — e.g.
    /// training data and a differently-sized test set — without the caller
    /// needing a second stack. A buffer reclaimed at one length is only ever
    /// handed back out at that same length.
    #[test]
    fn one_stack_serves_multiple_batch_sizes() {
        let mut stack = EvalBufferStack::new(3);

        let a = stack.acquire(3);
        let b = stack.acquire(5);
        assert_eq!(a.len(), 3);
        assert_eq!(b.len(), 5);
        stack.reclaim(a);
        stack.reclaim(b);

        // Both a length-3 and a length-5 buffer are idle in the pool at once.
        assert_eq!(stack.len(), 2);

        // Each is served back out at its own length, not the other's.
        assert_eq!(stack.acquire(3).len(), 3);
        assert_eq!(stack.acquire(5).len(), 5);
        // Both were taken from the pool (now empty), not freshly allocated.
        assert_eq!(stack.len(), 0);
    }

    /// A vectorized evaluation still produces correct results when run
    /// through a stack that has already served a different batch size.
    #[test]
    fn eval_batch_correct_after_stack_served_another_batch_size() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();

        let var = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let param = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let add = arena.add(ExprNode::new_binary(var, param, OperationId::from(0), ()));

        let ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut stack = EvalBufferStack::new(3);

        let inputs3 = arr2(&[[1.0], [2.0], [3.0]]);
        let result3 = ctx.eval_batch(add, inputs3.view(), &[10.0], &mut stack);
        assert_eq!(result3, arr1(&[11.0, 12.0, 13.0]));
        stack.reclaim(result3);

        let inputs5 = arr2(&[[1.0], [2.0], [3.0], [4.0], [5.0]]);
        let result5 = ctx.eval_batch(add, inputs5.view(), &[10.0], &mut stack);
        assert_eq!(result5, arr1(&[11.0, 12.0, 13.0, 14.0, 15.0]));
        stack.reclaim(result5);
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
            let result = ctx.eval_batch(add, inputs.view(), &[100.0], &mut stack);
            assert_eq!(result, arr1(&[i as Scalar + 100.0, i as Scalar + 101.0]));
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
