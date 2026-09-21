use std::collections::HashMap;

use ndarray::ArrayView2;

use crate::ast::{ExprArena, ExprNode, NodeKind};
use crate::ops::OperationTable;
use crate::types::{NodeId, Scalar};
use crate::vectorized::{Buffer, EvalBufferStack};

/// Gates which nodes batch results a [`CachedEvalContext`] should keep in an [`EvalCache`] once computed.
pub trait CachePolicy<Tag: Clone> {
    /// Whether `node`'s batch result should be cached under `node_id` once evaluated.
    fn should_cache(&self, node_id: NodeId, node: &ExprNode<Tag>) -> bool;
}

impl<Tag: Clone, F> CachePolicy<Tag> for F
where
    F: Fn(NodeId, &ExprNode<Tag>) -> bool,
{
    fn should_cache(&self, node_id: NodeId, node: &ExprNode<Tag>) -> bool {
        self(node_id, node)
    }
}

/// Holds batch results for nodes a [`CachePolicy`] chose to cache, keyed by [`NodeId`], for reuse across repeated evaluations of the same expression.
/// This cache is meant to enable fast evaluations of the same expression for e.g weight fitting.
///
/// ## Validity
///
/// A cached buffer is only valid for the exact `(inputs, parameters)` pair it
/// was computed under. [`CachedEvalContext::eval_batch`] detects and clears
/// the cache when the batch size (`inputs.nrows()`) changes, but it has no
/// way to detect a change in `parameters` or in the input *values* at a fixed
/// shape, that is the caller's responsibility.
/// Call [`Self::clear`] whenever parameters change and cached results must be recomputed under the new values.
pub struct EvalCache {
    entries: HashMap<NodeId, Buffer>,
    batch: usize,
}

impl EvalCache {
    pub fn new(batch: usize) -> Self {
        Self {
            entries: HashMap::new(),
            batch,
        }
    }

    /// The batch size this cache's entries are valid for.
    pub const fn batch_size(&self) -> usize {
        self.batch
    }

    /// Looks up a cached result by node id.
    pub fn get(&self, node_id: NodeId) -> Option<&Buffer> {
        self.entries.get(&node_id)
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if nothing is cached right now.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Discards every cached entry. Call this whenever `parameters` (or input values at a fixed shape) change.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Vectorized evaluation context that memoizes selected nodes' batch results  in a caller-supplied [`EvalCache`], gated by a [`CachePolicy`].
pub struct CachedEvalContext<'a, 'b, Tag: Clone, P: CachePolicy<Tag>> {
    pub arena: &'a ExprArena<Tag>,
    pub ops: &'b OperationTable,
    policy: P,
}

impl<'a, 'b, Tag: Clone, P: CachePolicy<Tag>> CachedEvalContext<'a, 'b, Tag, P> {
    pub const fn new(arena: &'a ExprArena<Tag>, ops: &'b OperationTable, policy: P) -> Self {
        Self { arena, ops, policy }
    }

    /// Evaluates the expression over a batch of inputs against one shared set
    /// of parameters, reusing scratch buffers from `stack` and memoized
    /// results from `cache`. Returns an owned result buffer; use
    /// [`EvalBufferStack::reclaim`] on it once done to return the allocation
    /// to the pool.
    ///
    /// ## Cache invalidation
    ///
    /// If `inputs.nrows()` != `cache.batch_size()`, the cache is cleared first. See [`EvalCache`] for what else invalidates a cached entry.
    ///
    /// ## Notes
    /// - `inputs` is expected to have shape `[batch_size, n_variables]`.
    /// - `parameters` holds one scalar per parameter slot, shared by every
    ///   sample in the batch.
    pub fn eval_batch(
        &self,
        node_id: NodeId,
        inputs: ArrayView2<Scalar>,
        parameters: &[Scalar],
        cache: &mut EvalCache,
        stack: &mut EvalBufferStack,
    ) -> Buffer {
        let batch = inputs.nrows();
        if cache.batch != batch {
            cache.clear();
            cache.batch = batch;
        }
        self.eval_batch_inner(node_id, inputs, parameters, cache, stack)
    }

    fn eval_batch_inner(
        &self,
        node_id: NodeId,
        inputs: ArrayView2<Scalar>,
        parameters: &[Scalar],
        cache: &mut EvalCache,
        stack: &mut EvalBufferStack,
    ) -> Buffer {
        if let Some(cached) = cache.entries.get(&node_id) {
            let mut out = stack.acquire(cached.len());
            out.assign(cached);
            return out;
        }

        let node = self
            .arena
            .get_node(node_id)
            .expect("node_id not present in arena");
        let batch = inputs.nrows();

        let result = match node.kind {
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
                let val = self.eval_batch_inner(value, inputs, parameters, cache, stack);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                let mut out = stack.acquire(batch);
                meta.call_vectorized(&[val.view()], out.view_mut());
                stack.reclaim(val);
                out
            }
            NodeKind::Binary { left, right, op } => {
                let l = self.eval_batch_inner(left, inputs, parameters, cache, stack);
                let r = self.eval_batch_inner(right, inputs, parameters, cache, stack);
                let meta = self.ops.lookup_by_id(op).expect("op not found");
                let mut out = stack.acquire(batch);
                meta.call_vectorized(&[l.view(), r.view()], out.view_mut());
                stack.reclaim(l);
                stack.reclaim(r);
                out
            }
        };

        if self.policy.should_cache(node_id, node) {
            cache.entries.insert(node_id, result.clone());
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use ndarray::{arr2, array};

    use crate::ast::{ExprArena, ExprNode};
    use crate::cached::{CachedEvalContext, EvalCache};
    use crate::ops::{Arity, Operation, OperationTableBuilder};
    use crate::types::{NodeId, OperationId, ParameterId, Scalar, VariableId};
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

    /// `neg(var0 + param0) + param1`, matching the nested-expr fixture used by
    /// `vectorized.rs`'s own differential test.
    fn build_nested(arena: &mut ExprArena<()>) -> crate::types::NodeId {
        let var = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1), ()));
        let add1 = arena.add(ExprNode::new_binary(var, p0, OperationId::from(0), ()));
        let neg = arena.add(ExprNode::new_unary(add1, OperationId::from(1), ()));
        arena.add(ExprNode::new_binary(neg, p1, OperationId::from(0), ()))
    }

    #[test]
    fn cached_matches_vectorized_with_cache_everything_policy() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();
        let root = build_nested(&mut arena);

        let inputs = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
        let params = [10.0, 100.0];

        let vec_ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut vec_stack = EvalBufferStack::new(4);
        let expected = vec_ctx.eval_batch(root, inputs.view(), &params, &mut vec_stack);

        let cached_ctx = CachedEvalContext::new(&arena, &ops, |_: NodeId, _: &ExprNode<()>| true);
        let mut cache = EvalCache::new(4);
        let mut stack = EvalBufferStack::new(4);
        let actual = cached_ctx.eval_batch(root, inputs.view(), &params, &mut cache, &mut stack);

        assert_eq!(actual, expected);
        // Every node in the tree should have been cached.
        assert_eq!(cache.len(), arena.node_count(root));
    }

    #[test]
    fn cached_matches_vectorized_with_cache_nothing_policy() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();
        let root = build_nested(&mut arena);

        let inputs = arr2(&[[1.0], [2.0], [3.0], [4.0]]);
        let params = [10.0, 100.0];

        let vec_ctx = VectorizedEvalContext::new(&arena, &ops);
        let mut vec_stack = EvalBufferStack::new(4);
        let expected = vec_ctx.eval_batch(root, inputs.view(), &params, &mut vec_stack);

        let cached_ctx = CachedEvalContext::new(&arena, &ops, |_: NodeId, _: &ExprNode<()>| false);
        let mut cache = EvalCache::new(4);
        let mut stack = EvalBufferStack::new(4);
        let actual = cached_ctx.eval_batch(root, inputs.view(), &params, &mut cache, &mut stack);

        assert_eq!(actual, expected);
        assert!(cache.is_empty());
    }

    #[test]
    fn cached_subtree_is_evaluated_once_across_two_root_evaluations() {
        thread_local! {
            static CALLS: Cell<u32> = const { Cell::new(0) };
        }

        struct CountingNeg;
        impl Operation for CountingNeg {
            const NAME: &'static str = "counting_neg";
            const ID: &'static str = "counting_neg";
            const ARITY: Arity = Arity::Unary;
            fn forward(input: &[Scalar]) -> Scalar {
                -input[0]
            }
            fn vectorized_forward(
                inputs: &[ndarray::ArrayView1<Scalar>],
                mut out: ndarray::ArrayViewMut1<Scalar>,
            ) {
                CALLS.with(|c| c.set(c.get() + 1));
                out.assign(&inputs[0].mapv(|x| -x));
            }
        }

        let mut b = OperationTableBuilder::new();
        b.register::<Add>();
        b.register::<CountingNeg>();
        let ops = b.build();

        let mut arena: ExprArena<()> = ExprArena::new();
        let var = arena.add(ExprNode::new_variable(VariableId::from(0), ()));
        // The cached node: neg(var0).
        let neg = arena.add(ExprNode::new_unary(var, OperationId::from(1), ()));
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1), ()));
        let root1 = arena.add(ExprNode::new_binary(neg, p0, OperationId::from(0), ()));
        let root2 = arena.add(ExprNode::new_binary(neg, p1, OperationId::from(0), ()));

        let cached_ctx =
            CachedEvalContext::new(&arena, &ops, |id: NodeId, _: &ExprNode<()>| id == neg);
        let mut cache = EvalCache::new(3);
        let mut stack = EvalBufferStack::new(3);

        let inputs = arr2(&[[1.0], [2.0], [3.0]]);
        let r1 = cached_ctx.eval_batch(root1, inputs.view(), &[10.0, 20.0], &mut cache, &mut stack);
        assert_eq!(r1, array![9.0, 8.0, 7.0]);
        stack.reclaim(r1);

        let r2 = cached_ctx.eval_batch(root2, inputs.view(), &[10.0, 20.0], &mut cache, &mut stack);
        assert_eq!(r2, array![19.0, 18.0, 17.0]);
        stack.reclaim(r2);

        // `neg` was cached after the first evaluation, so the second root
        // evaluation must have reused it instead of calling the op again.
        assert_eq!(CALLS.with(|c| c.get()), 1);
    }

    #[test]
    fn stale_cache_returns_old_value_after_parameter_change() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));

        let cached_ctx = CachedEvalContext::new(&arena, &ops, |_: NodeId, _: &ExprNode<()>| true);
        let mut cache = EvalCache::new(2);
        let mut stack = EvalBufferStack::new(2);
        let inputs = arr2(&[[], []]);

        let r1 = cached_ctx.eval_batch(p0, inputs.view(), &[5.0], &mut cache, &mut stack);
        assert_eq!(r1, array![5.0, 5.0]);
        stack.reclaim(r1);

        // Parameter changed, but the cache is not told: it must keep serving
        // the stale value until explicitly cleared. This pins the documented
        // contract in `EvalCache`.
        let r2 = cached_ctx.eval_batch(p0, inputs.view(), &[99.0], &mut cache, &mut stack);
        assert_eq!(r2, array![5.0, 5.0]);
        stack.reclaim(r2);

        cache.clear();
        let r3 = cached_ctx.eval_batch(p0, inputs.view(), &[99.0], &mut cache, &mut stack);
        assert_eq!(r3, array![99.0, 99.0]);
        stack.reclaim(r3);
    }

    #[test]
    fn changing_batch_size_auto_clears_the_cache() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let ops = build_ops_test_table();
        let var = arena.add(ExprNode::new_variable(VariableId::from(0), ()));

        let cached_ctx = CachedEvalContext::new(&arena, &ops, |_: NodeId, _: &ExprNode<()>| true);
        let mut cache = EvalCache::new(3);
        let mut stack = EvalBufferStack::new(3);

        let inputs3 = arr2(&[[1.0], [2.0], [3.0]]);
        let r1 = cached_ctx.eval_batch(var, inputs3.view(), &[], &mut cache, &mut stack);
        assert_eq!(r1, array![1.0, 2.0, 3.0]);
        stack.reclaim(r1);
        assert_eq!(cache.len(), 1);

        let inputs5 = arr2(&[[1.0], [2.0], [3.0], [4.0], [5.0]]);
        let r2 = cached_ctx.eval_batch(var, inputs5.view(), &[], &mut cache, &mut stack);
        assert_eq!(r2, array![1.0, 2.0, 3.0, 4.0, 5.0]);
        stack.reclaim(r2);
        // Re-populated for the new batch size, not stale from size 3.
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.batch_size(), 5);
    }
}
