use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::{
    ast::{ExprNode, NodeKind},
    gp::{
        Genome,
        builder::NodeBuilder,
        subtree::{GrowSubtreeConfig, TreeGenConfig, emit_terminal, gen_subtree},
    },
    types::{NodeId, Scalar},
};

use super::{Mutation, MutationContext};

// ---------------------------------------------------------------------------
// PointMutation — swap operator, same arity
// ---------------------------------------------------------------------------

/// Replaces a node's operator with a randomly chosen one of the same arity,
/// leaving all children unchanged. Leaf nodes are swapped for another leaf
/// of the same kind (variable↔variable or parameter↔parameter).
pub struct PointMutation;

impl<G: Genome> Mutation<G> for PointMutation {
    fn applies_to(&self, kind: NodeKind) -> bool {
        matches!(kind, NodeKind::Unary { .. } | NodeKind::Binary { .. })
    }

    fn apply(
        &self,
        _target: NodeId,
        node: &ExprNode<G::Tag>,
        ctx: &mut MutationContext<'_, G>,
    ) -> Option<NodeId> {
        let new_root = match node.kind {
            NodeKind::Unary { value, .. } => {
                let v = ctx.copy_subtree(value);
                let op = ctx.pick_random_unary_op();
                let kind = NodeKind::Unary { value: v, op };
                ctx.emit(ExprNode::new(kind, G::get_tag_for_node(kind)))
            }
            NodeKind::Binary { left, right, .. } => {
                let l = ctx.copy_subtree(left);
                let r = ctx.copy_subtree(right);
                let op = ctx.pick_random_binary_op();
                let kind = NodeKind::Binary {
                    left: l,
                    right: r,
                    op,
                };
                ctx.emit(ExprNode::new(kind, G::get_tag_for_node(kind)))
            }
            _ => unreachable!("guarded by applies_to"),
        };
        Some(new_root)
    }
}

// ---------------------------------------------------------------------------
// SubtreeMutation — replace a target subtree with a freshly grown random tree
// ---------------------------------------------------------------------------

/// Replaces the target node (and its entire subtree) with a randomly generated
/// subtree produced by the `grow` generator.
pub struct SubtreeMutation {
    pub grow: GrowSubtreeConfig,
}

impl<G: Genome> Mutation<G> for SubtreeMutation {
    fn applies_to(&self, _kind: NodeKind) -> bool {
        true // any node can be replaced
    }

    fn apply(
        &self,
        _target: NodeId,
        _node: &ExprNode<G::Tag>,
        ctx: &mut MutationContext<'_, G>,
    ) -> Option<NodeId> {
        Some(gen_subtree(ctx, &self.grow, self.grow.max_depth))
    }
}

// ---------------------------------------------------------------------------
// ParamJitter — perturb a single constant/parameter value
// ---------------------------------------------------------------------------

/// Adds Gaussian noise (`Normal(0, stddev)`) to a single parameter node's
/// value. The tree structure is unchanged; only the offspring parameter vector
/// is modified.
pub struct ParamJitter {
    pub stddev: Scalar,
}

impl<G: Genome> Mutation<G> for ParamJitter {
    fn applies_to(&self, kind: NodeKind) -> bool {
        matches!(kind, NodeKind::Parameter(_))
    }

    fn apply(
        &self,
        _target: NodeId,
        node: &ExprNode<G::Tag>,
        ctx: &mut MutationContext<'_, G>,
    ) -> Option<NodeId> {
        let NodeKind::Parameter(param_id) = node.kind else {
            unreachable!("guarded by applies_to");
        };

        let current = ctx.get_parameter(param_id);
        let delta: Scalar = Normal::new(0.0f64, self.stddev as f64)
            .expect("invalid stddev for ParamJitter")
            .sample(ctx.rng) as Scalar;
        ctx.set_parameter(param_id, current + delta);

        // Passthrough: only the parameter vector changed. The engine copies the
        // original parameter node verbatim.
        None
    }
}

// ---------------------------------------------------------------------------
// HoistMutation — replace a node with one of its own subtrees (shrink)
// ---------------------------------------------------------------------------

/// Replaces the target node with a randomly chosen descendant of its own
/// subtree, shrinking the tree. This is the structural counterpart to
/// [`SubtreeMutation`] and the primary tool for combating bloat.
pub struct HoistMutation;

impl<G: Genome> Mutation<G> for HoistMutation {
    fn applies_to(&self, kind: NodeKind) -> bool {
        // Only internal nodes have descendants to hoist.
        matches!(kind, NodeKind::Unary { .. } | NodeKind::Binary { .. })
    }

    fn apply(
        &self,
        target: NodeId,
        _node: &ExprNode<G::Tag>,
        ctx: &mut MutationContext<'_, G>,
    ) -> Option<NodeId> {
        // Collect the target's descendants (its subtree, excluding the target
        // itself) by walking the source arena.
        let descendants: Vec<NodeId> = ctx.source().walk_expr(target).skip(1).collect();
        if descendants.is_empty() {
            // Guarded by `applies_to`; fall back to a passthrough copy if reached.
            return None;
        }
        let idx = ctx.rng().random_range(0..descendants.len());
        let picked = descendants[idx];
        Some(ctx.copy_subtree(picked))
    }
}

// ---------------------------------------------------------------------------
// InsertMutation — wrap a subtree in a new operator node (grow by one level)
// ---------------------------------------------------------------------------

/// Wraps the target subtree in a freshly chosen operator. For a unary operator
/// the target becomes its operand; for a binary operator the target is placed on
/// a random side with a new random terminal as its sibling.
pub struct InsertMutation {
    /// Range [lo, hi) for any new constant terminal introduced as a sibling.
    pub const_range: (Scalar, Scalar),
    /// Probability in [0, 1] of wrapping in a binary operator (vs. a unary one)
    /// when both arities are registered. Ignored when only one arity exists.
    pub p_binary: f32,
}

impl<G: Genome> Mutation<G> for InsertMutation {
    fn applies_to(&self, _kind: NodeKind) -> bool {
        true // any node can be wrapped
    }

    fn apply(
        &self,
        target: NodeId,
        _node: &ExprNode<G::Tag>,
        ctx: &mut MutationContext<'_, G>,
    ) -> Option<NodeId> {
        let inner = ctx.copy_subtree(target);

        let has_unary = ctx.ops().iter_unary_ops().len() > 0;
        let has_binary = ctx.ops().iter_binary_ops().len() > 0;

        let use_binary = match (has_unary, has_binary) {
            (true, true) => ctx.rng().random::<f32>() < self.p_binary,
            (false, true) => true,
            (true, false) => false,
            (false, false) => return None, // no operators to wrap with
        };

        let kind = if use_binary {
            let op = ctx.pick_random_binary_op();
            let cfg = TreeGenConfig {
                p_terminal: 1.0,
                const_range: self.const_range,
            };
            let sibling = emit_terminal(ctx, &cfg);
            // Random side placement matters for non-commutative operators.
            if ctx.rng().random::<bool>() {
                NodeKind::Binary {
                    left: inner,
                    right: sibling,
                    op,
                }
            } else {
                NodeKind::Binary {
                    left: sibling,
                    right: inner,
                    op,
                }
            }
        } else {
            let op = ctx.pick_random_unary_op();
            NodeKind::Unary { value: inner, op }
        };

        Some(ctx.emit(ExprNode::new(kind, G::get_tag_for_node(kind))))
    }
}

// ---------------------------------------------------------------------------
// TerminalMutation — mutate a leaf (variable/constant)
// ---------------------------------------------------------------------------

/// Mutates a leaf node. A variable is swapped for another variable or replaced
/// by a fresh constant; a parameter is either re-initialised to a fresh random
/// value (structure unchanged) or replaced by a variable. When the genome has no
/// input variables (`INPUT_DIM == 0`), only constant behaviour is used.
pub struct TerminalMutation {
    /// Range [lo, hi) for fresh constant values.
    pub const_range: (Scalar, Scalar),
    /// Probability in [0, 1] of routing the leaf to a variable (vs. a constant).
    /// Forced to 0 when the genome has no input variables (`INPUT_DIM == 0`).
    pub p_variable: f32,
}

impl<G: Genome> Mutation<G> for TerminalMutation {
    fn applies_to(&self, kind: NodeKind) -> bool {
        matches!(kind, NodeKind::Variable(_) | NodeKind::Parameter(_))
    }

    fn apply(
        &self,
        _target: NodeId,
        node: &ExprNode<G::Tag>,
        ctx: &mut MutationContext<'_, G>,
    ) -> Option<NodeId> {
        // Whether to (re)route this leaf to a variable. Always false when the
        // genome has no inputs, so `pick_variable` is never called unsafely.
        let to_variable = G::INPUT_DIM > 0 && ctx.rng().random::<f32>() < self.p_variable;

        match node.kind {
            NodeKind::Variable(_) => {
                if to_variable {
                    let var = ctx.pick_variable();
                    let kind = NodeKind::Variable(var);
                    Some(ctx.emit(ExprNode::new(kind, G::get_tag_for_node(kind))))
                } else {
                    let (lo, hi) = self.const_range;
                    let value: Scalar = ctx.rng().random_range(lo..hi);
                    let pid = ctx.new_parameter(value);
                    let kind = NodeKind::Parameter(pid);
                    Some(ctx.emit(ExprNode::new(kind, G::get_tag_for_node(kind))))
                }
            }
            NodeKind::Parameter(pid) => {
                if to_variable {
                    let var = ctx.pick_variable();
                    let kind = NodeKind::Variable(var);
                    Some(ctx.emit(ExprNode::new(kind, G::get_tag_for_node(kind))))
                } else {
                    // Re-initialise in place: only the parameter vector changes.
                    let (lo, hi) = self.const_range;
                    let value: Scalar = ctx.rng().random_range(lo..hi);
                    ctx.set_parameter(pid, value);
                    None
                }
            }
            _ => unreachable!("guarded by applies_to"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::{
        ast::{ExprArena, ExprNode, NodeKind},
        gp::{
            Individual,
            mutation::{
                apply_mutation,
                builtin::{
                    HoistMutation, InsertMutation, ParamJitter, PointMutation, SubtreeMutation,
                    TerminalMutation,
                },
            },
            subtree::{GrowSubtreeConfig, TreeGenConfig},
            test_genome::TestSimpleGenome,
        },
        ops::{OperationTableBuilder, builtin::MathBaseOps},
        types::{OperationId, ParameterId, RootId, Scalar},
    };

    fn base_ops() -> crate::ops::OperationTable {
        let mut b = OperationTableBuilder::new();
        b.register_set::<MathBaseOps>();
        b.build()
    }

    fn build_two_param_tree(arena: &mut ExprArena<()>) -> (RootId, Vec<Scalar>) {
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1u16), ()));
        let add = arena.add(ExprNode::new_binary(p0, p1, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        (root, vec![1.0, 2.0])
    }

    // ---------------------------------------------------------------------------
    // PointMutation — keeps structure, changes operator
    // ---------------------------------------------------------------------------
    #[test]
    fn point_mutation_preserves_arity() {
        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();
        let (root, params) = build_two_param_tree(&mut src);

        let target = src.get_root(root).unwrap();
        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(1);

        let offspring = apply_mutation(
            &PointMutation,
            target,
            &parent,
            &src,
            &mut dest,
            &ops,
            &mut rng,
        )
        .unwrap();

        let parent_root = src.get_root(parent.root).unwrap();
        let offspring_root = dest.get_root(offspring.root).unwrap();
        let src_count = src.iter_expr_nodes(parent_root).count();
        let dest_count = dest.iter_expr_nodes(offspring_root).count();
        assert_eq!(src_count, dest_count);
    }

    // ---------------------------------------------------------------------------
    // ParamJitter — identical structure, one parameter shifted
    // ---------------------------------------------------------------------------
    #[test]
    fn param_jitter_changes_exactly_one_param() {
        let ops = base_ops();

        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        let (root, params) = build_two_param_tree(&mut src);
        let original_params = params.clone();

        let root_node = src.get_root(root).unwrap();
        let target = src
            .iter_expr_nodes(root_node)
            .find(|(_, n)| matches!(n.kind, NodeKind::Parameter(_)))
            .map(|(id, _)| id)
            .unwrap();

        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(2);

        let offspring = apply_mutation(
            &ParamJitter { stddev: 0.1 },
            target,
            &parent,
            &src,
            &mut dest,
            &ops,
            &mut rng,
        )
        .unwrap();

        let parent_root = src.get_root(parent.root).unwrap();
        let offspring_root = dest.get_root(offspring.root).unwrap();
        let src_count = src.iter_expr_nodes(parent_root).count();
        let dest_count = dest.iter_expr_nodes(offspring_root).count();
        assert_eq!(src_count, dest_count);

        let changed = offspring
            .parameters
            .iter()
            .zip(&original_params)
            .filter(|(a, b)| (*a - *b).abs() > 1e-9)
            .count();
        assert_eq!(changed, 1);
    }

    // ---------------------------------------------------------------------------
    // SubtreeMutation — offspring is valid and evaluates without panic
    // ---------------------------------------------------------------------------
    #[test]
    fn subtree_mutation_produces_valid_tree() {
        use crate::eval::EvalContext;
        use ndarray::array;

        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();
        let (root, params) = build_two_param_tree(&mut src);

        let target = src.get_root(root).unwrap();
        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(3);

        let cfg = GrowSubtreeConfig {
            max_depth: 3,
            tuning: TreeGenConfig {
                p_terminal: 0.3,
                const_range: (-1.0, 1.0),
            },
        };

        let offspring = apply_mutation(
            &SubtreeMutation { grow: cfg },
            target,
            &parent,
            &src,
            &mut dest,
            &ops,
            &mut rng,
        )
        .unwrap();

        let root_node = dest.get_root(offspring.root).unwrap();
        let eval_ctx = EvalContext::new(&dest, &ops);

        let inputs = array![[0.5f32, 1.0f32]];
        let n_params = offspring.parameters.len();
        let params_arr = ndarray::Array2::from_shape_vec((1, n_params.max(1)), {
            let mut v = offspring.parameters.clone();
            if n_params == 0 {
                v.push(0.0);
            }
            v
        })
        .unwrap();

        let _ = eval_ctx.eval_batch(root_node, inputs.view(), params_arr.view());
    }

    /// Evaluate an offspring once to confirm it is structurally valid (no panic).
    fn eval_ok(
        dest: &ExprArena<()>,
        ops: &crate::ops::OperationTable,
        offspring: &Individual<TestSimpleGenome>,
    ) {
        use crate::eval::EvalContext;
        use ndarray::array;

        let root_node = dest.get_root(offspring.root).unwrap();
        let eval_ctx = EvalContext::new(dest, ops);
        let inputs = array![[0.5f32, 1.0f32]];
        let n_params = offspring.parameters.len();
        let params_arr = ndarray::Array2::from_shape_vec((1, n_params.max(1)), {
            let mut v = offspring.parameters.clone();
            if n_params == 0 {
                v.push(0.0);
            }
            v
        })
        .unwrap();
        let _ = eval_ctx.eval_batch(root_node, inputs.view(), params_arr.view());
    }

    // -----------------------------------------------------------------------
    // HoistMutation — shrinks the tree to one of the target's own subtrees
    // -----------------------------------------------------------------------
    #[test]
    fn hoist_mutation_shrinks_tree() {
        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();
        let (root, params) = build_two_param_tree(&mut src);

        // The root binary node has descendants to hoist.
        let target = src.get_root(root).unwrap();
        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(5);

        let offspring =
            apply_mutation(&HoistMutation, target, &parent, &src, &mut dest, &ops, &mut rng)
                .unwrap();

        let parent_root = src.get_root(parent.root).unwrap();
        let offspring_root = dest.get_root(offspring.root).unwrap();
        let src_count = src.iter_expr_nodes(parent_root).count();
        let dest_count = dest.iter_expr_nodes(offspring_root).count();
        assert!(
            dest_count < src_count,
            "hoist should shrink the tree ({dest_count} !< {src_count})"
        );
        eval_ok(&dest, &ops, &offspring);
    }

    #[test]
    fn hoist_mutation_is_deterministic() {
        let ops = base_ops();
        let run = |seed: u64| {
            let mut src: ExprArena<()> = ExprArena::new();
            let mut dest: ExprArena<()> = ExprArena::new();
            let (root, params) = build_two_param_tree(&mut src);
            let target = src.get_root(root).unwrap();
            let parent = Individual::<TestSimpleGenome>::new(root, params);
            let mut rng = StdRng::seed_from_u64(seed);
            let off =
                apply_mutation(&HoistMutation, target, &parent, &src, &mut dest, &ops, &mut rng)
                    .unwrap();
            let off_root = dest.get_root(off.root).unwrap();
            let kinds: Vec<NodeKind> =
                dest.iter_expr_nodes(off_root).map(|(_, n)| n.kind).collect();
            (kinds, off.parameters)
        };
        assert_eq!(run(5), run(5));
    }

    // -----------------------------------------------------------------------
    // InsertMutation — wraps the target, growing the tree
    // -----------------------------------------------------------------------
    #[test]
    fn insert_mutation_grows_tree() {
        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();
        let (root, params) = build_two_param_tree(&mut src);

        let target = src.get_root(root).unwrap();
        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(11);

        let offspring = apply_mutation(
            &InsertMutation {
                const_range: (-1.0, 1.0),
                p_binary: 0.5,
            },
            target,
            &parent,
            &src,
            &mut dest,
            &ops,
            &mut rng,
        )
        .unwrap();

        let parent_root = src.get_root(parent.root).unwrap();
        let offspring_root = dest.get_root(offspring.root).unwrap();
        let src_count = src.iter_expr_nodes(parent_root).count();
        let dest_count = dest.iter_expr_nodes(offspring_root).count();
        assert!(
            dest_count > src_count,
            "insert should grow the tree ({dest_count} !> {src_count})"
        );
        eval_ok(&dest, &ops, &offspring);
    }

    // -----------------------------------------------------------------------
    // TerminalMutation — replaces a leaf; node count is preserved
    // -----------------------------------------------------------------------
    #[test]
    fn terminal_mutation_preserves_node_count() {
        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();
        let (root, params) = build_two_param_tree(&mut src);

        let root_node = src.get_root(root).unwrap();
        let target = src
            .iter_expr_nodes(root_node)
            .find(|(_, n)| matches!(n.kind, NodeKind::Parameter(_)))
            .map(|(id, _)| id)
            .unwrap();

        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(13);

        let offspring = apply_mutation(
            &TerminalMutation {
                const_range: (-1.0, 1.0),
                p_variable: 0.5,
            },
            target,
            &parent,
            &src,
            &mut dest,
            &ops,
            &mut rng,
        )
        .unwrap();

        let parent_root = src.get_root(parent.root).unwrap();
        let offspring_root = dest.get_root(offspring.root).unwrap();
        let src_count = src.iter_expr_nodes(parent_root).count();
        let dest_count = dest.iter_expr_nodes(offspring_root).count();
        assert_eq!(
            src_count, dest_count,
            "terminal mutation must not change node count"
        );
        eval_ok(&dest, &ops, &offspring);
    }
}
