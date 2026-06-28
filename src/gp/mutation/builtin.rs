use rand_distr::{Distribution, Normal};

use crate::{
    ast::{ExprNode, NodeKind},
    gp::{
        Genome,
        builder::NodeBuilder,
        subtree::{GrowSubtreeConfig, gen_subtree},
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
                builtin::{ParamJitter, PointMutation, SubtreeMutation},
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

        let src_count = src.iter_expr_nodes(parent.root).count();
        let dest_count = dest.iter_expr_nodes(offspring.root).count();
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

        let target = src
            .iter_expr_nodes(root)
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

        let src_count = src.iter_expr_nodes(parent.root).count();
        let dest_count = dest.iter_expr_nodes(offspring.root).count();
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
}
