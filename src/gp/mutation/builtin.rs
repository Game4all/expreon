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

    fn apply(&self, target: NodeId, ctx: &mut MutationContext<'_, G>) -> Option<NodeId> {
        let node = ctx
            .source
            .get_node(target)
            .expect("invalid target for PointMutation");
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

    fn apply(&self, _target: NodeId, ctx: &mut MutationContext<'_, G>) -> Option<NodeId> {
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

    fn apply(&self, target: NodeId, ctx: &mut MutationContext<'_, G>) -> Option<NodeId> {
        let node = ctx
            .source
            .get_node(target)
            .expect("invalid target for ParamJitter");

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
