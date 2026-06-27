use rand::Rng;

use crate::{
    ast::{ExprNode, NodeKind},
    types::{NodeId, Scalar, VariableId},
};

use super::{Genome, builder::NodeBuilder};

/// Tuning parameters for the random tree generators, shared by the `grow` and
/// `full` methods. Depth is supplied separately at the call site.
pub struct TreeGenConfig {
    /// Probability of emitting a terminal at a non-zero depth (grow only).
    pub p_terminal: f32,
    /// Number of input variables available.
    pub n_variables: u16,
    /// Range [lo, hi) for randomly-initialized constant parameters.
    pub const_range: (Scalar, Scalar),
}

/// Configuration for the `grow`-method subtree generator used by mutation.
pub struct GrowSubtreeConfig {
    /// Maximum tree depth (depth 0 = terminal only).
    pub max_depth: usize,
    /// Generator tuning shared with population initialization.
    pub tuning: TreeGenConfig,
}

/// The tree-construction strategy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TreeMethod {
    /// Every branch grows to `depth`; terminals appear only at `depth == 0`.
    Full,
    /// Terminals may appear early (with probability `p_terminal`), so trees are
    /// irregular in shape and depth.
    Grow,
}

/// Recursively generates a random tree into the builder, returning its root
/// `NodeId`.
///
/// For [`TreeMethod::Full`], an operator is emitted at every level until
/// `depth == 0`. For [`TreeMethod::Grow`], a terminal is emitted at `depth == 0`
/// or, at any non-zero depth, with probability `cfg.p_terminal`. Otherwise a
/// unary or binary operator is chosen and its children are generated recursively
/// at `depth - 1`.
pub fn gen_tree<G: Genome, B: NodeBuilder<G>>(
    b: &mut B,
    cfg: &TreeGenConfig,
    method: TreeMethod,
    depth: usize,
) -> NodeId {
    let should_be_terminal =
        depth == 0 || (method == TreeMethod::Grow && b.rng().random::<f32>() < cfg.p_terminal);

    if should_be_terminal {
        emit_terminal(b, cfg)
    } else {
        emit_operator(b, cfg, method, depth)
    }
}

/// Generates a random subtree using the `grow` method. Retained as a thin
/// wrapper over [`gen_tree`] for mutation call sites.
pub fn gen_subtree<G: Genome, B: NodeBuilder<G>>(
    b: &mut B,
    cfg: &GrowSubtreeConfig,
    depth: usize,
) -> NodeId {
    gen_tree(b, &cfg.tuning, TreeMethod::Grow, depth)
}

fn emit_terminal<G: Genome, B: NodeBuilder<G>>(b: &mut B, cfg: &TreeGenConfig) -> NodeId {
    // 50/50 between a variable and a new constant parameter.
    if cfg.n_variables > 0 && b.rng().random::<bool>() {
        let var = VariableId::from(b.rng().random_range(0..cfg.n_variables));
        let kind = NodeKind::Variable(var);
        b.emit(ExprNode::new(kind, G::get_tag_for_node(kind)))
    } else {
        let (lo, hi) = cfg.const_range;
        let value: Scalar = b.rng().random_range(lo..hi);
        let param_id = b.new_parameter(value);
        let kind = NodeKind::Parameter(param_id);
        b.emit(ExprNode::new(kind, G::get_tag_for_node(kind)))
    }
}

fn emit_operator<G: Genome, B: NodeBuilder<G>>(
    b: &mut B,
    cfg: &TreeGenConfig,
    method: TreeMethod,
    depth: usize,
) -> NodeId {
    let has_unary = b.ops().iter_unary_ops().len() > 0;
    let has_binary = b.ops().iter_binary_ops().len() > 0;

    // Prefer binary if both are available (tends to generate interesting trees).
    let use_binary = match (has_unary, has_binary) {
        (true, true) => b.rng().random::<bool>(),
        (false, true) => true,
        (true, false) => false,
        (false, false) => return emit_terminal(b, cfg), // fallback
    };

    if use_binary {
        let op = b.pick_random_binary_op();
        let left = gen_tree(b, cfg, method, depth - 1);
        let right = gen_tree(b, cfg, method, depth - 1);
        let kind = NodeKind::Binary { left, right, op };
        b.emit(ExprNode::new(kind, G::get_tag_for_node(kind)))
    } else {
        let op = b.pick_random_unary_op();
        let value = gen_tree(b, cfg, method, depth - 1);
        let kind = NodeKind::Unary { value, op };
        b.emit(ExprNode::new(kind, G::get_tag_for_node(kind)))
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::{
        ast::ExprArena,
        gp::{
            mutation::MutationContext,
            subtree::{GrowSubtreeConfig, TreeGenConfig, gen_subtree},
            test_genome::TestSimpleGenome,
        },
        ops::{OperationTableBuilder, builtin::MathBaseOps},
        types::Scalar,
    };

    fn make_ops() -> crate::ops::OperationTable {
        let mut b = OperationTableBuilder::new();
        b.register_set::<MathBaseOps>();
        b.build()
    }

    #[test]
    fn grow_produces_valid_tree() {
        let ops = make_ops();

        let src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        let mut params: Vec<Scalar> = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);

        let cfg = GrowSubtreeConfig {
            max_depth: 4,
            tuning: TreeGenConfig {
                p_terminal: 0.3,
                n_variables: 2,
                const_range: (-1.0, 1.0),
            },
        };

        let mut ctx =
            MutationContext::<TestSimpleGenome>::new(&src, &ops, &mut rng, &mut dest, &mut params);
        let root_node = gen_subtree(&mut ctx, &cfg, cfg.max_depth);
        drop(ctx);

        // Root node must be valid.
        assert!(dest.get_node(root_node).is_some());
    }

    #[test]
    fn grow_is_deterministic() {
        let ops = make_ops();
        let src: ExprArena<()> = ExprArena::new();
        let cfg = GrowSubtreeConfig {
            max_depth: 3,
            tuning: TreeGenConfig {
                p_terminal: 0.4,
                n_variables: 2,
                const_range: (-1.0, 1.0),
            },
        };

        let root_a = {
            let mut dest: ExprArena<()> = ExprArena::new();
            let mut params: Vec<Scalar> = Vec::new();
            let mut rng = StdRng::seed_from_u64(7);
            let mut ctx = MutationContext::<TestSimpleGenome>::new(
                &src,
                &ops,
                &mut rng,
                &mut dest,
                &mut params,
            );
            gen_subtree(&mut ctx, &cfg, cfg.max_depth)
        };

        let root_b = {
            let mut dest: ExprArena<()> = ExprArena::new();
            let mut params: Vec<Scalar> = Vec::new();
            let mut rng = StdRng::seed_from_u64(7);
            let mut ctx = MutationContext::<TestSimpleGenome>::new(
                &src,
                &ops,
                &mut rng,
                &mut dest,
                &mut params,
            );
            gen_subtree(&mut ctx, &cfg, cfg.max_depth)
        };

        assert_eq!(root_a, root_b);
    }
}
