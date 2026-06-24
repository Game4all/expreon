use rand::Rng;

use crate::{
    ast::NodeKind,
    types::{NodeId, Scalar, VariableId},
};

use super::{Genome, mutation::MutationContext};

/// Configuration for the random subtree generator.
pub struct GrowSubtreeConfig {
    /// Maximum tree depth (depth 0 = terminal only).
    pub max_depth: usize,
    /// Probability of emitting a terminal at a non-zero depth.
    pub p_terminal: f32,
    /// Number of input variables available.
    pub n_variables: u16,
    /// Range [lo, hi) for randomly-initialized constant parameters.
    pub const_range: (Scalar, Scalar),
}

/// Recursively generates a random subtree into `ctx.dest`, returning its root `NodeId`.
///
/// At `depth == 0` or with probability `p_terminal`, a terminal (variable or
/// constant parameter) is emitted. Otherwise a unary or binary operator is chosen
/// and its children are generated recursively at `depth - 1`.
pub fn gen_subtree<G: Genome>(
    ctx: &mut MutationContext<'_, G>,
    cfg: &GrowSubtreeConfig,
    depth: usize,
) -> NodeId {
    let should_be_terminal = depth == 0 || ctx.rng.random::<f32>() < cfg.p_terminal;

    if should_be_terminal {
        emit_terminal(ctx, cfg)
    } else {
        emit_operator(ctx, cfg, depth)
    }
}

fn emit_terminal<G: Genome>(ctx: &mut MutationContext<'_, G>, cfg: &GrowSubtreeConfig) -> NodeId {
    // 50/50 between a variable and a new constant parameter.
    if cfg.n_variables > 0 && ctx.rng.random::<bool>() {
        let var = VariableId::from(ctx.rng.random_range(0..cfg.n_variables));
        ctx.emit(NodeKind::Variable(var))
    } else {
        let (lo, hi) = cfg.const_range;
        let value: Scalar = ctx.rng.random_range(lo..hi);
        let param_id = ctx.new_parameter(value);
        ctx.emit(NodeKind::Parameter(param_id))
    }
}

fn emit_operator<G: Genome>(
    ctx: &mut MutationContext<'_, G>,
    cfg: &GrowSubtreeConfig,
    depth: usize,
) -> NodeId {
    let has_unary = ctx.ops.iter_unary_ops().len() > 0;
    let has_binary = ctx.ops.iter_binary_ops().len() > 0;

    // Prefer binary if both are available (tends to generate interesting trees).
    let use_binary = match (has_unary, has_binary) {
        (true, true) => ctx.rng.random::<bool>(),
        (false, true) => true,
        (true, false) => false,
        (false, false) => return emit_terminal(ctx, cfg), // fallback
    };

    if use_binary {
        let op = ctx.random_binary_op();
        let left = gen_subtree(ctx, cfg, depth - 1);
        let right = gen_subtree(ctx, cfg, depth - 1);
        ctx.emit(NodeKind::Binary { left, right, op })
    } else {
        let op = ctx.random_unary_op();
        let value = gen_subtree(ctx, cfg, depth - 1);
        ctx.emit(NodeKind::Unary { value, op })
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
            subtree::{GrowSubtreeConfig, gen_subtree},
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
            p_terminal: 0.3,
            n_variables: 2,
            const_range: (-1.0, 1.0),
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
            p_terminal: 0.4,
            n_variables: 2,
            const_range: (-1.0, 1.0),
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
