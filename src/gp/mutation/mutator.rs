use rand::{Rng, RngCore};

use crate::{
    ast::{ExprArena, ExprNode, NodeKind},
    gp::{Genome, Individual},
    ops::OperationTable,
    types::NodeId,
};

use super::{Mutation, MutationContext};

fn copy_over_replacing<G: Genome + 'static>(
    src_node: NodeId,
    target: NodeId,
    mutation: &dyn Mutation<G>,
    ctx: &mut MutationContext<'_, G>,
) -> NodeId {
    if src_node == target {
        // A passthrough mutation returns `None`; copy the target subtree
        // verbatim (preserving tags) in that case.
        return mutation
            .apply(target, ctx)
            .unwrap_or_else(|| ctx.copy_subtree(target));
    }

    // Bind arena as a copy of the shared ref before borrowing `ctx` mutably below.
    let arena: &ExprArena<G::Tag> = ctx.source;
    let node = arena
        .get_node(src_node)
        .expect("invalid source node in rebuild");

    let kind = node.kind;
    let tag = node.tag.clone();

    let new_kind = match kind {
        NodeKind::Unary { value, op } => {
            let v = copy_over_replacing(value, target, mutation, ctx);
            NodeKind::Unary { value: v, op }
        }
        NodeKind::Binary { left, right, op } => {
            let l = copy_over_replacing(left, target, mutation, ctx);
            let r = copy_over_replacing(right, target, mutation, ctx);
            NodeKind::Binary {
                left: l,
                right: r,
                op,
            }
        }
        leaf => leaf,
    };

    // Preserve tag: this node is on the unchanged path.
    ctx.dest.add(ExprNode::new(new_kind, tag))
}

/// Weighted, pluggable registry of mutations.
///
/// On each call to `mutate`, one mutation is selected (weighted by its
/// registered weight, restricted to mutations that have at least one valid
/// target) and applied to a randomly chosen qualifying node.
pub struct Mutator<G: Genome> {
    entries: Vec<(f32, Box<dyn Mutation<G>>)>,
    total_weight: f32,
}

impl<G: Genome + 'static> Mutator<G> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_weight: 0.0,
        }
    }

    /// Register a mutation with the given selection weight.
    pub fn add(&mut self, weight: f32, m: impl Mutation<G> + 'static) -> &mut Self {
        assert!(weight > 0.0, "mutation weight must be positive");
        self.total_weight += weight;
        self.entries.push((weight, Box::new(m)));
        self
    }

    /// Apply one mutation to `parent`, building the offspring into `dest`.
    ///
    /// Returns the offspring `Individual`, or `None` if no registered mutation
    /// has a valid target node in the parent's tree.
    pub fn mutate(
        &self,
        parent: &Individual<G>,
        source: &ExprArena<G::Tag>,
        dest: &mut ExprArena<G::Tag>,
        ops: &OperationTable,
        rng: &mut dyn RngCore,
    ) -> Option<Individual<G>> {
        let root_node = source.get_root(parent.root)?;

        // Collect mutable candidate nodes from the Genome.
        let candidates: Vec<NodeId> = G::mutation_targets(parent.root, source);
        if candidates.is_empty() {
            return None;
        }

        // Build (mutation_index, valid_targets) pairs.
        let applicable: Vec<(usize, Vec<NodeId>)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, (_, m))| {
                let targets: Vec<NodeId> = candidates
                    .iter()
                    .copied()
                    .filter(|&id| source.get_node(id).is_some_and(|n| m.applies_to(n.kind)))
                    .collect();
                if targets.is_empty() {
                    None
                } else {
                    Some((i, targets))
                }
            })
            .collect();

        if applicable.is_empty() {
            return None;
        }

        // Weighted selection restricted to applicable mutations.
        let applicable_weight: f32 = applicable.iter().map(|(i, _)| self.entries[*i].0).sum();

        let mut pick = rng.random::<f32>() * applicable_weight;
        let (chosen_idx, targets) = applicable
            .iter()
            .find(|(i, _)| {
                pick -= self.entries[*i].0;
                pick <= 0.0
            })
            .unwrap_or(applicable.last().unwrap());

        let mutation = &*self.entries[*chosen_idx].1;

        // Pick a target uniformly.
        let target = targets[rng.random_range(0..targets.len())];

        // Clone parent params — offspring gets its own copy.
        let mut params = parent.parameters.clone();

        let mut ctx = MutationContext::new(source, ops, rng, dest, &mut params);
        let new_root_node = copy_over_replacing(root_node, target, mutation, &mut ctx);
        drop(ctx);

        let new_root = dest.add_root(new_root_node);
        Some(Individual::new(new_root, params))
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
        ast::{ExprArena, ExprNode},
        gp::{
            Individual,
            mutation::{
                MutationContext, Mutator,
                builtin::{ParamJitter, PointMutation, SubtreeMutation},
            },
            subtree::GrowSubtreeConfig,
            test_genome::TestSimpleGenome,
        },
        ops::{OperationTableBuilder, builtin::MathBaseOps},
        types::{NodeId, OperationId, ParameterId, RootId, Scalar},
    };

    fn base_ops() -> crate::ops::OperationTable {
        let mut b = OperationTableBuilder::new();
        b.register_set::<MathBaseOps>();
        b.build()
    }

    // Build a simple binary tree: add(param(0), param(1))
    fn build_two_param_tree(arena: &mut ExprArena<()>) -> (RootId, Vec<Scalar>) {
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1u16), ()));
        let add = arena.add(ExprNode::new_binary(p0, p1, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        (root, vec![1.0, 2.0])
    }

    #[test]
    fn copy_subtree_round_trip() {
        let mut rng = StdRng::seed_from_u64(0);
        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        let (root, mut params) = build_two_param_tree(&mut src);
        let root_node = src.get_root(root).unwrap();

        {
            let mut ctx: MutationContext<'_, TestSimpleGenome> =
                MutationContext::new(&src, &ops, &mut rng, &mut dest, &mut params);

            let copied = ctx.copy_subtree(root_node);

            // Node counts should match.
            let src_ids: Vec<_> = src.iter_expr_nodes(root).map(|(id, _)| id).collect();
            let dest_root = dest.add_root(copied);
            let dest_ids: Vec<_> = dest.iter_expr_nodes(dest_root).map(|(id, _)| id).collect();
            assert_eq!(src_ids.len(), dest_ids.len());
        }
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

        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(1);

        let mut mutator: Mutator<TestSimpleGenome> = Mutator::new();
        mutator.add(1.0, PointMutation);

        let offspring = mutator
            .mutate(&parent, &src, &mut dest, &ops, &mut rng)
            .unwrap();

        // Same number of nodes.
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

        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(2);

        let mut mutator: Mutator<TestSimpleGenome> = Mutator::new();
        mutator.add(1.0, ParamJitter { stddev: 0.1 });

        let offspring = mutator
            .mutate(&parent, &src, &mut dest, &ops, &mut rng)
            .unwrap();

        // Same tree structure (same number of nodes).
        let src_count = src.iter_expr_nodes(parent.root).count();
        let dest_count = dest.iter_expr_nodes(offspring.root).count();
        assert_eq!(src_count, dest_count);

        // Exactly one parameter should have changed.
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

        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(3);

        let cfg = GrowSubtreeConfig {
            max_depth: 3,
            p_terminal: 0.3,
            n_variables: 2,
            const_range: (-1.0, 1.0),
        };

        let mut mutator: Mutator<TestSimpleGenome> = Mutator::new();
        mutator.add(1.0, SubtreeMutation { grow: cfg });

        let offspring = mutator
            .mutate(&parent, &src, &mut dest, &ops, &mut rng)
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

        // Must not panic.
        let _ = eval_ctx.eval_batch(root_node, inputs.view(), params_arr.view());
    }

    // ---------------------------------------------------------------------------
    // Determinism — same seed ⇒ identical offspring
    // ---------------------------------------------------------------------------
    #[test]
    fn mutator_is_deterministic() {
        let ops = base_ops();

        let run = |seed: u64| -> Vec<NodeId> {
            let mut src: ExprArena<()> = ExprArena::new();
            let mut dest: ExprArena<()> = ExprArena::new();
            let (root, params) = build_two_param_tree(&mut src);
            let parent = Individual::<TestSimpleGenome>::new(root, params);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut mutator: Mutator<TestSimpleGenome> = Mutator::new();
            mutator.add(1.0, PointMutation);

            let offspring = mutator
                .mutate(&parent, &src, &mut dest, &ops, &mut rng)
                .unwrap();
            dest.iter_expr_nodes(offspring.root)
                .map(|(id, _)| id)
                .collect()
        };

        assert_eq!(run(99), run(99));
    }
}
