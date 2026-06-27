//! Initial population construction for genetic programming runs.
//!
//! Provides [`init_population_into`] (and the [`Context::init_population`]
//! convenience on [`crate::gp::Context`]) which build a fresh set of random
//! [`Individual`]s from scratch — the first step of any GP run, before any
//! selection/mutation takes place.

use rand::RngCore;

use crate::{ast::ExprArena, ops::OperationTable};

use super::{
    Genome, Individual, IndividualBuilder,
    subtree::{TreeGenConfig, TreeMethod, gen_tree},
};

/// Strategy used to populate the initial generation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InitMethod {
    /// Every individual is a `full` tree of depth `max_depth`.
    Full,
    /// Every individual is a `grow` tree of depth `max_depth`.
    Grow,
    /// GP-standard ramped half-and-half: individuals are spread across the depth
    /// buckets `min_depth..=max_depth`, alternating between `full` and `grow`.
    RampedHalfAndHalf,
}

/// Configuration for building an initial population.
pub struct PopulationConfig {
    /// Number of individuals to create.
    pub size: usize,
    /// Strategy used to assign each individual a generation method and depth.
    pub method: InitMethod,
    /// Smallest tree depth used by ramped half-and-half.
    pub min_depth: usize,
    /// Largest tree depth (and the fixed depth for the `Full`/`Grow` methods).
    pub max_depth: usize,
    /// Generator tuning (terminal probability, variable count, constant range).
    pub tuning: TreeGenConfig,
}

/// A collection of individuals making up a generation.
pub struct Population<G: Genome> {
    individuals: Vec<Individual<G>>,
}

impl<G: Genome> Population<G> {
    pub fn new(individuals: Vec<Individual<G>>) -> Self {
        Self { individuals }
    }

    /// Number of individuals in the population.
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Whether the population is empty.
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Iterates over the individuals.
    pub fn iter(&self) -> impl Iterator<Item = &Individual<G>> {
        self.individuals.iter()
    }

    /// Borrows the underlying individuals as a slice.
    pub fn individuals(&self) -> &[Individual<G>] {
        &self.individuals
    }

    /// Consumes the population and returns the owned individuals.
    pub fn into_individuals(self) -> Vec<Individual<G>> {
        self.individuals
    }
}

/// Chooses the `(method, depth)` for the `i`-th individual under `cfg`.
fn plan_individual(cfg: &PopulationConfig, i: usize) -> (TreeMethod, usize) {
    match cfg.method {
        InitMethod::Full => (TreeMethod::Full, cfg.max_depth),
        InitMethod::Grow => (TreeMethod::Grow, cfg.max_depth),
        InitMethod::RampedHalfAndHalf => {
            let min = cfg.min_depth.min(cfg.max_depth);
            let span = cfg.max_depth - min + 1;
            // Cycle through depth buckets; flip the method each full cycle so
            // every (depth, method) combination is covered evenly.
            let depth = min + (i % span);
            let method = if (i / span).is_multiple_of(2) {
                TreeMethod::Full
            } else {
                TreeMethod::Grow
            };
            (method, depth)
        }
    }
}

/// Builds an initial population of `cfg.size` individuals into `arena`,
/// returning the [`Population`]. Each individual's tree is rooted in `arena` and
/// owns its own freshly-allocated parameter vector.
pub fn init_population_into<G: Genome>(
    arena: &mut ExprArena<G::Tag>,
    ops: &OperationTable,
    cfg: &PopulationConfig,
    rng: &mut dyn RngCore,
) -> Population<G> {
    let mut individuals = Vec::with_capacity(cfg.size);

    for i in 0..cfg.size {
        let (method, depth) = plan_individual(cfg, i);
        let mut b = IndividualBuilder::<G>::new(arena, ops, rng);
        let root_node = gen_tree(&mut b, &cfg.tuning, method, depth);
        let (individual, _) = b.finish(root_node);
        individuals.push(individual);
    }

    Population::new(individuals)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::{
        ast::{ExprArena, NodeKind},
        gp::test_genome::TestSimpleGenome,
        ops::{OperationTable, OperationTableBuilder, builtin::MathBaseOps},
        types::{NodeId, RootId},
    };

    fn make_ops() -> OperationTable {
        let mut b = OperationTableBuilder::new();
        b.register_set::<MathBaseOps>();
        b.build()
    }

    fn cfg(method: InitMethod, min_depth: usize, max_depth: usize) -> PopulationConfig {
        PopulationConfig {
            size: 16,
            method,
            min_depth,
            max_depth,
            tuning: TreeGenConfig {
                p_terminal: 0.3,
                n_variables: 2,
                const_range: (-1.0, 1.0),
            },
        }
    }

    /// Measures the depth of a tree rooted at `node` (a single terminal = 0).
    fn tree_depth(arena: &ExprArena<()>, node: NodeId) -> usize {
        match arena.get_node(node).unwrap().kind {
            NodeKind::Variable(_) | NodeKind::Parameter(_) => 0,
            NodeKind::Unary { value, .. } => 1 + tree_depth(arena, value),
            NodeKind::Binary { left, right, .. } => {
                1 + tree_depth(arena, left).max(tree_depth(arena, right))
            }
        }
    }

    /// Returns the minimum root-to-leaf depth of a tree.
    fn min_leaf_depth(arena: &ExprArena<()>, node: NodeId) -> usize {
        match arena.get_node(node).unwrap().kind {
            NodeKind::Variable(_) | NodeKind::Parameter(_) => 0,
            NodeKind::Unary { value, .. } => 1 + min_leaf_depth(arena, value),
            NodeKind::Binary { left, right, .. } => {
                1 + min_leaf_depth(arena, left).min(min_leaf_depth(arena, right))
            }
        }
    }

    fn count_param_nodes(arena: &ExprArena<()>, root: RootId) -> usize {
        arena
            .iter_expr_nodes(root)
            .filter(|(_, n)| matches!(n.kind, NodeKind::Parameter(_)))
            .count()
    }

    #[test]
    fn produces_requested_size_with_valid_roots() {
        let ops = make_ops();
        let mut arena: ExprArena<()> = ExprArena::new();
        let mut rng = StdRng::seed_from_u64(1);

        let pop = init_population_into::<TestSimpleGenome>(
            &mut arena,
            &ops,
            &cfg(InitMethod::RampedHalfAndHalf, 2, 5),
            &mut rng,
        );

        assert_eq!(pop.len(), 16);
        for ind in pop.iter() {
            assert!(arena.get_root(ind.root).is_some());
        }
    }

    #[test]
    fn respects_max_depth_and_full_is_full() {
        let ops = make_ops();

        // Full: every leaf sits at exactly max_depth.
        let mut arena: ExprArena<()> = ExprArena::new();
        let mut rng = StdRng::seed_from_u64(2);
        let pop = init_population_into::<TestSimpleGenome>(
            &mut arena,
            &ops,
            &cfg(InitMethod::Full, 4, 4),
            &mut rng,
        );
        for ind in pop.iter() {
            let node = arena.get_root(ind.root).unwrap();
            assert_eq!(tree_depth(&arena, node), 4);
            assert_eq!(min_leaf_depth(&arena, node), 4);
        }

        // Ramped: no tree exceeds max_depth.
        let mut arena2: ExprArena<()> = ExprArena::new();
        let mut rng2 = StdRng::seed_from_u64(3);
        let pop2 = init_population_into::<TestSimpleGenome>(
            &mut arena2,
            &ops,
            &cfg(InitMethod::RampedHalfAndHalf, 2, 5),
            &mut rng2,
        );
        for ind in pop2.iter() {
            let node = arena2.get_root(ind.root).unwrap();
            assert!(tree_depth(&arena2, node) <= 5);
        }
    }

    #[test]
    fn parameter_vector_matches_param_nodes() {
        let ops = make_ops();
        let mut arena: ExprArena<()> = ExprArena::new();
        let mut rng = StdRng::seed_from_u64(4);

        let pop = init_population_into::<TestSimpleGenome>(
            &mut arena,
            &ops,
            &cfg(InitMethod::RampedHalfAndHalf, 2, 5),
            &mut rng,
        );

        for ind in pop.iter() {
            assert_eq!(ind.parameters.len(), count_param_nodes(&arena, ind.root));
        }
    }

    #[test]
    fn deterministic_for_same_seed() {
        let ops = make_ops();

        let run = || -> Vec<Vec<NodeId>> {
            let mut arena: ExprArena<()> = ExprArena::new();
            let mut rng = StdRng::seed_from_u64(99);
            let pop = init_population_into::<TestSimpleGenome>(
                &mut arena,
                &ops,
                &cfg(InitMethod::RampedHalfAndHalf, 2, 5),
                &mut rng,
            );
            pop.iter()
                .map(|ind| arena.iter_expr_nodes(ind.root).map(|(id, _)| id).collect())
                .collect()
        };

        assert_eq!(run(), run());
    }

    #[test]
    fn individuals_evaluate_without_panic() {
        use crate::eval::EvalContext;
        use ndarray::Array2;

        let ops = make_ops();
        let mut arena: ExprArena<()> = ExprArena::new();
        let mut rng = StdRng::seed_from_u64(5);

        let pop = init_population_into::<TestSimpleGenome>(
            &mut arena,
            &ops,
            &cfg(InitMethod::RampedHalfAndHalf, 2, 5),
            &mut rng,
        );

        let eval_ctx = EvalContext::new(&arena, &ops);
        let inputs = Array2::from_shape_vec((1, 2), vec![0.5f32, 1.0f32]).unwrap();

        for ind in pop.iter() {
            let root_node = arena.get_root(ind.root).unwrap();
            let n = ind.parameters.len().max(1);
            let mut p = ind.parameters.clone();
            if p.is_empty() {
                p.push(0.0);
            }
            let params = Array2::from_shape_vec((1, n), p).unwrap();
            let _ = eval_ctx.eval_batch(root_node, inputs.view(), params.view());
        }
    }
}
