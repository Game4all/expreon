use rand::{Rng, RngCore};

use expreon_ast::{ExprArena, NodeId};
use expreon_eval::ops::OperationTable;

use crate::gp::{Breeder, Fitness, Genome, Individual, Scored};

use super::Mutation;

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
        // Collect mutable candidate nodes from the genome of the invidivual.
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

        super::apply_mutation(mutation, target, parent, source, dest, ops, rng)
    }

    /// Breeds `parent` (from the breeder's source generation) by applying one
    /// mutation, building the offspring into the breeder's destination arena and
    /// offering it to the breeder for insertion (unscored) into the
    /// destination population.
    ///
    /// Returns `None` if no registered mutation had a valid target in the
    /// parent's tree, or if the breeder refused the finished offspring (e.g.
    /// a [`crate::gp::GatedGenerationBreeder`]'s hook closure rejected it) —
    /// the two cases aren't distinguished by the return value.
    pub fn breed<'b, F: Fitness, B: Breeder<G, F> + ?Sized>(
        &self,
        breeder: &'b mut B,
        parent: &Scored<G, F>,
        rng: &mut dyn RngCore,
    ) -> Option<&'b mut Scored<G, F>> {
        let child = {
            let parts = breeder.parts();
            self.mutate(&parent.individual, parts.source, parts.dest, parts.ops, rng)?
        };
        breeder.commit(child)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use expreon_ast::{ExprArena, ExprNode, NodeId, OperationId, ParameterId, RootId, Scalar};
    use expreon_eval::ops::{OperationTable, OperationTableBuilder, builtin::MathBaseOps};

    use crate::gp::builder::NodeBuilder;
    use crate::gp::{
        Context, GatedGenerationBreeder, Individual, ScalarFitness,
        mutation::{Mutator, builtin::PointMutation},
        test_genome::TestSimpleGenome,
    };

    fn base_ops() -> OperationTable {
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
            let root_node = dest.get_root(offspring.root).unwrap();
            dest.iter_expr_nodes(root_node).map(|(id, _)| id).collect()
        };

        assert_eq!(run(99), run(99));
    }

    // ---------------------------------------------------------------------------
    // `breed` is generic over `Breeder` — a gated breeder can refuse a child
    // ---------------------------------------------------------------------------
    #[test]
    fn breed_respects_breeder_acceptance() {
        let ops = base_ops();

        let mut ctx: Context<TestSimpleGenome, ScalarFitness> = Context::new(ops);
        let mut rng = StdRng::seed_from_u64(99);
        {
            let mut b = ctx.builder(&mut rng);
            let pid0 = b.new_parameter(1.0);
            let pid1 = b.new_parameter(2.0);
            let n0 = b.emit(ExprNode::new_parameter(pid0, ()));
            let n1 = b.emit(ExprNode::new_parameter(pid1, ()));
            let add = b.emit(ExprNode::new_binary(n0, n1, OperationId::from(0u16), ()));
            b.finish(add);
        }

        let mut mutator: Mutator<TestSimpleGenome> = Mutator::new();
        mutator.add(1.0, PointMutation);
        let parent = &ctx.current.population[0];

        // An always-accepting gated breeder still produces a child.
        {
            let mut breeding = GatedGenerationBreeder::new(
                &ctx.current,
                &mut ctx.next,
                &ctx.operations,
                |_: &_, _: &_| true,
            );
            assert!(mutator.breed(&mut breeding, parent, &mut rng).is_some());
        }
        ctx.next.clear();

        // An always-rejecting gated breeder never commits, even though
        // `mutate` itself succeeds deterministically on this tree (the `add`
        // node is PointMutation's only valid target).
        {
            let mut breeding = GatedGenerationBreeder::new(
                &ctx.current,
                &mut ctx.next,
                &ctx.operations,
                |_: &_, _: &_| false,
            );
            assert!(mutator.breed(&mut breeding, parent, &mut rng).is_none());
        }
        assert_eq!(ctx.next.population.len(), 0);
    }
}
