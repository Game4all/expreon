use rand::RngCore;

use expreon_eval::ops::OperationTable;

use crate::gp::mutation::Mutator;
use crate::gp::{Fitness, Generation, Genome, Individual, IndividualBuilder, Scored};

/// Split-borrow view used to populate the next generation.
///
/// Exposes the source generation (read) and the destination generation
/// (write): breeding operations read parents from `source` and build offspring
/// into `dest`.
///
/// Calling [`crate::gp::Context::advance`] on the parent context finalizes the generation.
pub struct GenerationBreeder<'a, G: Genome, F: Fitness> {
    pub source: &'a Generation<G, F>,
    pub dest: &'a mut Generation<G, F>,
    pub ops: &'a OperationTable,
}

impl<'a, G: Genome, F: Fitness> GenerationBreeder<'a, G, F> {
    pub fn new(
        source: &'a Generation<G, F>,
        dest: &'a mut Generation<G, F>,
        ops: &'a OperationTable,
    ) -> Self {
        Self { source, dest, ops }
    }

    /// Copies `parent` unchanged from the current generation into the next
    /// one: its AST is deep-copied into the destination arena and its fitness
    /// is carried over (so it won't be re-evaluated). This is the primitive for
    /// elitism / survivor copying.
    pub fn copy_individual_over(&mut self, parent: &Scored<G, F>) -> &mut Scored<G, F> {
        let source = self.source;
        let new_root = source
            .arena
            .copy_root_over(parent.individual.root, &mut self.dest.arena)
            .expect("invalid root in copy_individual_over");

        let child = self
            .dest
            .population
            .insert(Individual::new(new_root, parent.individual.parameters.clone()));
        child.fitness = parent.fitness.clone();
        child
    }

    /// Returns a builder for constructing a brand-new individual into the
    /// destination arena. [`IndividualBuilder::finish`] inserts it (unscored)
    /// into the next generation and returns a reference to it.
    pub fn builder<'b>(&'b mut self, rng: &'b mut dyn RngCore) -> IndividualBuilder<'b, G, F> {
        IndividualBuilder::new(
            &mut self.dest.arena,
            &mut self.dest.population,
            self.ops,
            rng,
        )
    }
}

// `Mutator` boxes its mutations (`Box<dyn Mutation<G>>`), which requires `G: 'static`.
impl<'a, G: Genome + 'static, F: Fitness> GenerationBreeder<'a, G, F> {
    /// Breeds `parent` (from the source generation) via `mutator`, building
    /// the offspring into the destination arena and inserting it (unscored)
    /// into the destination population.
    ///
    /// Returns the new individual, or `None` if no registered mutation had a
    /// valid target in the parent's tree.
    pub fn breed(
        &mut self,
        parent: &Scored<G, F>,
        mutator: &Mutator<G>,
        rng: &mut dyn RngCore,
    ) -> Option<&mut Scored<G, F>> {
        let source = self.source;
        let child = mutator.mutate(
            &parent.individual,
            &source.arena,
            &mut self.dest.arena,
            self.ops,
            rng,
        )?;
        Some(self.dest.population.insert(child))
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use expreon_ast::ExprNode;
    use expreon_eval::ops::{OperationTableBuilder, builtin::MathBaseOps};

    use crate::gp::builder::NodeBuilder;
    use crate::gp::test_genome::TestSimpleGenome;
    use crate::gp::{Context, GenerationBreeder, ScalarFitness};

    #[test]
    fn copy_individual_over_carries_ast_and_fitness() {
        let mut ob = OperationTableBuilder::new();
        ob.register_set::<MathBaseOps>();
        let ops = ob.build();

        let mut ctx: Context<TestSimpleGenome, ScalarFitness> = Context::new(ops);
        let mut rng = StdRng::seed_from_u64(0);

        // Build a single-parameter individual into the current generation and
        // score it through the reference `finish` returns.
        {
            let mut b = ctx.builder(&mut rng);
            let p = b.new_parameter(1.5);
            let node = b.emit(ExprNode::new_parameter(p, ()));
            b.finish(node).fitness = Some(ScalarFitness(42.0));
        }

        // Copy it into the next generation, then make that generation current.
        // `parent` borrows `ctx.current` (shared) while `Breeding::new` borrows
        // `ctx.next` mutably — disjoint field borrows.
        let parent = &ctx.current.population[0];
        let mut breeding = GenerationBreeder::new(&ctx.current, &mut ctx.next, &ctx.operations);
        breeding.copy_individual_over(parent);
        ctx.advance();

        let copied = &ctx.current.population[0];
        // Fitness carried over (elitism skips re-evaluation)...
        assert_eq!(copied.fitness, Some(ScalarFitness(42.0)));
        // ...and the AST + parameters were deep-copied into the dest arena.
        assert_eq!(copied.individual.parameters, vec![1.5]);
        assert!(ctx.current.arena.get_root(copied.individual.root).is_some());
    }
}
