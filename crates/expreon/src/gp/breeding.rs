use rand::RngCore;

use expreon_ast::{ExprArena, ExprNode, NodeId, ParameterId, Scalar};
use expreon_eval::ops::OperationTable;

use crate::gp::builder::NodeBuilder;
use crate::gp::{Fitness, Generation, Genome, Individual, IndividualBuilder, Population, Scored};

/// Borrow-split pieces a breeding operation needs to build an offspring's
/// nodes: the source arena to read the parent from, the destination arena to
/// write the offspring into, and the operation table.
pub struct GenerationBreederParts<'a, G: Genome> {
    pub source: &'a ExprArena<G::Tag>,
    pub dest: &'a mut ExprArena<G::Tag>,
    pub ops: &'a OperationTable,
    pub input_dim: u16,
}

/// Represents a breeder that is used to prepare the
/// next generation of individuals based on the current source one.
pub trait Breeder<G: Genome, F: Fitness> {
    /// Borrow-split view over the source/destination arenas and the
    /// operation table, for building an offspring's nodes.
    fn parts(&mut self) -> GenerationBreederParts<'_, G>;

    /// Prepares a finished `offspring` (already rooted in the destination
    /// arena) for insertion into the destination generation.
    ///
    /// Returns a mutable reference to its (unscored) [`Scored`] individual slot if
    /// accepted, or `None` if the breeder implementation rejected it. This method can
    /// be used to implement hooks to enforce constraints on individuals after mutation / crossover.
    fn commit(&mut self, offspring: Individual<G>) -> Option<&mut Scored<G, F>>;

    /// Copies `parent` unchanged from the source generation into the
    /// destination one: its AST is deep-copied into the destination arena and
    /// its fitness is carried over (so it won't be re-evaluated). This is the
    /// primitive for elitism / survivor copying.
    ///
    /// ### Note
    ///
    /// This method DOES go through [`Breeder::commit()`] so an elite may
    /// be potentially rejected from being copied-over if the downstream breeder implementation has
    /// constraint enforcement in its [`Breeder::commit()`] method implementation.
    ///
    fn copy_individual_over(&mut self, parent: &Scored<G, F>) -> Option<&mut Scored<G, F>> {
        let new_root = {
            let parts = self.parts();
            parts
                .source
                .copy_root_over(parent.individual.root, parts.dest)
                .expect("invalid root in copy_individual_over")
        };
        let child = self.commit(Individual::new(
            new_root,
            parent.individual.parameters.clone(),
        ))?;
        child.fitness = parent.fitness.clone();
        Some(child)
    }
}

/// Split-borrow view used to populate the next generation.
///
/// Wraps the source generation (read) and the destination generation
/// (write): breeding operations read parents from `source` and build offspring
/// into `dest`.
///
/// Calling [`crate::gp::Context::advance`] on the parent context finalizes the generation.
pub struct GenerationBreeder<'a, G: Genome, F: Fitness> {
    pub source: &'a Generation<G, F>,
    dest: &'a mut Generation<G, F>,
    ops: &'a OperationTable,
    input_dim: u16,
}

impl<'a, G: Genome, F: Fitness> GenerationBreeder<'a, G, F> {
    pub fn new(
        source: &'a Generation<G, F>,
        dest: &'a mut Generation<G, F>,
        ops: &'a OperationTable,
        input_dim: u16,
    ) -> Self {
        Self {
            source,
            dest,
            ops,
            input_dim,
        }
    }

    /// Returns a builder for constructing a brand-new individual into the
    /// destination arena. [`IndividualBuilder::finish`] inserts it (unscored)
    /// into the next generation and returns a reference to it.
    pub fn builder<'b>(&'b mut self, rng: &'b mut dyn RngCore) -> IndividualBuilder<'b, G, F> {
        IndividualBuilder::new(
            &mut self.dest.arena,
            &mut self.dest.population,
            self.ops,
            self.input_dim,
            rng,
        )
    }
}

impl<'a, G: Genome, F: Fitness> Breeder<G, F> for GenerationBreeder<'a, G, F> {
    fn parts(&mut self) -> GenerationBreederParts<'_, G> {
        GenerationBreederParts {
            source: &self.source.arena,
            dest: &mut self.dest.arena,
            ops: self.ops,
            input_dim: self.input_dim,
        }
    }

    fn commit(&mut self, offspring: Individual<G>) -> Option<&mut Scored<G, F>> {
        Some(self.dest.population.insert(offspring))
    }
}

/// A [`GenerationBreeder`] variant that runs a hook closure before every
/// insertion into the destination generation, letting the caller reject a
/// finished offspring. Otherwise behaves exactly like [`GenerationBreeder`];
/// the two variants don't influence each other.
///
/// The hook is asked to accept or reject each finished offspring: `candidate`
/// is the offspring about to be inserted, and `arena` is the *destination*
/// arena — `candidate`'s nodes and its root are already written into it, so
/// the hook can measure the finished tree (depth, node count, ...) before
/// deciding. Useful for post-hoc constraints that can only be checked on a
/// *finished* tree — e.g. a maximum depth — since mutations don't generally
/// know how deep their target sits in the parent, or how deep the subtree
/// they graft will end up being once rebuilt.
///
/// The hook is stored by value and must be `'static` (own everything it
/// captures), so a plain closure — optionally capturing and mutating its own
/// state across calls — is all that's needed; there's no trait to implement.
///
/// ### Rejected candidates leave their nodes in the destination arena
///
/// Offspring are built directly into `dest`'s arena; there is no staging
/// buffer. By the time the hook sees a candidate, its nodes *and its root*
/// are already there, and [`ExprArena`] cannot take them back — it is a bump
/// arena whose only reclamation is [`ExprArena::clear`].
///
/// Rejecting therefore orphans those nodes, deliberately. The cost is bounded:
/// - nothing in the destination population references them, so they are
///   invisible to scoring, selection and reporting;
/// - the following generation is bred by copying from the *population*, so
///   orphans are never carried forward;
/// - [`crate::gp::Context::advance`] clears the arena when the generation is
///   retired, at which point they are gone.
///
/// The trade is peak arena memory for a single generation, against a staging
/// arena plus an extra deep copy for every *accepted* offspring. A hook that
/// rejects a large fraction of candidates leaves a correspondingly large
/// number of dead nodes in that generation's arena.
///
/// Note also that a rejected candidate still consumed a `RootId`, so root ids
/// in the destination arena are not dense with respect to population
/// indices. Nothing in the crate assumes they are.
pub struct GatedGenerationBreeder<'a, G: Genome, F: Fitness, H>
where
    H: FnMut(&Individual<G>, &ExprArena<G::Tag>) -> bool + 'static,
{
    pub source: &'a Generation<G, F>,
    dest: &'a mut Generation<G, F>,
    ops: &'a OperationTable,
    input_dim: u16,
    hook: H,
}

impl<'a, G: Genome, F: Fitness, H> GatedGenerationBreeder<'a, G, F, H>
where
    H: FnMut(&Individual<G>, &ExprArena<G::Tag>) -> bool + 'static,
{
    pub fn new(
        source: &'a Generation<G, F>,
        dest: &'a mut Generation<G, F>,
        ops: &'a OperationTable,
        input_dim: u16,
        hook: H,
    ) -> Self {
        Self {
            source,
            dest,
            ops,
            input_dim,
            hook,
        }
    }

    /// Read-only view of the destination generation as bred so far.
    pub fn dest(&self) -> &Generation<G, F> {
        self.dest
    }

    /// Returns a builder for constructing a brand-new individual into the
    /// destination arena, gated by this breeder's hook closure.
    /// [`GatedIndividualBuilder::finish`] inserts it (unscored) into the next
    /// generation only if the hook accepts it.
    pub fn builder<'b>(
        &'b mut self,
        rng: &'b mut dyn RngCore,
    ) -> GatedIndividualBuilder<'b, G, F, H> {
        GatedIndividualBuilder::new(
            &mut self.dest.arena,
            &mut self.dest.population,
            self.ops,
            self.input_dim,
            rng,
            &mut self.hook,
        )
    }
}

impl<'a, G: Genome, F: Fitness, H> Breeder<G, F> for GatedGenerationBreeder<'a, G, F, H>
where
    H: FnMut(&Individual<G>, &ExprArena<G::Tag>) -> bool + 'static,
{
    fn parts(&mut self) -> GenerationBreederParts<'_, G> {
        GenerationBreederParts {
            source: &self.source.arena,
            dest: &mut self.dest.arena,
            ops: self.ops,
            input_dim: self.input_dim,
        }
    }

    fn commit(&mut self, offspring: Individual<G>) -> Option<&mut Scored<G, F>> {
        if !(self.hook)(&offspring, &self.dest.arena) {
            return None;
        }
        Some(self.dest.population.insert(offspring))
    }
}

/// A [`GatedGenerationBreeder`] counterpart to [`IndividualBuilder`], gated by
/// a hook closure.
///
/// See [`GatedGenerationBreeder`] for the orphan-node caveat:
/// [`Self::finish`] registers the root and writes the individual into the
/// arena *before* consulting the hook, so a rejected individual's nodes
/// remain in the arena, unreferenced by the population.
pub struct GatedIndividualBuilder<'a, G: Genome, F: Fitness, H>
where
    H: FnMut(&Individual<G>, &ExprArena<G::Tag>) -> bool + 'static,
{
    arena: &'a mut ExprArena<G::Tag>,
    population: &'a mut Population<G, F>,
    ops: &'a OperationTable,
    input_dim: u16,
    rng: &'a mut dyn RngCore,
    hook: &'a mut H,
    params: Vec<Scalar>,
}

impl<'a, G: Genome, F: Fitness, H> GatedIndividualBuilder<'a, G, F, H>
where
    H: FnMut(&Individual<G>, &ExprArena<G::Tag>) -> bool + 'static,
{
    pub(crate) fn new(
        arena: &'a mut ExprArena<G::Tag>,
        population: &'a mut Population<G, F>,
        ops: &'a OperationTable,
        input_dim: u16,
        rng: &'a mut dyn RngCore,
        hook: &'a mut H,
    ) -> Self {
        Self {
            arena,
            population,
            ops,
            input_dim,
            rng,
            hook,
            params: Vec::new(),
        }
    }

    /// Registers `root_node` as an expression root and offers the finished
    /// (unscored) individual to the builder's hook closure. Returns a
    /// mutable reference to its [`Scored`] slot if accepted, or `None` if the
    /// hook rejected it — see [`GatedGenerationBreeder`] for what that means
    /// for the individual's nodes. Consumes the builder.
    pub fn finish(self, root_node: NodeId) -> Option<&'a mut Scored<G, F>> {
        let root = self.arena.add_root(root_node);
        let candidate = Individual::new(root, self.params);
        if !(self.hook)(&candidate, self.arena) {
            return None;
        }
        Some(self.population.insert(candidate))
    }
}

impl<'a, G: Genome, F: Fitness, H> NodeBuilder for GatedIndividualBuilder<'a, G, F, H>
where
    H: FnMut(&Individual<G>, &ExprArena<G::Tag>) -> bool + 'static,
{
    type Genome = G;

    fn rng(&mut self) -> &mut dyn RngCore {
        self.rng
    }

    fn ops(&self) -> &OperationTable {
        self.ops
    }

    fn input_dim(&self) -> u16 {
        self.input_dim
    }

    fn emit(&mut self, node: ExprNode<G::Tag>) -> NodeId {
        self.arena.add(node)
    }

    fn new_parameter(&mut self, value: Scalar) -> ParameterId {
        let id = ParameterId::from(self.params.len() as u16);
        self.params.push(value);
        id
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use expreon_ast::{ExprNode, NodeId};
    use expreon_eval::ops::{OperationTableBuilder, builtin::MathBaseOps};

    use crate::gp::builder::NodeBuilder;
    use crate::gp::test_genome::{TestSimpleGenome, test_dataset};
    use crate::gp::{Breeder, Context, GatedGenerationBreeder, GenerationBreeder, ScalarFitness};

    #[test]
    fn copy_individual_over_carries_ast_and_fitness() {
        let mut ob = OperationTableBuilder::new();
        ob.register_set::<MathBaseOps>();
        let ops = ob.build();

        let mut ctx: Context<TestSimpleGenome, ScalarFitness> = Context::new(ops, &test_dataset());
        let mut rng = StdRng::seed_from_u64(0);

        // Build a single-parameter individual into the current generation and
        // score it through the reference `finish` returns.
        {
            let mut b = ctx.builder(&mut rng);
            let p = b.new_parameter(1.5);
            let node = b.emit(ExprNode::new_parameter(p, ()));
            b.finish(node).fitness = Some(ScalarFitness(42.0));
        }

        let parent = &ctx.current.population[0];
        let mut breeding =
            GenerationBreeder::new(&ctx.current, &mut ctx.next, &ctx.operations, 2);
        breeding.copy_individual_over(parent);
        ctx.advance();

        let copied = &ctx.current.population[0];
        assert_eq!(copied.fitness, Some(ScalarFitness(42.0)));
        assert_eq!(copied.individual.parameters, vec![1.5]);
        assert!(ctx.current.arena.get_root(copied.individual.root).is_some());
    }

    #[test]
    fn source_field_outlives_a_later_mutable_call() {
        let mut ob = OperationTableBuilder::new();
        ob.register_set::<MathBaseOps>();
        let ops = ob.build();

        let mut ctx: Context<TestSimpleGenome, ScalarFitness> = Context::new(ops, &test_dataset());
        let mut rng = StdRng::seed_from_u64(0);
        {
            let mut b = ctx.builder(&mut rng);
            let p = b.new_parameter(1.5);
            let node = b.emit(ExprNode::new_parameter(p, ()));
            b.finish(node).fitness = Some(ScalarFitness(42.0));
        }

        let mut breeding =
            GenerationBreeder::new(&ctx.current, &mut ctx.next, &ctx.operations, 2);
        let parent = &breeding.source.population[0];
        breeding.copy_individual_over(parent);
        drop(breeding);
        assert_eq!(ctx.next.population.len(), 1);
    }

    /// Builds a context with a single scored individual in `current`, ready
    /// to be bred from.
    fn ctx_with_one_individual() -> Context<TestSimpleGenome, ScalarFitness> {
        let mut ob = OperationTableBuilder::new();
        ob.register_set::<MathBaseOps>();
        let ops = ob.build();

        let mut ctx: Context<TestSimpleGenome, ScalarFitness> = Context::new(ops, &test_dataset());
        let mut rng = StdRng::seed_from_u64(0);
        let mut b = ctx.builder(&mut rng);
        let p = b.new_parameter(1.5);
        let node = b.emit(ExprNode::new_parameter(p, ()));
        b.finish(node).fitness = Some(ScalarFitness(42.0));
        ctx
    }

    #[test]
    fn gated_breeder_source_field_outlives_a_later_mutable_call() {
        let mut ctx = ctx_with_one_individual();

        let mut breeding = GatedGenerationBreeder::new(
            &ctx.current,
            &mut ctx.next,
            &ctx.operations,
            2,
            |_: &_, _: &_| true,
        );
        let parent = &breeding.source.population[0];
        breeding.copy_individual_over(parent);
        assert_eq!(breeding.dest().population.len(), 1);
    }

    #[test]
    fn gated_breeder_commits_accepted_candidate() {
        let mut ctx = ctx_with_one_individual();
        let mut rng = StdRng::seed_from_u64(0);

        let mut breeding = GatedGenerationBreeder::new(
            &ctx.current,
            &mut ctx.next,
            &ctx.operations,
            2,
            |_: &_, _: &_| true,
        );
        {
            let mut b = breeding.builder(&mut rng);
            let p = b.new_parameter(2.5);
            let node = b.emit(ExprNode::new_parameter(p, ()));
            assert!(b.finish(node).is_some());
        }
        assert_eq!(breeding.dest().population.len(), 1);
    }

    #[test]
    fn gated_breeder_rejects_candidate_and_orphans_its_nodes() {
        let mut ctx = ctx_with_one_individual();
        let mut rng = StdRng::seed_from_u64(0);

        let mut breeding = GatedGenerationBreeder::new(
            &ctx.current,
            &mut ctx.next,
            &ctx.operations,
            2,
            |_: &_, _: &_| false,
        );
        {
            let mut b = breeding.builder(&mut rng);
            let p = b.new_parameter(2.5);
            let node = b.emit(ExprNode::new_parameter(p, ()));
            assert!(b.finish(node).is_none());
        }
        // Nothing is referenced by the population...
        assert_eq!(breeding.dest().population.len(), 0);
        // ...but the rejected node was still written into the dest arena: the
        // parameter node built by the builder landed at index 0.
        assert!(
            breeding
                .dest()
                .arena
                .get_node(NodeId::from(0usize))
                .is_some()
        );
    }

    #[test]
    fn gated_breeder_gates_elitism_copies() {
        let mut ctx = ctx_with_one_individual();
        let parent = &ctx.current.population[0];

        let mut breeding = GatedGenerationBreeder::new(
            &ctx.current,
            &mut ctx.next,
            &ctx.operations,
            2,
            |_: &_, _: &_| false,
        );
        assert!(breeding.copy_individual_over(parent).is_none());
        assert_eq!(breeding.dest().population.len(), 0);
        // The rejected copy's node still landed in the dest arena at index 0.
        assert!(
            breeding
                .dest()
                .arena
                .get_node(NodeId::from(0usize))
                .is_some()
        );
    }

    #[test]
    fn orphaned_nodes_do_not_affect_the_promoted_generation() {
        let mut ctx = ctx_with_one_individual();
        let mut rng = StdRng::seed_from_u64(0);

        {
            // Rejected: writes a node into the dest arena but never commits.
            let mut breeding = GatedGenerationBreeder::new(
                &ctx.current,
                &mut ctx.next,
                &ctx.operations,
                2,
                |_: &_, _: &_| false,
            );
            let mut b = breeding.builder(&mut rng);
            let p = b.new_parameter(9.0);
            let node = b.emit(ExprNode::new_parameter(p, ()));
            assert!(b.finish(node).is_none());
        }
        {
            // A second pass (always-accept hook) commits one individual into
            // the same `next` generation, alongside the orphan above.
            let mut breeding = GatedGenerationBreeder::new(
                &ctx.current,
                &mut ctx.next,
                &ctx.operations,
                2,
                |_: &_, _: &_| true,
            );
            let mut b = breeding.builder(&mut rng);
            let p = b.new_parameter(4.0);
            let node = b.emit(ExprNode::new_parameter(p, ()));
            assert!(b.finish(node).is_some());
        }

        ctx.advance();

        // The promoted generation's population holds exactly the accepted
        // individual: the orphaned node left no trace in it.
        assert_eq!(ctx.current.population.len(), 1);
        assert_eq!(ctx.current.population[0].individual.parameters, vec![4.0]);
    }
}
