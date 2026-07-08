use std::marker::PhantomData;

use rand::RngCore;

use expreon_ast::{ExprArena, ExprNode, NodeId, NodeKind, ParameterId, RootId, Scalar};
use expreon_eval::ops::OperationTable;

use crate::gp::builder::NodeBuilder;

pub mod breeding;
pub mod builder;
pub mod fitness;
pub mod mutation;
pub mod population;
pub mod subtree;

pub use breeding::GenerationBreeder;
pub use fitness::{
    Fitness, IntegerFitness, ParetoFitness, ScalarFitness, k_best_of, k_best_of_with_comparator,
    k_tournament_selection, k_tournament_selection_with_comparator, pareto_cmp,
};
pub use population::{Population, Scored};

/// Base trait for a genome.
///
/// A genome may expose an additional tag type used to control what parts can explicitly be modified by overriding [`Self::mutation_targets`]
pub trait Genome: Clone {
    type Tag: Clone;

    /// Dimension of the input vector: the number of input variables available
    /// to expressions built from this genome. Valid variable IDs are `0..INPUT_DIM`.
    const INPUT_DIM: u16;

    /// Returns a list of potential mutation targets for this individual genome.
    fn mutation_targets(root: RootId, arena: &ExprArena<Self::Tag>) -> Vec<NodeId> {
        arena.walk_root(root).into_iter().flatten().collect()
    }

    /// Gets a tag to attach to a new or structurally-changed expression node.
    fn get_tag_for_node(kind: NodeKind) -> Self::Tag;
}

/// A single individual: contains an handle to the root expression node
/// and its parameters. 
pub struct Individual<G: Genome> {
    pub root: RootId,
    pub parameters: Vec<Scalar>,
    _ctx: PhantomData<G>,
}

impl<G: Genome> Individual<G> {
    pub fn new(root: RootId, parameters: Vec<Scalar>) -> Self {
        Self {
            root,
            parameters,
            _ctx: PhantomData,
        }
    }
}

/// One generation: an expression arena together with the population of
/// individuals whose roots live in it. An individual's `RootId` is only
/// meaningful paired with its generation's arena.
pub struct Generation<G: Genome, F: Fitness> {
    pub arena: ExprArena<G::Tag>,
    pub population: Population<G, F>,
}

impl<G: Genome, F: Fitness> Generation<G, F> {
    /// An empty generation.
    pub const fn new() -> Self {
        Self {
            arena: ExprArena::new(),
            population: Population::new(),
        }
    }

    /// Removes all expressions and individuals, resetting the buffer.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.population.clear();
    }
}

impl<G: Genome, F: Fitness> Default for Generation<G, F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Global context for genetic programming.
///
/// Holds the operation table and two generations: `current` is the live one
/// (individuals are built, scored and selected here) and `next` is the one
/// being bred into via [`GenerationBreeder`]. [`Context::advance`] promotes `next` to
/// `current`.
///
/// The fields are public on purpose: accessing them directly
/// (`&ctx.current.arena`, `&mut ctx.current.population`, ...) lets the borrow
/// checker split the borrows for ie. reading the arena while writing fitness
/// into the population.
pub struct Context<G: Genome, F: Fitness> {
    pub current: Generation<G, F>,
    pub next: Generation<G, F>,
    pub operations: OperationTable,
}

/// A `NodeBuilder` returned by [`Context::builder`] for constructing a single
/// individual by hand. Call [`IndividualBuilder::finish`] to register the root
/// and insert the individual into the target generation, obtaining a mutable
/// reference to it. The individual is built into, and inserted into, the
/// generation the builder was created for.
pub struct IndividualBuilder<'a, G: Genome, F: Fitness> {
    arena: &'a mut ExprArena<G::Tag>,
    population: &'a mut Population<G, F>,
    ops: &'a OperationTable,
    rng: &'a mut dyn RngCore,
    params: Vec<Scalar>,
}

impl<G: Genome, F: Fitness> Context<G, F> {
    pub const fn new(op: OperationTable) -> Self {
        Self {
            current: Generation::new(),
            next: Generation::new(),
            operations: op,
        }
    }

    /// Promotes the freshly-bred `next` generation to `current`. The old
    /// current generation is cleared, ready to receive the one after.
    pub fn advance(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
        self.next.clear();
    }

    /// Returns a [`GenerationBreeder`] view over `current` (read) and `next` (write) to
    /// build the next generation. Finalize the cycle with [`Context::advance`].
    ///
    /// This borrows the whole context mutably; if a reference into `current`
    /// (e.g. a selected parent) is already held, assemble the view from the
    /// fields with [`GenerationBreeder::new`] instead.
    pub fn breeder(&mut self) -> GenerationBreeder<'_, G, F> {
        GenerationBreeder::new(&self.current, &mut self.next, &self.operations)
    }

    /// Returns a builder for constructing a single individual into the
    /// `current` generation. [`IndividualBuilder::finish`] registers the root
    /// and inserts the individual (unscored), returning a reference to it.
    pub fn builder<'a>(&'a mut self, rng: &'a mut dyn RngCore) -> IndividualBuilder<'a, G, F> {
        IndividualBuilder::new(
            &mut self.current.arena,
            &mut self.current.population,
            &self.operations,
            rng,
        )
    }
}

impl<'a, G: Genome, F: Fitness> IndividualBuilder<'a, G, F> {
    pub(crate) fn new(
        arena: &'a mut ExprArena<G::Tag>,
        population: &'a mut Population<G, F>,
        ops: &'a OperationTable,
        rng: &'a mut dyn RngCore,
    ) -> Self {
        Self {
            arena,
            population,
            ops,
            rng,
            params: Vec::new(),
        }
    }

    /// Registers `root_node` as an expression root, inserts the finished
    /// (unscored) individual into the builder's population, and returns a
    /// mutable reference to its [`Scored`] slot (e.g. to seed its fitness).
    /// Consumes the builder.
    pub fn finish(self, root_node: NodeId) -> &'a mut Scored<G, F> {
        let root = self.arena.add_root(root_node);
        self.population.insert(Individual::new(root, self.params))
    }
}

impl<'a, G: Genome, F: Fitness> NodeBuilder<G> for IndividualBuilder<'a, G, F> {
    fn rng(&mut self) -> &mut dyn RngCore {
        self.rng
    }

    fn ops(&self) -> &OperationTable {
        self.ops
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

/// A simple test genome useful for tests: `Tag = ()`, all nodes are mutable.
#[cfg(test)]
pub(crate) mod test_genome {
    use super::*;

    #[derive(Clone)]
    pub struct TestSimpleGenome;

    impl Genome for TestSimpleGenome {
        type Tag = ();
        const INPUT_DIM: u16 = 2;

        fn get_tag_for_node(_kind: NodeKind) -> () {}
    }
}
