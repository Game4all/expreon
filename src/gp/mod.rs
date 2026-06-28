use std::marker::PhantomData;

use rand::RngCore;

use crate::{
    ast::{ExprArena, ExprNode, NodeKind},
    gp::builder::NodeBuilder,
    ops::OperationTable,
    types::{NodeId, ParameterId, RootId, Scalar},
};

pub mod builder;
pub mod mutation;
pub mod subtree;

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
        arena.walk_expr(root).unwrap().collect()
    }

    /// Gets a tag to attach to a new or structurally-changed expression node.
    fn get_tag_for_node(kind: NodeKind) -> Self::Tag;
}

/// A single individual in the population.
/// Holds a reference to its root expression node and its parameters
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

/// A collection of individuals making up a generation.
pub struct Population<G: Genome> {
    individuals: Vec<Individual<G>>,
}

enum CurrentBuffer {
    A,
    B,
}

/// Global context for genetic programming.
/// Holds operation table and ping-pong expression arenas.
pub struct Context<G: Genome> {
    arena_a: ExprArena<G::Tag>,
    arena_b: ExprArena<G::Tag>,
    current_buffer: CurrentBuffer,
    pub operations: OperationTable,
}

/// A `NodeBuilder` returned by [`Context::builder`] for constructing a single
/// individual by hand. Call [`IndividualBuilder::finish`] to register the root
/// and obtain the [`Individual`].
pub struct IndividualBuilder<'a, G: Genome> {
    arena: &'a mut ExprArena<G::Tag>,
    ops: &'a OperationTable,
    rng: &'a mut dyn RngCore,
    params: Vec<Scalar>,
}

impl<G: Genome> Context<G> {
    pub const fn new(op: OperationTable) -> Self {
        Self {
            arena_a: ExprArena::new(),
            arena_b: ExprArena::new(),
            current_buffer: CurrentBuffer::A,
            operations: op,
        }
    }

    /// Returns `(source_arena, dest_arena, ops)`. Source is the current live buffer;
    /// dest is the buffer offspring are built into.
    pub fn borrow_parts(&mut self) -> (&ExprArena<G::Tag>, &mut ExprArena<G::Tag>, &OperationTable) {
        match self.current_buffer {
            CurrentBuffer::A => (&self.arena_a, &mut self.arena_b, &self.operations),
            CurrentBuffer::B => (&self.arena_b, &mut self.arena_a, &self.operations),
        }
    }

    /// Returns an immutable reference to the currently active (source) arena.
    pub fn source_arena(&self) -> &ExprArena<G::Tag> {
        match self.current_buffer {
            CurrentBuffer::A => &self.arena_a,
            CurrentBuffer::B => &self.arena_b,
        }
    }

    /// Swaps the active buffer. The old dest becomes the new source.
    /// The new dest is cleared, ready for the next generation.
    pub fn swap(&mut self) {
        self.current_buffer = match self.current_buffer {
            CurrentBuffer::A => {
                self.arena_a.clear();
                CurrentBuffer::B
            }
            CurrentBuffer::B => {
                self.arena_b.clear();
                CurrentBuffer::A
            }
        };
    }

    /// Returns a builder for constructing a single individual into the active
    /// (source) arena. Call [`IndividualBuilder::finish`] once the root node is
    /// ready to register the root and obtain the [`Individual`].
    pub fn builder<'a>(&'a mut self, rng: &'a mut dyn RngCore) -> IndividualBuilder<'a, G> {
        let arena = match self.current_buffer {
            CurrentBuffer::A => &mut self.arena_a,
            CurrentBuffer::B => &mut self.arena_b,
        };
        IndividualBuilder::new(arena, &self.operations, rng)
    }
}

impl<'a, G: Genome> IndividualBuilder<'a, G> {
    pub(crate) fn new(
        arena: &'a mut ExprArena<G::Tag>,
        ops: &'a OperationTable,
        rng: &'a mut dyn RngCore,
    ) -> Self {
        Self {
            arena,
            ops,
            rng,
            params: Vec::new(),
        }
    }

    /// Register `root_node` as an expression root and return the finished
    /// individual. Consumes the builder.
    pub fn finish(self, root_node: NodeId) -> (Individual<G>, RootId) {
        let root = self.arena.add_root(root_node);
        (Individual::new(root, self.params), root)
    }
}

impl<'a, G: Genome> NodeBuilder<G> for IndividualBuilder<'a, G> {
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

impl<G: Genome> Population<G> {
    pub fn new(individuals: Vec<Individual<G>>) -> Self {
        Self { individuals }
    }

    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Individual<G>> {
        self.individuals.iter()
    }

    pub fn individuals(&self) -> &[Individual<G>] {
        &self.individuals
    }

    pub fn into_individuals(self) -> Vec<Individual<G>> {
        self.individuals
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
