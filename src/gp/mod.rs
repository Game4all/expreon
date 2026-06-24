use std::marker::PhantomData;

use crate::{
    ast::{ExprArena, ExprNode, NodeKind},
    ops::OperationTable,
    types::{NodeId, ParameterId, RootId, Scalar},
};

pub mod mutation;
pub mod subtree;

/// Base trait for a genome.
pub trait Genome: Clone {
    type Tag: Clone;

    /// Returns a list of potential mutation targets for this individual genome.
    fn mutation_targets(root: RootId, arena: &ExprArena<Self::Tag>) -> Vec<NodeId>;

    /// Gets a tag to attach to a new or structurally-changed expression node.
    fn get_tag_for_node(kind: NodeKind) -> Self::Tag;
}

/// A single individual in the population.
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

impl<G: Genome> Context<G> {
    pub const fn new(op: OperationTable) -> Self {
        Self {
            arena_a: ExprArena::new(),
            arena_b: ExprArena::new(),
            current_buffer: CurrentBuffer::A,
            operations: op,
        }
    }

    /// Returns `(source_arena, dest_arena)`. Source is the current live buffer;
    /// dest is the buffer offspring are built into.
    pub fn buffers(&mut self) -> (&ExprArena<G::Tag>, &mut ExprArena<G::Tag>) {
        match self.current_buffer {
            CurrentBuffer::A => (&self.arena_a, &mut self.arena_b),
            CurrentBuffer::B => (&self.arena_b, &mut self.arena_a),
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
}

/// A helper to add nodes to an arena with automatic tag generation via `G`.
pub(crate) fn emit_node<G: Genome>(arena: &mut ExprArena<G::Tag>, kind: NodeKind) -> NodeId {
    let tag = G::get_tag_for_node(kind);
    arena.add(ExprNode::new(kind, tag))
}

/// A helper to add a node to an arena preserving an existing tag.
pub(crate) fn preserve_node<G: Genome>(
    arena: &mut ExprArena<G::Tag>,
    kind: NodeKind,
    tag: G::Tag,
) -> NodeId {
    arena.add(ExprNode::new(kind, tag))
}

/// Resolves a `ParameterId` for `Individual`, allocating a new slot if needed.
pub(crate) fn resolve_or_alloc_param(params: &mut Vec<Scalar>, id: ParameterId) -> ParameterId {
    let idx = *id as usize;
    if idx < params.len() {
        id
    } else {
        let new_id = ParameterId::from(params.len() as u16);
        params.push(0.0);
        new_id
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

        fn mutation_targets(root: RootId, arena: &ExprArena<()>) -> Vec<NodeId> {
            arena.walk_expr(root).unwrap().collect()
        }

        fn get_tag_for_node(_kind: NodeKind) -> () {}
    }
}
