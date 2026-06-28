pub mod builtin;
pub mod mutator;

pub use mutator::{Mutator, apply_mutation};

use rand::RngCore;

use crate::types::Scalar;
use crate::{
    ast::{ExprArena, ExprNode, NodeKind},
    ops::OperationTable,
    types::{NodeId, ParameterId},
};

use super::Genome;
use super::builder::NodeBuilder;

/// A pluggable tree mutation.
///
/// Implementations answer two questions: can this mutation act on `node`, and
/// what subtree should replace `target` in the offspring? Unchanged parts of
/// the tree are copied verbatim via `MutationContext::copy_subtree`.
pub trait Mutation<G: Genome>: 'static {
    /// Whether this mutation can act on a node of the given kind.
    fn applies_to(&self, kind: NodeKind) -> bool;

    /// Emit a replacement subtree for `target` into the dest arena and return
    /// its root `NodeId`. Use `ctx.copy_subtree` to carry over unchanged
    /// children of an emitted node.
    ///
    /// Return `None` for a *passthrough* mutation — one that changes only side
    /// data such as the parameter vector and leaves the tree structure intact.
    /// The engine then copies the original `target` subtree verbatim
    /// (preserving tags), so passthrough mutations need not touch the arena.
    fn apply(&self, target: NodeId, ctx: &mut MutationContext<'_, G>) -> Option<NodeId>;
}

/// Mutation context passed to a mutation
pub struct MutationContext<'a, G: Genome> {
    pub(crate) source: &'a ExprArena<G::Tag>,
    pub(crate) dest: &'a mut ExprArena<G::Tag>,
    pub(crate) ops: &'a OperationTable,
    pub(crate) rng: &'a mut dyn RngCore,
    pub(crate) params: &'a mut Vec<Scalar>,
}

impl<'a, G: Genome> MutationContext<'a, G> {
    pub const fn new(
        source: &'a ExprArena<G::Tag>,
        ops: &'a OperationTable,
        rng: &'a mut dyn RngCore,
        dest: &'a mut ExprArena<G::Tag>,
        params: &'a mut Vec<Scalar>,
    ) -> Self {
        Self {
            source,
            ops,
            rng,
            dest,
            params,
        }
    }

    /// Verbatim deep copy of a source subtree, preserving each node's tag.
    pub fn copy_subtree(&mut self, src: NodeId) -> NodeId {
        // Bind the &ExprArena as a plain reference so we can re-borrow `self` mutably.
        let arena: &ExprArena<G::Tag> = self.source;
        let node = arena
            .get_node(src)
            .expect("invalid source node in copy_subtree");
        let kind = node.kind;
        let tag = node.tag.clone();
        let new_kind = match kind {
            NodeKind::Unary { value, op } => {
                let v = self.copy_subtree(value);
                NodeKind::Unary { value: v, op }
            }
            NodeKind::Binary { left, right, op } => {
                let l = self.copy_subtree(left);
                let r = self.copy_subtree(right);
                NodeKind::Binary {
                    left: l,
                    right: r,
                    op,
                }
            }
            leaf => leaf,
        };

        self.dest.add(ExprNode::new(new_kind, tag))
    }

    /// Read the current value of a parameter.
    pub fn get_parameter(&self, id: ParameterId) -> Scalar {
        self.params[*id as usize]
    }

    /// Write a value back to an existing parameter slot.
    pub fn set_parameter(&mut self, id: ParameterId, value: Scalar) {
        self.params[*id as usize] = value;
    }
}

impl<'a, G: Genome> NodeBuilder<G> for MutationContext<'a, G> {
    fn rng(&mut self) -> &mut dyn RngCore {
        self.rng
    }

    fn ops(&self) -> &OperationTable {
        self.ops
    }

    fn emit(&mut self, node: ExprNode<G::Tag>) -> NodeId {
        self.dest.add(node)
    }

    /// Allocate a new parameter slot initialised to `value` and return its id.
    fn new_parameter(&mut self, value: Scalar) -> ParameterId {
        let id = ParameterId::from(self.params.len() as u16);
        self.params.push(value);
        id
    }
}
