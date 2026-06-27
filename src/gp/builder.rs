use rand::{Rng, RngCore};

use crate::{
    ast::ExprNode,
    ops::OperationTable,
    types::{NodeId, OperationId, ParameterId, Scalar},
};

use super::Genome;

/// The minimal node-construction surface required to build expression trees.
/// Provides methods to emit nodes in an arena and helpers to create new parameters.
pub trait NodeBuilder<G: Genome> {
    /// Returns the underlying RNG.
    fn rng(&mut self) -> &mut dyn RngCore;

    /// Returns the operation table.
    fn ops(&self) -> &OperationTable;

    /// Emit a node into the arena. The caller controls the tag on the node.
    fn emit(&mut self, node: ExprNode<G::Tag>) -> NodeId;

    /// Allocate a new parameter slot initialised to `value` and return its id.
    fn new_parameter(&mut self, value: Scalar) -> ParameterId;

    /// Pick a random unary operator from the operation table and return its ID.
    fn pick_random_unary_op(&mut self) -> OperationId {
        let n = self.ops().iter_unary_ops().len();
        assert!(n > 0, "no unary ops registered");
        let idx = self.rng().random_range(0..n);
        self.ops().iter_unary_ops().nth(idx).unwrap()
    }

    /// Pick a random binary operator from the operation table and return its ID.
    fn pick_random_binary_op(&mut self) -> OperationId {
        let n = self.ops().iter_binary_ops().len();
        assert!(n > 0, "no binary ops registered");
        let idx = self.rng().random_range(0..n);
        self.ops().iter_binary_ops().nth(idx).unwrap()
    }
}
