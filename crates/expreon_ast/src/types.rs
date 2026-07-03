//! Base types used around the crate
use derive_more::with_trait::{Debug, Deref, Display, From, Index, Into};

/// Base scalar type
pub type Scalar = f32;

/// Identifier for an expression root in an arena.
#[derive(From, Into, Display, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RootId(usize);

/// Identifier for a node in an arena.
#[derive(From, Into, Display, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NodeId(usize);

/// Identifier for a user variable.
#[derive(From, Into, Display, Debug, Deref, Clone, Copy, PartialEq, PartialOrd)]
pub struct VariableId(u16);

/// Identifier for a constant or optimizable parameter.
#[derive(From, Into, Display, Debug, Deref, Clone, Copy, PartialEq, PartialOrd)]
pub struct ParameterId(u16);

/// Identifier for an operation.
#[derive(From, Into, Display, Debug, Index, Clone, Copy, PartialEq, PartialOrd)]
pub struct OperationId(u16);
