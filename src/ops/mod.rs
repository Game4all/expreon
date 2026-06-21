use std::collections::{HashMap, hash_map::Entry as HashMapEntry};

pub mod builtin;

use crate::types::{OperationId, Scalar};

/// Represents the arity of an operation.
pub enum Arity {
    Unary,
    Binary,
}

/// Base trait for all Operations
pub trait Operation: 'static {
    const NAME: &'static str;
    const ID: &'static str;
    const ARITY: Arity;

    /// Evaluates the given inputs with the current operation
    fn forward(input: &[Scalar]) -> Scalar;
}

/// Metadata about an operation
pub struct OpMetadata {
    /// Human-redable name for this operation
    name: &'static str,
    /// String based unique ID for this operation
    id: &'static str,
    /// Arity of the operation
    arity: Arity,

    /// Pointer to the forward pass implementation of the operation
    forward_pass: fn(&[Scalar]) -> Scalar,
}

/// Builder struct to register all operation and operation sets
pub struct OperationTableBuilder {
    pub(crate) ops: Vec<OpMetadata>,
    pub(crate) lookup: HashMap<&'static str, usize>,
}

/// An immutable lookup table for operations
/// Provides functions to look operations up.
pub struct OperationTable {
    pub(crate) ops: Vec<OpMetadata>,
    pub(crate) lookup: HashMap<&'static str, usize>,

    unary_ops: Vec<OperationId>,
    binary_ops: Vec<OperationId>,
}

/// Base traits for operation sets, which register sets of operations at a time.
pub trait OperationSet: Send + Sync + 'static {
    /// Registers all operations in the provided builder.
    fn register(op: &mut OperationTableBuilder);
}

impl OperationTableBuilder {
    pub fn new() -> Self {
        Self {
            ops: Default::default(),
            lookup: Default::default(),
        }
    }

    /// Registers a new operation into the builder
    pub fn register<Op: Operation>(&mut self) {
        let index = self.ops.len();
        match self.lookup.entry(Op::ID) {
            HashMapEntry::Vacant(entry) => {
                entry.insert(index);
            }
            _ => panic!("Operation with name {} is already registered.", Op::ID),
        }
        self.ops.push(OpMetadata {
            name: Op::NAME,
            id: Op::ID,
            arity: Op::ARITY,
            forward_pass: Op::forward,
        });
    }

    /// Registers a set of operations into the builder
    pub fn register_set<Set: OperationSet>(&mut self) {
        Set::register(self);
    }

    /// Consumes the builder and returns a final immutable lookup table
    pub fn build(self) -> OperationTable {
        let mut unary_ops = Vec::new();
        let mut binary_ops = Vec::new();
        for (index, meta) in self.ops.iter().enumerate() {
            let op_id = OperationId::from(index as u16);
            match meta.arity {
                Arity::Unary => unary_ops.push(op_id),
                Arity::Binary => binary_ops.push(op_id),
            }
        }
        OperationTable {
            lookup: self.lookup,
            ops: self.ops,
            unary_ops,
            binary_ops,
        }
    }
}

impl OperationTable {
    // Looks up operation metadata from an operation Id.
    pub fn lookup_by_id(&self, id: OperationId) -> Option<&OpMetadata> {
        let operation_index = u16::from(id);
        self.ops.get(operation_index as usize)
    }

    /// Looks up operation metadata from an Operation itself
    pub fn lookup<Op: Operation>(&self) -> Option<&OpMetadata> {
        self.lookup.get(Op::ID).and_then(|x| self.ops.get(*x))
    }

    /// Returns an iterator over the IDs of all registered unary operations.
    pub fn iter_unary(&self) -> impl ExactSizeIterator<Item = OperationId> {
        self.unary_ops.iter().copied()
    }

    /// Returns an iterator over the IDs of all registered binary operations.
    pub fn iter_binary(&self) -> impl ExactSizeIterator<Item = OperationId> {
        self.binary_ops.iter().copied()
    }
}

impl OpMetadata {
    /// Executes the operation forward pass.
    pub fn call(&self, args: &[Scalar]) -> Scalar {
        (self.forward_pass)(args)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::Entry;

    use crate::{
        ops::{Arity, Operation, OperationSet, OperationTableBuilder},
        types::Scalar,
    };

    struct TestAdd;

    impl Operation for TestAdd {
        const NAME: &'static str = "add";
        const ID: &'static str = "test_add";
        const ARITY: Arity = Arity::Binary;

        fn forward(input: &[Scalar]) -> Scalar {
            return input[0] + input[1];
        }
    }

    struct TestOpSet;

    impl OperationSet for TestOpSet {
        fn register(op: &mut OperationTableBuilder) {
            op.register::<TestAdd>();
        }
    }

    #[test]
    pub fn test_op_set_builder_works() {
        let mut builder = OperationTableBuilder::new();
        builder.register::<TestAdd>();

        assert_eq!(builder.ops.len(), 1);
        assert!(matches!(
            builder.lookup.entry("test_add"),
            Entry::Occupied(_)
        ));
    }

    #[test]
    #[should_panic]
    pub fn test_registering_same_op_twice_fails() {
        let mut builder = OperationTableBuilder::new();
        builder.register::<TestAdd>();
        builder.register::<TestAdd>();
    }

    #[test]
    pub fn test_registering_operation_set() {
        let mut builder = OperationTableBuilder::new();
        builder.register_set::<TestOpSet>();

        assert_eq!(builder.ops.len(), 1);
        assert!(matches!(
            builder.lookup.entry("test_add"),
            Entry::Occupied(_)
        ));
    }
}
