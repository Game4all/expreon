use std::collections::{HashMap, hash_map::Entry as HashMapEntry};

use ndarray::{ArrayView1, ArrayViewMut1};

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

    /// Evaluates the given inputs with the current operation.
    /// Input arity is at most `Binary` (2 args).
    fn forward(input: &[Scalar]) -> Scalar;

    /// Evaluates a whole batch at once, writing one result per element into
    /// `out`. `inputs` holds one column view per operand (length == arity);
    /// every input view and `out` share the same length (the batch size).
    ///
    /// The default implementation calls [`Self::forward`] per element;
    /// override this for a vectorized/SIMD kernel.
    /// Input arity is at most `Binary` (2 args).
    fn vectorized_forward(inputs: &[ArrayView1<Scalar>], mut out: ArrayViewMut1<Scalar>) {
        let arity = inputs.len();
        for i in 0..out.len() {
            let mut args = [0.0 as Scalar; 2];
            for a in 0..arity {
                args[a] = inputs[a][i];
            }
            out[i] = Self::forward(&args[..arity]);
        }
    }
}

/// Metadata about an operation
pub struct OpMetadata {
    /// Human-redable name for this operation
    pub name: &'static str,
    /// String based unique ID for this operation
    pub id: &'static str,
    /// Arity of the operation
    pub arity: Arity,

    /// Pointer to the forward pass implementation of the operation
    forward_pass: fn(&[Scalar]) -> Scalar,
    /// Pointer to the vectorized forward pass implementation of the operation
    vectorized_forward_pass: fn(&[ArrayView1<Scalar>], ArrayViewMut1<Scalar>),
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
            vectorized_forward_pass: Op::vectorized_forward,
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

    // Looks up the operation ID for a registered operation.
    pub fn get_id_for_op<Op: Operation>(&self) -> Option<OperationId> {
        self.lookup
            .get(Op::ID)
            .map(|x| OperationId::from(*x as u16))
    }

    /// Looks up operation metadata from an Operation itself
    pub fn lookup<Op: Operation>(&self) -> Option<&OpMetadata> {
        self.lookup.get(Op::ID).and_then(|x| self.ops.get(*x))
    }

    /// Returns an iterator over the IDs of all registered unary operations.
    pub fn iter_unary_ops(&self) -> impl ExactSizeIterator<Item = OperationId> {
        self.unary_ops.iter().copied()
    }

    /// Returns an iterator over the IDs of all registered binary operations.
    pub fn iter_binary_ops(&self) -> impl ExactSizeIterator<Item = OperationId> {
        self.binary_ops.iter().copied()
    }
}

impl OpMetadata {
    /// Executes the operation forward pass.
    pub fn call(&self, args: &[Scalar]) -> Scalar {
        (self.forward_pass)(args)
    }

    /// Executes the operation's vectorized forward pass, writing one result
    /// per element into `out`.
    pub fn call_vectorized(&self, inputs: &[ArrayView1<Scalar>], out: ArrayViewMut1<Scalar>) {
        (self.vectorized_forward_pass)(inputs, out)
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

    #[test]
    fn forward_vectorized_default_matches_forward_elementwise() {
        use ndarray::{Array1, arr1};

        let mut b = OperationTableBuilder::new();
        b.register::<TestAdd>();
        let table = b.build();

        let meta = table.lookup::<TestAdd>().unwrap();

        let lhs = arr1(&[1.0, 2.0, 3.0]);
        let rhs = arr1(&[10.0, 20.0, 30.0]);
        let mut out = Array1::zeros(3);

        meta.call_vectorized(&[lhs.view(), rhs.view()], out.view_mut());

        assert_eq!(out, arr1(&[11.0, 22.0, 33.0]));
        for i in 0..3 {
            assert_eq!(out[i], meta.call(&[lhs[i], rhs[i]]));
        }
    }
}
