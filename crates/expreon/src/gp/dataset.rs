use ndarray::{Array2, ArrayView2};

use expreon_ast::Scalar;

/// A source of input samples for expression evaluation.
///
/// This is the authority on how many input variables expressions built from a
/// genome may reference: valid [`VariableId`](expreon_ast::VariableId)s are
/// `0..input_dim()`. Targets (if any) live outside this trait — the GP core
/// stays objective-agnostic, and a fitness function is free to pull them from
/// wherever it likes.
pub trait Dataset {
    /// Maximum input size: the number of input variables (columns) samples
    /// carry. This value is cached by users of the trait **so the call should be deterministic**.
    fn input_dim(&self) -> u16;

    /// Number of samples (rows).
    fn len(&self) -> usize;

    /// `true` if this dataset has no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The samples, shaped `[len(), input_dim()]` — the layout
    /// [`VectorizedEvalContext::eval_batch`](expreon_eval::vectorized::VectorizedEvalContext::eval_batch)
    /// and [`EagerEvalContext`](expreon_eval::eval::EagerEvalContext) expect.
    fn inputs(&self) -> ArrayView2<'_, Scalar>;
}

/// A [`Dataset`] backed by an owned, dense `[samples, input_dim]` array.
pub struct ArrayDataset {
    inputs: Array2<Scalar>,
}

impl ArrayDataset {
    /// Wraps `inputs`, shaped `[samples, input_dim]`.
    ///
    /// # Panics
    ///
    /// Panics if `inputs` has more columns than fit in a `u16` — the width of
    /// [`VariableId`](expreon_ast::VariableId).
    pub fn new(inputs: Array2<Scalar>) -> Self {
        assert!(
            inputs.ncols() <= usize::from(u16::MAX),
            "dataset has {} input columns, more than the {} a VariableId can address",
            inputs.ncols(),
            u16::MAX
        );
        Self { inputs }
    }
}

impl Dataset for ArrayDataset {
    fn input_dim(&self) -> u16 {
        self.inputs.ncols() as u16
    }

    fn len(&self) -> usize {
        self.inputs.nrows()
    }

    fn inputs(&self) -> ArrayView2<'_, Scalar> {
        self.inputs.view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_dataset_reports_shape() {
        let data = ArrayDataset::new(Array2::zeros((5, 3)));
        assert_eq!(data.input_dim(), 3);
        assert_eq!(data.len(), 5);
        assert!(!data.is_empty());
    }

    #[test]
    fn array_dataset_empty_when_no_rows() {
        let data = ArrayDataset::new(Array2::zeros((0, 3)));
        assert!(data.is_empty());
    }

    #[test]
    fn array_dataset_inputs_view_matches_shape() {
        let data = ArrayDataset::new(Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as Scalar));
        let view = data.inputs();
        assert_eq!(view.nrows(), 2);
        assert_eq!(view.ncols(), 2);
        assert_eq!(view[[1, 1]], 2.0);
    }

    #[test]
    #[should_panic(expected = "more than the")]
    fn array_dataset_rejects_too_many_columns() {
        // A width beyond u16::MAX would silently truncate in `input_dim`'s
        // `as u16` cast; the constructor must catch it instead.
        let too_wide = Array2::<Scalar>::zeros((1, usize::from(u16::MAX) + 1));
        ArrayDataset::new(too_wide);
    }
}
