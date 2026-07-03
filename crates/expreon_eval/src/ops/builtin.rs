use crate::ops::{Arity, Operation, OperationSet, OperationTableBuilder};
use crate::types::Scalar;

macro_rules! impl_op {
    ($name:ident, $str_name:literal, $str_id:literal, $arity:expr, $forward:expr) => {
        pub struct $name;
        impl Operation for $name {
            const NAME: &'static str = $str_name;
            const ID: &'static str = $str_id;
            const ARITY: Arity = $arity;
            fn forward(input: &[Scalar]) -> Scalar {
                ($forward)(input)
            }
        }
    };
}

// Arithmetic
impl_op!(Add, "add", "add", Arity::Binary, |i: &[Scalar]| i[0] + i[1]);
impl_op!(Sub, "sub", "sub", Arity::Binary, |i: &[Scalar]| i[0] - i[1]);
impl_op!(Mul, "mul", "mul", Arity::Binary, |i: &[Scalar]| i[0] * i[1]);
impl_op!(Div, "div", "div", Arity::Binary, |i: &[Scalar]| i[0] / i[1]);

// Trigonometric
impl_op!(Sin, "sin", "sin", Arity::Unary, |i: &[Scalar]| i[0].sin());
impl_op!(Cos, "cos", "cos", Arity::Unary, |i: &[Scalar]| i[0].cos());
impl_op!(Tan, "tan", "tan", Arity::Unary, |i: &[Scalar]| i[0].tan());

// Exponential and logarithmic
impl_op!(Exp, "exp", "exp", Arity::Unary, |i: &[Scalar]| i[0].exp());
impl_op!(Ln, "ln", "ln", Arity::Unary, |i: &[Scalar]| i[0].ln());
impl_op!(Log2, "log2", "log2", Arity::Unary, |i: &[Scalar]| i[0]
    .log2());
impl_op!(Log10, "log10", "log10", Arity::Unary, |i: &[Scalar]| i[0]
    .log10());

/// Arithmetic operations: add, sub, mul, div.
pub struct MathBaseOps;

impl OperationSet for MathBaseOps {
    fn register(op: &mut OperationTableBuilder) {
        op.register::<Add>();
        op.register::<Sub>();
        op.register::<Mul>();
        op.register::<Div>();
    }
}

/// Trigonometric operations: sin, cos, tan.
pub struct MathTrigOps;

impl OperationSet for MathTrigOps {
    fn register(op: &mut OperationTableBuilder) {
        op.register::<Sin>();
        op.register::<Cos>();
        op.register::<Tan>();
    }
}

/// Exponential and logarithmic operations: exp, ln, log2, log10.
pub struct MathExpLogOps;

impl OperationSet for MathExpLogOps {
    fn register(op: &mut OperationTableBuilder) {
        op.register::<Exp>();
        op.register::<Ln>();
        op.register::<Log2>();
        op.register::<Log10>();
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::{E, FRAC_PI_2, FRAC_PI_4, PI};

    use crate::ops::builtin::{
        Add, Cos, Div, Exp, Ln, Log2, Log10, MathBaseOps, MathExpLogOps, MathTrigOps, Mul, Sin,
        Sub, Tan,
    };
    use crate::ops::{OperationTable, OperationTableBuilder};

    fn math_ops() -> OperationTable {
        let mut b = OperationTableBuilder::new();
        b.register_set::<MathBaseOps>();
        b.build()
    }

    fn trig_ops() -> OperationTable {
        let mut b = OperationTableBuilder::new();
        b.register_set::<MathTrigOps>();
        b.build()
    }

    fn explog_ops() -> OperationTable {
        let mut b = OperationTableBuilder::new();
        b.register_set::<MathExpLogOps>();
        b.build()
    }

    #[test]
    fn test_add() {
        let ops = math_ops();
        assert_eq!(ops.lookup::<Add>().unwrap().call(&[3.0, 4.0]), 7.0);
        assert_eq!(ops.lookup::<Add>().unwrap().call(&[-1.0, 1.0]), 0.0);
        assert_eq!(ops.lookup::<Add>().unwrap().call(&[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_sub() {
        let ops = math_ops();
        assert_eq!(ops.lookup::<Sub>().unwrap().call(&[10.0, 3.0]), 7.0);
        assert_eq!(ops.lookup::<Sub>().unwrap().call(&[3.0, 10.0]), -7.0);
        assert_eq!(ops.lookup::<Sub>().unwrap().call(&[5.0, 5.0]), 0.0);
    }

    #[test]
    fn test_mul() {
        let ops = math_ops();
        assert_eq!(ops.lookup::<Mul>().unwrap().call(&[3.0, 4.0]), 12.0);
        assert_eq!(ops.lookup::<Mul>().unwrap().call(&[-2.0, 5.0]), -10.0);
        assert_eq!(ops.lookup::<Mul>().unwrap().call(&[0.0, 99.0]), 0.0);
    }

    #[test]
    fn test_div() {
        let ops = math_ops();
        assert_eq!(ops.lookup::<Div>().unwrap().call(&[10.0, 2.0]), 5.0);
        assert_eq!(ops.lookup::<Div>().unwrap().call(&[1.0, 4.0]), 0.25);
        assert_eq!(ops.lookup::<Div>().unwrap().call(&[0.0, 5.0]), 0.0);
    }

    #[test]
    fn test_sin() {
        let ops = trig_ops();
        assert!(ops.lookup::<Sin>().unwrap().call(&[0.0]).abs() < 1e-6);
        assert!((ops.lookup::<Sin>().unwrap().call(&[FRAC_PI_2]) - 1.0).abs() < 1e-6);
        assert!(ops.lookup::<Sin>().unwrap().call(&[PI]).abs() < 1e-6);
    }

    #[test]
    fn test_cos() {
        let ops = trig_ops();
        assert!((ops.lookup::<Cos>().unwrap().call(&[0.0]) - 1.0).abs() < 1e-6);
        assert!(ops.lookup::<Cos>().unwrap().call(&[FRAC_PI_2]).abs() < 1e-6);
        assert!((ops.lookup::<Cos>().unwrap().call(&[PI]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tan() {
        let ops = trig_ops();
        assert!(ops.lookup::<Tan>().unwrap().call(&[0.0]).abs() < 1e-6);
        assert!((ops.lookup::<Tan>().unwrap().call(&[FRAC_PI_4]) - 1.0).abs() < 1e-6);
        assert!((ops.lookup::<Tan>().unwrap().call(&[-FRAC_PI_4]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exp() {
        let ops = explog_ops();
        assert!((ops.lookup::<Exp>().unwrap().call(&[0.0]) - 1.0).abs() < 1e-6);
        assert!((ops.lookup::<Exp>().unwrap().call(&[1.0]) - E).abs() < 1e-6);
        assert!((ops.lookup::<Exp>().unwrap().call(&[-1.0]) - 1.0 / E).abs() < 1e-6);
    }

    #[test]
    fn test_ln() {
        let ops = explog_ops();
        assert!(ops.lookup::<Ln>().unwrap().call(&[1.0]).abs() < 1e-6);
        assert!((ops.lookup::<Ln>().unwrap().call(&[E]) - 1.0).abs() < 1e-6);
        assert!((ops.lookup::<Ln>().unwrap().call(&[E * E]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_log2() {
        let ops = explog_ops();
        assert!(ops.lookup::<Log2>().unwrap().call(&[1.0]).abs() < 1e-6);
        assert!((ops.lookup::<Log2>().unwrap().call(&[2.0]) - 1.0).abs() < 1e-6);
        assert!((ops.lookup::<Log2>().unwrap().call(&[8.0]) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_log10() {
        let ops = explog_ops();
        assert!(ops.lookup::<Log10>().unwrap().call(&[1.0]).abs() < 1e-6);
        assert!((ops.lookup::<Log10>().unwrap().call(&[10.0]) - 1.0).abs() < 1e-6);
        assert!((ops.lookup::<Log10>().unwrap().call(&[100.0]) - 2.0).abs() < 1e-6);
    }
}
