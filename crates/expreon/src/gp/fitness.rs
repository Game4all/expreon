use std::cmp::Ordering;

use expreon_ast::Scalar;

/// Fitness metric of an individual, comparable by quality.
///
/// The ordering is a *partial* one: [`Self::quality_cmp`] may return `None`
/// when two fitnesses are genuine trade-offs, better on one criterion, worse
/// on another. Single-objective fitness scores are totally ordered and never return
/// `None`; multi-objective fitness scores generally do, which is what enables
/// Pareto-based selection.
pub trait Fitness: Clone {
    /// Orders `self` against `other` by quality, from the optimizer's view.
    ///
    /// ### Value interpretation
    /// - `Some(Greater)`: `self` is strictly better (it *dominates* `other`).
    /// - `Some(Less)`:  `self` is strictly worse (it is dominated).
    /// - `Some(Equal)`:  equally good on every criterion.
    /// - `None`:  neither dominates the other (a genuine trade-off).
    fn quality_cmp(&self, other: &Self) -> Option<Ordering>;

    /// `true` if `self` Pareto-dominates `other` (strictly better overall).
    fn dominates(&self, other: &Self) -> bool {
        self.quality_cmp(other) == Some(Ordering::Greater)
    }
}

/// A single-objective fitness backed by a float where **lower is better**:
/// a smaller value dominates. NaN (and any non-comparable result, i.e from an
/// invalid expression) is treated as the worst possible fitness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarFitness(pub f32);

impl ScalarFitness {
    pub const WORST: ScalarFitness = ScalarFitness(f32::INFINITY);

    pub const fn new(value: f32) -> Self {
        Self(value)
    }
}

impl Fitness for ScalarFitness {
    fn quality_cmp(&self, other: &Self) -> Option<Ordering> {
        let rank = |x: f32| if x.is_nan() { Scalar::INFINITY } else { x };
        rank(other.0).partial_cmp(&rank(self.0))
    }
}

impl From<f32> for ScalarFitness {
    fn from(v: f32) -> Self {
        Self(v)
    }
}

impl From<ScalarFitness> for f32 {
    fn from(f: ScalarFitness) -> Self {
        f.0
    }
}

/// A multi-objective fitness: `N` criteria, **all lower-is-better**, compared
/// by Pareto dominance. One fitness dominates another when it is at least as
/// good on every criterion and strictly better on at least one; when each is
/// better on some criterion they are an incomparable trade-off
/// ([`Fitness::quality_cmp`] returns `None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParetoFitness<const N: usize>(pub [ScalarFitness; N]);

impl<const N: usize> ParetoFitness<N> {
    /// The worst possible fitness: every criterion at its worst.
    pub const WORST: Self = ParetoFitness([ScalarFitness::WORST; N]);
}

impl<const N: usize> Fitness for ParetoFitness<N> {
    fn quality_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut acc = Ordering::Equal;
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            match a.quality_cmp(b)? {
                // `?`: an incomparable component makes the whole vector so.
                Ordering::Equal => {}
                ord if acc == Ordering::Equal => acc = ord,
                ord if ord != acc => return None, // better on one axis, worse on another
                _ => {}
            }
        }
        Some(acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_value_dominates_higher() {
        assert!(ScalarFitness(1.0).dominates(&ScalarFitness(2.0)));
        assert!(!ScalarFitness(2.0).dominates(&ScalarFitness(1.0)));
    }

    #[test]
    fn equal_values_are_equal_quality() {
        let a = ScalarFitness(3.0);
        let b = ScalarFitness(3.0);
        assert_eq!(a.quality_cmp(&b), Some(Ordering::Equal));
        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn nan_is_worst() {
        let nan = ScalarFitness(Scalar::NAN);
        let finite = ScalarFitness(1.0);
        assert!(finite.dominates(&nan));
        assert!(!nan.dominates(&finite));
        assert_eq!(nan.quality_cmp(&finite), Some(Ordering::Less));
    }

    #[test]
    fn worst_is_dominated_by_any_finite_fitness() {
        assert!(ScalarFitness(1000.0).dominates(&ScalarFitness::WORST));
        assert!(!ScalarFitness::WORST.dominates(&ScalarFitness(1000.0)));
    }

    fn pareto<const N: usize>(vals: [f32; N]) -> ParetoFitness<N> {
        ParetoFitness(vals.map(ScalarFitness))
    }

    #[test]
    fn pareto_dominates_when_better_or_equal_on_all_and_strictly_better_on_one() {
        // Equal on criterion 0, strictly better on 1 and 2.
        let a = pareto([1.0, 2.0, 3.0]);
        let b = pareto([1.0, 5.0, 4.0]);
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
        assert_eq!(a.quality_cmp(&b), Some(Ordering::Greater));
    }

    #[test]
    fn pareto_trade_off_is_incomparable() {
        // Better on the first axis, worse on the second: neither dominates.
        let a = pareto([1.0, 2.0]);
        let b = pareto([2.0, 1.0]);
        assert_eq!(a.quality_cmp(&b), None);
        assert!(!a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn pareto_all_equal_is_equal() {
        let a = pareto([1.0, 2.0, 3.0]);
        let b = pareto([1.0, 2.0, 3.0]);
        assert_eq!(a.quality_cmp(&b), Some(Ordering::Equal));
        assert!(!a.dominates(&b));
    }

    #[test]
    fn pareto_worst_is_dominated_by_any_finite_fitness() {
        let finite = pareto([1000.0, 1000.0, 1000.0]);
        assert!(finite.dominates(&ParetoFitness::<3>::WORST));
        assert!(!ParetoFitness::<3>::WORST.dominates(&finite));
    }
}
