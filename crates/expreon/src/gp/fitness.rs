use std::cmp::Ordering;

use rand::{Rng, RngCore};

use expreon_ast::Scalar;

use crate::gp::{Genome, Population, Scored};

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

/// Default comparator ranking [`Scored`] individuals by [`Fitness::quality_cmp`]:
/// an unscored individual is worst, and a genuine trade-off (`None`) compares
/// equal. Used by [`k_best_of`] and [`k_tournament_selection`]; pass a custom
/// comparator to [`k_best_of_with_quality`] / [`k_tournament_selection_with_quality`]
/// instead when trade-offs need a tie-break (e.g. a secondary objective).
fn compare_by_fitness<G, F>(a: &Scored<G, F>, b: &Scored<G, F>) -> Ordering
where
    G: Genome,
    F: Fitness,
{
    match (&a.fitness, &b.fitness) {
        (Some(fa), Some(fb)) => fa.quality_cmp(fb).unwrap_or(Ordering::Equal),
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (None, None) => Ordering::Equal,
    }
}

/// Returns the `k` best individuals of `pop`, best first, ranked by
/// [`Fitness::quality_cmp`] (unscored individuals are treated as worst; a
/// genuine trade-off compares equal). See [`k_best_of_with_quality`] to
/// supply a custom comparator.
#[inline(always)]
pub fn k_best_of<G, F>(pop: &Population<G, F>, k: usize) -> Vec<&Scored<G, F>>
where
    G: Genome,
    F: Fitness,
{
    k_best_of_with_comparator(pop, k, compare_by_fitness)
}

/// Returns the `k` best individuals of `pop`, best first, ranked by `compare`
/// (`Greater` = the first argument is better). `k` is clamped to the population
/// size; an empty population (or `k == 0`) yields an empty vec.
pub fn k_best_of_with_comparator<'p, G, F>(
    pop: &'p Population<G, F>,
    k: usize,
    compare: impl Fn(&Scored<G, F>, &Scored<G, F>) -> Ordering,
) -> Vec<&'p Scored<G, F>>
where
    G: Genome,
    F: Fitness,
{
    let mut ranked: Vec<&Scored<G, F>> = pop.iter().collect();
    let k = k.min(ranked.len());
    if k == 0 {
        return Vec::new();
    }
    ranked.select_nth_unstable_by(k - 1, |a, b| compare(b, a));
    ranked.truncate(k);
    ranked.sort_by(|a, b| compare(b, a));
    ranked
}

/// k-tournament selection: draws `k` random individuals (with replacement)
/// from `pop` and returns a reference to the best, ranked by
/// [`Fitness::quality_cmp`] (unscored individuals are treated as worst; a
/// genuine trade-off compares equal). See
/// [`k_tournament_selection_with_quality`] to supply a custom comparator.
///
/// Panics if `pop` is empty.
pub fn k_tournament_selection<'p, G, F>(
    pop: &'p Population<G, F>,
    k: usize,
    rng: &mut dyn RngCore,
) -> &'p Scored<G, F>
where
    G: Genome,
    F: Fitness,
{
    k_tournament_selection_with_comparator(pop, k, rng, compare_by_fitness)
}

/// k-tournament selection: draws `k` random individuals (with replacement) from
/// `pop` and returns a reference to the best, ranked by `compare` (`Greater` =
/// the first argument is better).
///
/// Panics if `pop` is empty.
pub fn k_tournament_selection_with_comparator<'p, G, F>(
    pop: &'p Population<G, F>,
    k: usize,
    rng: &mut dyn RngCore,
    compare: impl Fn(&Scored<G, F>, &Scored<G, F>) -> Ordering,
) -> &'p Scored<G, F>
where
    G: Genome,
    F: Fitness,
{
    let n = pop.len();
    let mut best = &pop[rng.random_range(0..n)];
    for _ in 1..k {
        let candidate = &pop[rng.random_range(0..n)];
        if compare(candidate, best) == Ordering::Greater {
            best = candidate;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::test_genome::TestSimpleGenome;
    use expreon_ast::RootId;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn ind(root: usize) -> crate::gp::Individual<TestSimpleGenome> {
        crate::gp::Individual::new(RootId::from(root), Vec::new())
    }

    fn quality(
        a: &Scored<TestSimpleGenome, ScalarFitness>,
        b: &Scored<TestSimpleGenome, ScalarFitness>,
    ) -> Ordering {
        a.fitness.unwrap().quality_cmp(&b.fitness.unwrap()).unwrap()
    }

    #[test]
    fn k_best_with_quality_returns_best_first_and_clamps() {
        let mut pop: Population<TestSimpleGenome, ScalarFitness> = Population::new();
        pop.insert(ind(0)).fitness = Some(ScalarFitness(3.0));
        pop.insert(ind(1)).fitness = Some(ScalarFitness(1.0));
        pop.insert(ind(2)).fitness = Some(ScalarFitness(2.0));

        let top2 = k_best_of_with_comparator(&pop, 2, quality);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].fitness, Some(ScalarFitness(1.0)));
        assert_eq!(top2[1].fitness, Some(ScalarFitness(2.0)));

        let clamped = k_best_of_with_comparator(&pop, 10, quality);
        assert_eq!(clamped.len(), 3);

        let empty = k_best_of_with_comparator(&pop, 0, quality);
        assert!(empty.is_empty());
    }

    #[test]
    fn k_best_of_with_quality_of_empty_population_is_empty() {
        let pop: Population<TestSimpleGenome, ScalarFitness> = Population::new();
        assert!(k_best_of_with_comparator(&pop, 3, quality).is_empty());
    }

    #[test]
    fn tournament_with_quality_finds_the_dominant_individual() {
        let mut pop: Population<TestSimpleGenome, ScalarFitness> = Population::new();
        for i in 0..10 {
            pop.insert(ind(i)).fitness = Some(ScalarFitness(i as f32));
        }
        let mut rng = StdRng::seed_from_u64(7);
        // Enough draws that, with a fixed seed, the lone best (fitness 0.0) is
        // certain to be sampled at least once.
        let winner = k_tournament_selection_with_comparator(&pop, 200, &mut rng, quality);
        assert_eq!(winner.fitness, Some(ScalarFitness(0.0)));
    }

    #[test]
    fn k_best_of_uses_default_fitness_ordering_and_treats_unscored_as_worst() {
        let mut pop: Population<TestSimpleGenome, ScalarFitness> = Population::new();
        pop.insert(ind(0)).fitness = Some(ScalarFitness(3.0));
        pop.insert(ind(1)).fitness = Some(ScalarFitness(1.0));
        pop.insert(ind(2)); // unscored: worse than any scored individual

        let top2 = k_best_of(&pop, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].fitness, Some(ScalarFitness(1.0)));
        assert_eq!(top2[1].fitness, Some(ScalarFitness(3.0)));
    }

    #[test]
    fn k_tournament_selection_uses_default_fitness_ordering() {
        let mut pop: Population<TestSimpleGenome, ScalarFitness> = Population::new();
        for i in 0..10 {
            pop.insert(ind(i)).fitness = Some(ScalarFitness(i as f32));
        }
        let mut rng = StdRng::seed_from_u64(7);
        let winner = k_tournament_selection(&pop, 200, &mut rng);
        assert_eq!(winner.fitness, Some(ScalarFitness(0.0)));
    }

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
