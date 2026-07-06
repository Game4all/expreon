use std::ops::{Deref, DerefMut};

use crate::gp::{Fitness, Genome, Individual};

/// An [`Individual`] together with its (optional) fitness, as stored in a
/// [`Population`]. Fitness is `None` until scored.
pub struct Scored<G: Genome, F: Fitness> {
    pub individual: Individual<G>,
    pub fitness: Option<F>,
}

impl<G: Genome, F: Fitness> Scored<G, F> {
    /// Wraps an individual as unscored.
    pub fn new(individual: Individual<G>) -> Self {
        Self {
            individual,
            fitness: None,
        }
    }
}

/// The individuals making up a generation, each paired with its fitness.
///
/// A thin wrapper over `Vec<Scored<G, F>>` that derefs to a slice, so
/// iteration, indexing, `len`, etc. come for free and yield `&Scored`
/// directly.
pub struct Population<G: Genome, F: Fitness>(Vec<Scored<G, F>>);

impl<G: Genome, F: Fitness> Population<G, F> {
    /// An empty population.
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    /// Removes all individuals, resetting the buffer.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Inserts an (unscored) individual and returns a mutable reference to its
    /// [`Scored`] slot.
    pub fn insert(&mut self, individual: Individual<G>) -> &mut Scored<G, F> {
        self.0.push(Scored::new(individual));
        self.0.last_mut().unwrap()
    }

    /// Iterates over the individuals that have not been scored yet.
    pub fn iter_unscored(&self) -> impl Iterator<Item = &Scored<G, F>> {
        self.0.iter().filter(|s| s.fitness.is_none())
    }

    /// Scores every unscored individual with `f` and records the result.
    ///
    /// The scoring function sees only the genetic material (`&Individual`),
    /// keeping it independent of the fitness type. Convenient when the scoring
    /// function borrows only the arena (not the population).
    pub fn score_unscored(&mut self, mut f: impl FnMut(&Individual<G>) -> F) {
        for s in &mut self.0 {
            if s.fitness.is_none() {
                s.fitness = Some(f(&s.individual));
            }
        }
    }
}

impl<G: Genome, F: Fitness> Deref for Population<G, F> {
    type Target = [Scored<G, F>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<G: Genome, F: Fitness> DerefMut for Population<G, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<G: Genome, F: Fitness> Default for Population<G, F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::ScalarFitness;
    use crate::gp::test_genome::TestSimpleGenome;
    use expreon_ast::RootId;

    fn ind(root: usize) -> Individual<TestSimpleGenome> {
        Individual::new(RootId::from(root), Vec::new())
    }

    #[test]
    fn insert_appends_and_returns_ref() {
        let mut pop: Population<TestSimpleGenome, ScalarFitness> = Population::new();
        pop.insert(ind(0));
        pop.insert(ind(1)).fitness = Some(ScalarFitness(1.5));
        assert_eq!(pop.len(), 2);
        assert_eq!(pop[0].fitness, None);
        assert_eq!(pop[1].fitness, Some(ScalarFitness(1.5)));
    }

    #[test]
    fn iter_unscored_yields_only_unscored() {
        let mut pop: Population<TestSimpleGenome, ScalarFitness> = Population::new();
        pop.insert(ind(0)).fitness = Some(ScalarFitness(1.0));
        pop.insert(ind(1));
        let unscored: Vec<RootId> = pop.iter_unscored().map(|s| s.individual.root).collect();
        assert_eq!(unscored, vec![RootId::from(1usize)]);
    }

    #[test]
    fn score_unscored_only_touches_unscored() {
        let mut pop: Population<TestSimpleGenome, ScalarFitness> = Population::new();
        pop.insert(ind(0)).fitness = Some(ScalarFitness(1.0));
        pop.insert(ind(1));
        pop.score_unscored(|_| ScalarFitness(9.0));
        assert_eq!(pop[0].fitness, Some(ScalarFitness(1.0))); // untouched
        assert_eq!(pop[1].fitness, Some(ScalarFitness(9.0)));
    }

    #[test]
    fn clear_resets_population() {
        let mut pop: Population<TestSimpleGenome, ScalarFitness> = Population::new();
        pop.insert(ind(0)).fitness = Some(ScalarFitness(1.0));
        pop.clear();
        assert!(pop.is_empty());
        assert_eq!(pop.iter().count(), 0);
    }
}
