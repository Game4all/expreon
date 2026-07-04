use std::ops::{Deref, DerefMut};

use crate::gp::{Genome, Individual};

/// Fitness score of an individual. Lower or higher being "better" is left to the
/// caller — the GP types store fitness but stay neutral about ordering.
pub type Fitness = f32;

/// The individuals making up a generation.
///
/// A thin wrapper over `Vec<Individual<G>>` that derefs to a slice, so
/// iteration, indexing, `len`, etc. come for free and yield `&Individual`
/// directly. Fitness lives inside each [`Individual`] (`None` until scored).
pub struct Population<G: Genome>(Vec<Individual<G>>);

impl<G: Genome> Population<G> {
    /// An empty population.
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    /// Removes all individuals, resetting the buffer.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Inserts an individual and returns a mutable reference to it.
    pub fn insert(&mut self, individual: Individual<G>) -> &mut Individual<G> {
        self.0.push(individual);
        self.0.last_mut().unwrap()
    }

    /// Iterates over the individuals that have not been scored yet.
    pub fn iter_unscored(&self) -> impl Iterator<Item = &Individual<G>> {
        self.0.iter().filter(|ind| ind.fitness.is_none())
    }

    /// Scores every unscored individual with `f` and records the result.
    ///
    /// Convenient when the scoring function borrows only the arena (not the
    /// population).
    pub fn score_unscored(&mut self, mut f: impl FnMut(&Individual<G>) -> Fitness) {
        for ind in &mut self.0 {
            if ind.fitness.is_none() {
                ind.fitness = Some(f(ind));
            }
        }
    }
}

impl<G: Genome> Deref for Population<G> {
    type Target = [Individual<G>];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<G: Genome> DerefMut for Population<G> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<G: Genome> Default for Population<G> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::test_genome::TestSimpleGenome;
    use expreon_ast::RootId;

    fn ind(root: usize) -> Individual<TestSimpleGenome> {
        Individual::new(RootId::from(root), Vec::new())
    }

    #[test]
    fn insert_appends_and_returns_ref() {
        let mut pop: Population<TestSimpleGenome> = Population::new();
        pop.insert(ind(0));
        pop.insert(ind(1)).fitness = Some(1.5);
        assert_eq!(pop.len(), 2);
        assert_eq!(pop[0].fitness, None);
        assert_eq!(pop[1].fitness, Some(1.5));
    }

    #[test]
    fn iter_unscored_yields_only_unscored() {
        let mut pop: Population<TestSimpleGenome> = Population::new();
        pop.insert(ind(0)).fitness = Some(1.0);
        pop.insert(ind(1));
        let unscored: Vec<RootId> = pop.iter_unscored().map(|i| i.root).collect();
        assert_eq!(unscored, vec![RootId::from(1usize)]);
    }

    #[test]
    fn score_unscored_only_touches_unscored() {
        let mut pop: Population<TestSimpleGenome> = Population::new();
        pop.insert(ind(0)).fitness = Some(1.0);
        pop.insert(ind(1));
        pop.score_unscored(|_| 9.0);
        assert_eq!(pop[0].fitness, Some(1.0)); // untouched
        assert_eq!(pop[1].fitness, Some(9.0));
    }

    #[test]
    fn clear_resets_population() {
        let mut pop: Population<TestSimpleGenome> = Population::new();
        pop.insert(ind(0)).fitness = Some(1.0);
        pop.clear();
        assert!(pop.is_empty());
        assert_eq!(pop.iter().count(), 0);
    }
}
