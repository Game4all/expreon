//! 2-D symbolic-regression GP loop that rediscovers `2x*x + 4y + 3`.
//!
//! Uses tournament selection with elitism and six built-in mutations.
//! Individuals carry their own fitness and live in the [`Context`]'s two
//! generations (`current` / `next`); selection and breeding pass around plain
//! `&Individual` references. Training data is 20 noise-free samples over
//! x ∈ [-5, 5], y ∈ [-4, 4].
//!
//! Run with:
//!   cargo run --release --example gp_2x_plus_4

use std::cmp::Ordering;

use expreon::gp::prelude::*;
use expreon::gp::{
    ArrayDataset, IntegerFitness, ScalarFitness,
    fitness::pareto_cmp,
    mutation::builtin::{
        HoistMutation, InsertMutation, ParamJitter, ParamResample, PointMutation, SubtreeMutation,
        TerminalTypeSwap,
    },
    subtree::{GrowSubtreeConfig, TreeGenConfig, TreeMethod, gen_tree},
};
use expreon::ops::builtin::{Add, Div, MathBaseOps, Mul, Sub};
use expreon::{
    eval::{EvalBufferStack, VectorizedEvalContext},
    prelude::*,
};
use ndarray::{Array1, Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A simple genome with 2D inputs
#[derive(Clone)]
struct Scalar2DGenome;

impl Genome for Scalar2DGenome {
    type Tag = ();
    fn get_tag_for_node(_: NodeKind) -> () {}
}

/// Multi-objective fitness for the search: mean-squared error (accuracy) plus
/// two integer size objectives, node count and tree depth. Compared by Pareto
/// dominance across all three; genuine trade-offs are broken in favour of the
/// lower MSE.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RegressionFitness {
    mse: ScalarFitness,
    nodes: IntegerFitness,
    depth: IntegerFitness,
}

impl RegressionFitness {
    /// The worst possible fitness: every objective at its worst.
    const WORST: Self = Self {
        mse: ScalarFitness::WORST,
        nodes: IntegerFitness::WORST,
        depth: IntegerFitness::WORST,
    };
}

impl Fitness for RegressionFitness {
    fn quality_cmp(&self, other: &Self) -> Option<Ordering> {
        let pareto = pareto_cmp([
            self.mse.quality_cmp(&other.mse),
            self.nodes.quality_cmp(&other.nodes),
            self.depth.quality_cmp(&other.depth),
        ]);
        // Break genuine trade-offs by MSE (the accuracy objective) so the
        // ordering is total; `quality_cmp` on scalars never returns `None`.
        Some(pareto.unwrap_or_else(|| self.mse.quality_cmp(&other.mse).unwrap()))
    }
}

/// Build the operation table
fn build_op_table() -> OperationTable {
    let mut b = OperationTableBuilder::new();
    b.register_set::<MathBaseOps>();
    b.build()
}

/// MSE of `ind` on `inputs` / `targets`.
fn mse(
    ind: &Individual<Scalar2DGenome>,
    eval: &VectorizedEvalContext<'_, '_, ()>,
    dataset: &ArrayDataset,
    targets: ArrayView1<Scalar>,
    stack: &mut EvalBufferStack,
) -> f32 {
    let Some(root_node) = eval.arena.get_root(ind.root) else {
        return f32::MAX; // no root: a dead individual, scored as the worst possible.
    };
    let inputs = dataset.inputs();
    let preds_buf = eval.eval_batch(root_node, inputs, &ind.parameters, stack);

    let batch = inputs.nrows();
    let err: f32 = preds_buf
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / batch as f32;
    stack.reclaim(preds_buf);
    err
}

const POP_SIZE: usize = 10_000;
const GEN_COUNT: usize = 250;
const K: usize = 15; // tournament size
const MSE_TARGET: f32 = 1e-9; // constant by which to stop accounting for MSE and look at other pareto criterias
const CONST_RANGE: (Scalar, Scalar) = (-5.0, 5.0); // range shared by every mutation/generator that draws a random constant.

// Hard structural cap on tree depth, enforced via a `GatedGenerationBreeder`
// hook at breed time.
const MAX_DEPTH: usize = 12;
// Per-offspring-slot retries against the depth hook before giving up and
// copying the parent through unchanged for that slot.
const MAX_BREED_ATTEMPTS: usize = 5;

// Score every unscored individual in the current generation
fn evaluate_population(
    ctx: &mut Context<Scalar2DGenome, RegressionFitness>,
    dataset: &ArrayDataset,
    targets: &ArrayView1<Scalar>,
    stack: &mut EvalBufferStack,
) {
    let arena = &ctx.current.arena;
    let eval = VectorizedEvalContext::new(arena, &ctx.operations);
    ctx.current.population.score_unscored(|ind| {
        let raw = mse(ind, &eval, dataset, targets.view(), stack);
        let accuracy = if raw < MSE_TARGET { 0.0 } else { raw };
        RegressionFitness {
            mse: accuracy.into(),
            nodes: arena.node_count_of_root(ind.root).into(),
            depth: arena.depth_of_root(ind.root).into(),
        }
    });
}

/// Format the AST tree expression for display.
fn fmt_node(
    node_id: NodeId,
    arena: &ExprArena<()>,
    params: &[Scalar],
    ops: &OperationTable,
) -> String {
    let node = arena.get_node(node_id).unwrap();
    match node.kind {
        NodeKind::Variable(v) => format!("x{}", *v),
        NodeKind::Parameter(p) => format!("{:.4}", params[*p as usize]),
        NodeKind::Unary { value, op } => {
            let inner = fmt_node(value, arena, params, ops);
            let name = ops.lookup_by_id(op).map(|m| m.name).unwrap_or("op?");
            format!("{name}({inner})")
        }
        NodeKind::Binary { left, right, op } => {
            let l = fmt_node(left, arena, params, ops);
            let r = fmt_node(right, arena, params, ops);
            match ops.lookup_by_id(op).map(|m| m.name).unwrap_or("") {
                Add::NAME => format!("({l} + {r})"),
                Sub::NAME => format!("({l} - {r})"),
                Mul::NAME => format!("({l} * {r})"),
                Div::NAME => format!("({l} / {r})"),
                name => format!("{name}({l}, {r})"),
            }
        }
    }
}

fn main() {
    // Training data: 20 points, x ∈ [−5, 5], y ∈ [−4, 4], target = 2x² + 4y + 3.
    const N: usize = 128;
    let xs: Vec<Scalar> = (0..N)
        .map(|i| -5.0 + 10.0 * i as f32 / (N - 1) as f32)
        .collect();
    let ys: Vec<Scalar> = (0..N)
        .map(|i| -4.0 + 8.0 * i as f32 / (N - 1) as f32)
        .collect();
    let targets: Vec<Scalar> = (0..N)
        .map(|i| 2.0 * xs[i] * xs[i] + 4.0 * ys[i] + 3.0)
        .collect();
    let dataset = ArrayDataset::new(Array2::from_shape_fn((N, 2), |(i, j)| {
        if j == 0 { xs[i] } else { ys[i] }
    }));
    let targets = Array1::from_vec(targets);

    // Scratch buffers for the vectorized evaluator.
    let mut stack = EvalBufferStack::new(N);

    let tree_cfg = TreeGenConfig {
        const_range: CONST_RANGE,
        ..Default::default()
    };

    let mut gp_context: Context<Scalar2DGenome, RegressionFitness> =
        Context::new(build_op_table(), &dataset);
    let mut rng = StdRng::seed_from_u64(42);

    let mut mutator: Mutator<Scalar2DGenome> = Mutator::new();
    mutator
        .add(
            0.5,
            SubtreeMutation {
                grow: GrowSubtreeConfig {
                    tuning: TreeGenConfig {
                        p_terminal: 0.4,
                        const_range: CONST_RANGE,
                    },
                    ..Default::default()
                },
            },
        )
        .add(0.3, PointMutation)
        .add(0.2, ParamJitter { stddev: 0.5 })
        .add(0.2, HoistMutation)
        .add(
            0.2,
            InsertMutation {
                const_range: CONST_RANGE,
                p_binary: 0.5,
            },
        )
        .add(
            0.1,
            TerminalTypeSwap {
                const_range: CONST_RANGE,
            },
        )
        .add(
            0.1,
            ParamResample {
                const_range: CONST_RANGE,
            },
        );

    // Initial population in the current generation (`finish` inserts).
    for _ in 0..POP_SIZE {
        let mut b = gp_context.builder(&mut rng);
        let root = gen_tree(&mut b, &tree_cfg, TreeMethod::Grow, 4);
        b.finish(root);
    }

    println!("Symbolic regression example");
    println!("Target expression: 2x² + 4y + 3 (2-D input)");
    println!("pop={POP_SIZE}  gens={GEN_COUNT}  tournament k={K}\n");

    while gp_context.generation() < GEN_COUNT {
        evaluate_population(&mut gp_context, &dataset, &targets.view(), &mut stack);
        let best: Scored<Scalar2DGenome, RegressionFitness> =
            k_best_of(&gp_context.current.population, 1)
                .first()
                .cloned()
                .unwrap()
                .clone();

        let best_root_node_id = gp_context
            .current
            .arena
            .get_root(best.individual.root)
            .expect("no individual with mathing root id");

        let best_fitness = best.fitness.unwrap_or(RegressionFitness::WORST);
        let generation = gp_context.generation();

        if generation % 10 == 0 {
            println!(
                "gen {:3}: MSE={:.4e}  nodes={}  depth={} | expression={}",
                generation,
                best_fitness.mse.0,
                best_fitness.nodes.0,
                best_fitness.depth.0,
                fmt_node(
                    best_root_node_id,
                    &gp_context.current.arena,
                    &best.individual.parameters,
                    &gp_context.operations
                )
            );
        }

        // Build the new generation through a gated breeder that enforces the hard depth limit.
        {
            let mut breeding = gp_context.gated_breeder(
                |ind: &Individual<Scalar2DGenome>, arena: &ExprArena<()>| {
                    arena.depth_of_root(ind.root) <= MAX_DEPTH
                },
            );

            // Elitism: carry the best individual over unchanged (keeps its
            // fitness).
            breeding
                .copy_individual_over(&best)
                .expect("elite individual unexpectedly rejected by depth hook");

            for _ in 1..POP_SIZE {
                let parent = k_tournament_selection(&breeding.source.population, K, &mut rng);

                // Attempt to breed a new offspring from `parent` up to `MAX_BREED_ATTEMPTS` times, until the depth hook accepts it. If all attempts fail, copy the parent over unchanged.
                let bred = (0..MAX_BREED_ATTEMPTS)
                    .any(|_| mutator.breed(&mut breeding, parent, &mut rng).is_some());
                if !bred {
                    breeding.copy_individual_over(parent);
                }
            }
        }

        gp_context.advance();
    }

    // The last generation produced by the loop is unscored after the final
    // advance; score it before reporting the overall best.
    evaluate_population(&mut gp_context, &dataset, &targets.view(), &mut stack);
    let best = k_best_of(&gp_context.current.population, 1)
        .into_iter()
        .next()
        .unwrap();

    let arena = &gp_context.current.arena;
    let root_node = arena.get_root(best.individual.root).unwrap();
    let n_nodes = arena.node_count_of_root(best.individual.root);
    let depth = arena.depth_of_root(best.individual.root);
    let eval = VectorizedEvalContext::new(arena, &gp_context.operations);
    let raw_mse = mse(
        &best.individual,
        &eval,
        &dataset,
        targets.view(),
        &mut stack,
    );
    println!(
        "\nBest individual: MSE={raw_mse:.4e}  depth={depth}  nodes={n_nodes}  params={:.4?}",
        best.individual.parameters
    );
    println!(
        "Expression: {}",
        fmt_node(
            root_node,
            arena,
            &best.individual.parameters,
            &gp_context.operations
        )
    );

    let test_pts: [(f32, f32); 5] = [
        (-4.0, -3.0),
        (-2.0, 0.0),
        (0.0, 2.0),
        (2.0, -1.0),
        (4.0, 3.0),
    ];
    let test_inputs =
        Array2::from_shape_fn(
            (5, 2),
            |(i, j)| {
                if j == 0 { test_pts[i].0 } else { test_pts[i].1 }
            },
        );

    let preds_buf = eval.eval_batch(
        root_node,
        test_inputs.view(),
        &best.individual.parameters,
        &mut stack,
    );

    println!("\nTest predictions (target = 2x² + 4y + 3):");
    println!(
        "  {:>5}  {:>5}  {:>9}  {:>10}  {:>10}",
        "x", "y", "target", "predicted", "error"
    );
    for (i, &(x, y)) in test_pts.iter().enumerate() {
        let t = 2.0 * x * x + 4.0 * y + 3.0;
        println!(
            "  {:>5.1}  {:>5.1}  {:>9.2}  {:>10.4}  {:>10.2e}",
            x,
            y,
            t,
            preds_buf[i],
            preds_buf[i] - t
        );
    }
    stack.reclaim(preds_buf);
}
