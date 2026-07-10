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

use expreon::ops::builtin::{Add, Div, MathBaseOps, Mul, Sub};
use expreon::{
    eval::{EvalBufferStack, VectorizedEvalContext},
    gp::{
        Context, Fitness, GenerationBreeder, Genome, Individual, IntegerFitness, ScalarFitness,
        fitness::pareto_cmp,
        k_best_of, k_tournament_selection,
        mutation::{
            Mutator,
            builtin::{
                HoistMutation, InsertMutation, ParamJitter, PointMutation, SubtreeMutation,
                TerminalMutation,
            },
        },
        subtree::{GrowSubtreeConfig, TreeGenConfig, TreeMethod, gen_tree},
    },
    prelude::*,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A simple genome with 2D inputs
#[derive(Clone)]
struct Scalar2DGenome;

impl Genome for Scalar2DGenome {
    type Tag = ();
    const INPUT_DIM: u16 = 2;
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
    inputs: ArrayView2<Scalar>,
    targets: ArrayView1<Scalar>,
    stack: &mut EvalBufferStack,
) -> f32 {
    let root_node = match eval.arena.get_root(ind.root) {
        Some(n) => n,
        None => return f32::MAX,
    };

    let batch = inputs.nrows();
    let params_arr = make_params_array(&ind.parameters, batch);

    let preds_buf = eval.eval_batch(root_node, inputs, params_arr.view(), stack);
    let err: f32 = preds_buf
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / batch as f32;
    stack.reclaim(preds_buf);

    if err.is_nan() || err.is_infinite() {
        f32::MAX
    } else {
        err
    }
}

/// Compute total expression AST tree depth
fn tree_depth(root: RootId, arena: &ExprArena<()>) -> usize {
    fn depth_of(id: NodeId, arena: &ExprArena<()>) -> usize {
        let node = arena.get_node(id).unwrap();
        match node.kind {
            NodeKind::Variable(_) | NodeKind::Parameter(_) => 0,
            NodeKind::Unary { value, .. } => 1 + depth_of(value, arena),
            NodeKind::Binary { left, right, .. } => {
                1 + depth_of(left, arena).max(depth_of(right, arena))
            }
        }
    }
    arena.get_root(root).map_or(0, |n| depth_of(n, arena))
}

/// Total number of nodes in the expression tree.
fn node_count(root: RootId, arena: &ExprArena<()>) -> usize {
    arena
        .get_root(root)
        .map_or(0, |n| arena.iter_expr_nodes(n).count())
}

const POP_SIZE: usize = 10_000;
const GEN_COUNT: usize = 250;
const K: usize = 15; // tournament size
const MSE_TARGET: f32 = 1e-9; // constant by which to stop accounting for MSE and look at other pareto criterias

// Score every unscored individual in the current generation
fn evaluate_population(
    ctx: &mut Context<Scalar2DGenome, RegressionFitness>,
    inputs: &ArrayView2<Scalar>,
    targets: &ArrayView1<Scalar>,
    stack: &mut EvalBufferStack,
) {
    let arena = &ctx.current.arena;
    let eval = VectorizedEvalContext::new(arena, &ctx.operations);
    ctx.current.population.score_unscored(|ind| {
        let raw = mse(ind, &eval, *inputs, targets.view(), stack);
        let accuracy = if raw < MSE_TARGET { 0.0 } else { raw };
        RegressionFitness {
            mse: accuracy.into(),
            nodes: node_count(ind.root, arena).into(),
            depth: tree_depth(ind.root, arena).into(),
        }
    });
}

/// Format the AST tree expression for display
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

fn make_params_array(params: &[Scalar], batch: usize) -> Array2<Scalar> {
    let n = params.len().max(1); // eval_batch expects at least 1 param column
    Array2::from_shape_fn((batch, n), |(_, j)| params.get(j).copied().unwrap_or(0.0))
}

fn main() {
    // Training data: 20 points, x ∈ [−5, 5], y ∈ [−4, 4], target = 2x² + 4y + 3.
    const N: usize = 64;
    let xs: Vec<Scalar> = (0..N)
        .map(|i| -5.0 + 10.0 * i as f32 / (N - 1) as f32)
        .collect();
    let ys: Vec<Scalar> = (0..N)
        .map(|i| -4.0 + 8.0 * i as f32 / (N - 1) as f32)
        .collect();
    let targets: Vec<Scalar> = (0..N)
        .map(|i| 2.0 * xs[i] * xs[i] + 4.0 * ys[i] + 3.0)
        .collect();
    let inputs = Array2::from_shape_fn((N, 2), |(i, j)| if j == 0 { xs[i] } else { ys[i] });
    let targets = Array1::from_vec(targets);

    // Scratch buffers for the vectorized evaluator, reused across every
    // individual and every generation (all training batches are size N).
    let mut stack = EvalBufferStack::new(N);

    let tree_cfg = TreeGenConfig {
        p_terminal: 0.3,
        const_range: (-5.0, 5.0),
    };

    let mut gp_context: Context<Scalar2DGenome, RegressionFitness> = Context::new(build_op_table());
    let mut rng = StdRng::seed_from_u64(42);

    let mut mutator: Mutator<Scalar2DGenome> = Mutator::new();
    mutator
        .add(
            0.5,
            SubtreeMutation {
                grow: GrowSubtreeConfig {
                    max_depth: 3,
                    tuning: TreeGenConfig {
                        p_terminal: 0.4,
                        const_range: (-5.0, 5.0),
                    },
                },
            },
        )
        .add(0.3, PointMutation)
        .add(0.2, ParamJitter { stddev: 0.5 })
        .add(0.2, HoistMutation)
        .add(
            0.2,
            InsertMutation {
                const_range: (-5.0, 5.0),
                p_binary: 0.5,
            },
        )
        .add(
            0.2,
            TerminalMutation {
                const_range: (-5.0, 5.0),
                p_variable: 0.5,
            },
        );

    // Initial population in the current generation (`finish` inserts).
    for _ in 0..POP_SIZE {
        let mut b = gp_context.builder(&mut rng);
        let root = gen_tree::<Scalar2DGenome, _>(&mut b, &tree_cfg, TreeMethod::Grow, 4);
        b.finish(root);
    }

    println!("Symbolic regression example");
    println!("Target expression: 2x² + 4y + 3 (2-D input)");
    println!("pop={POP_SIZE}  gens={GEN_COUNT}  tournament k={K}\n");

    for generation in 0..GEN_COUNT {
        evaluate_population(&mut gp_context, &inputs.view(), &targets.view(), &mut stack);
        let best = k_best_of(&gp_context.current.population, 1)
            .into_iter()
            .next()
            .unwrap();

        let best_root_node_id = gp_context
            .current
            .arena
            .get_root(best.individual.root)
            .expect("no individual with mathing root id");

        let best_fitness = best.fitness.unwrap_or(RegressionFitness::WORST);

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

        // Build the next generation, then advance. `Breeding::new` takes the
        // context fields directly so the outstanding `best` reference into
        // `current` stays valid while `next` is borrowed mutably.
        {
            let mut breeding = GenerationBreeder::new(
                &gp_context.current,
                &mut gp_context.next,
                &gp_context.operations,
            );
            // Elitism: carry the best individual over unchanged (keeps its fitness).
            breeding.copy_individual_over(best);
            for _ in 1..POP_SIZE {
                let parent = k_tournament_selection(&breeding.source.population, K, &mut rng);
                breeding.breed(parent, &mutator, &mut rng);
            }
        }

        gp_context.advance();
    }

    // The last generation produced by the loop is unscored after the final
    // advance; score it before reporting the overall best.
    evaluate_population(&mut gp_context, &inputs.view(), &targets.view(), &mut stack);
    let best = k_best_of(&gp_context.current.population, 1)
        .into_iter()
        .next()
        .unwrap();

    let arena = &gp_context.current.arena;
    let root_node = arena.get_root(best.individual.root).unwrap();
    let n_nodes = arena.iter_expr_nodes(root_node).count();
    let depth = tree_depth(best.individual.root, arena);
    let eval = VectorizedEvalContext::new(arena, &gp_context.operations);
    let raw_mse = mse(
        &best.individual,
        &eval,
        inputs.view(),
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
    let params_arr = make_params_array(&best.individual.parameters, 5);
    // Different batch size (5 test points vs. N training points) needs its
    // own stack;
    let mut test_stack = EvalBufferStack::new(5);
    let preds_buf = eval.eval_batch(
        root_node,
        test_inputs.view(),
        params_arr.view(),
        &mut test_stack,
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
    test_stack.reclaim(preds_buf);
}
