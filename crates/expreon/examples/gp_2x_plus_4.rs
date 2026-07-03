//! 2-D symbolic-regression GP loop that rediscovers `2x*x + 4y + 3`.
//!
//! Uses tournament selection and three built-in mutations (subtree replacement,
//! point (operator swap), and parameter jitter). Training data is 20 noise-free
//! samples over x ∈ [-5, 5], y ∈ [-4, 4].
//!
//! Run with:
//!   cargo run --example gp_2x_plus_4

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use expreon::ops::Operation;
use expreon::ops::builtin::{Add, Div, MathBaseOps, Mul, Sub};
use expreon::{
    ast::{ExprArena, NodeKind},
    eval::EvalContext,
    gp::{
        Context, Genome, Individual,
        mutation::{
            Mutator,
            builtin::{
                HoistMutation, InsertMutation, ParamJitter, PointMutation, SubtreeMutation,
                TerminalMutation,
            },
        },
        subtree::{GrowSubtreeConfig, TreeGenConfig, TreeMethod, gen_tree},
    },
    ops::{OperationTable, OperationTableBuilder},
    types::{NodeId, RootId, Scalar},
};

/// A simple genome with 2D inputs
#[derive(Clone)]
struct Scalar2DGenome;

impl Genome for Scalar2DGenome {
    type Tag = ();
    const INPUT_DIM: u16 = 2;
    fn get_tag_for_node(_: NodeKind) -> () {}
}

/// Build the operation table
fn build_op_table() -> OperationTable {
    let mut b = OperationTableBuilder::new();
    b.register_set::<MathBaseOps>();
    b.build()
}

/// k-tournament: returns the index of the lowest-fitness candidate among k draws.
fn tournament(fitness: &[f32], k: usize, rng: &mut dyn RngCore) -> usize {
    let n = fitness.len();
    let mut best = rng.random_range(0..n);
    for _ in 1..k {
        let idx = rng.random_range(0..n);
        if fitness[idx] < fitness[best] {
            best = idx;
        }
    }
    best
}

/// MSE of `ind` on `inputs` / `targets`.
fn mse(
    ind: &Individual<Scalar2DGenome>,
    eval: &EvalContext<'_, '_, ()>,
    inputs: ArrayView2<Scalar>,
    targets: ArrayView1<Scalar>,
) -> f32 {
    let root_node = match eval.arena.get_root(ind.root) {
        Some(n) => n,
        None => return f32::MAX,
    };

    let batch = inputs.nrows();
    let params_arr = make_params_array(&ind.parameters, batch);

    let preds = eval.eval_batch(root_node, inputs, params_arr.view());
    let err: f32 = preds
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum::<f32>()
        / batch as f32;

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
    const N: usize = 20;
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

    const POP: usize = 10_000;
    const GENS: usize = 100;
    const K: usize = 10; // tournament size
    const EPSILON: f32 = 1e-12;
    const DEPTH_PENALTY: f32 = 0.01; // added to fitness per unit of tree depth

    let tree_cfg = TreeGenConfig {
        p_terminal: 0.3,
        const_range: (-5.0, 5.0),
    };

    let mut gp_context: Context<Scalar2DGenome> = Context::new(build_op_table());
    let mut rng = StdRng::seed_from_u64(42);

    // Initial population in the source arena.
    let mut population: Vec<Individual<Scalar2DGenome>> = (0..POP)
        .map(|_| {
            let mut b = gp_context.builder(&mut rng);
            let root = gen_tree::<Scalar2DGenome, _>(&mut b, &tree_cfg, TreeMethod::Grow, 4);
            let (ind, _) = b.finish(root);
            ind
        })
        .collect();

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

    // Penalized fitness = MSE + DEPTH_PENALTY * depth.
    // Raw MSE is tracked separately for convergence checks and display.
    let compute_fitness =
        |pop: &[Individual<Scalar2DGenome>], ctx: &Context<Scalar2DGenome>| -> Vec<f32> {
            let arena = ctx.source_arena();
            let eval = EvalContext::new(arena, &ctx.operations);
            pop.iter()
                .map(|ind| {
                    let raw = mse(ind, &eval, inputs.view(), targets.view());
                    if raw == f32::MAX {
                        return f32::MAX;
                    }
                    raw + DEPTH_PENALTY * tree_depth(ind.root, arena) as f32
                })
                .collect()
        };

    let mut fitness = compute_fitness(&population, &gp_context);
    let best_of = |f: &[f32]| {
        f.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap()
    };

    println!("Symbolic regression — target: 2x² + 4y + 3 (2-D input)");
    println!("pop={POP}  gens={GENS}  tournament k={K}\n");

    for generation in 0..GENS {
        let best_idx = best_of(&fitness);
        let best_fitness = fitness[best_idx];

        // Compute raw MSE and depth only for the best individual (cheap).
        let (raw_mse, best_depth) = {
            let arena = gp_context.source_arena();
            let eval = EvalContext::new(arena, &gp_context.operations);
            let ind = &population[best_idx];
            (
                mse(ind, &eval, inputs.view(), targets.view()),
                tree_depth(ind.root, arena),
            )
        };

        if generation % 10 == 0 || raw_mse < EPSILON {
            println!(
                "gen {:3}: fitness={:.4e}  MSE={:.4e}  depth={}",
                generation, best_fitness, raw_mse, best_depth
            );
        }

        if raw_mse < EPSILON {
            println!("\nConverged at generation {generation}.");
            break;
        }

        // Build next generation into the dest arena, then swap.
        let mut next: Vec<Individual<Scalar2DGenome>> = Vec::with_capacity(POP);
        {
            let (source, dest, ops) = gp_context.borrow_parts();
            for _ in 0..POP {
                let parent = &population[tournament(&fitness, K, &mut rng)];
                let child = mutator
                    .mutate(parent, source, dest, ops, &mut rng)
                    .expect("mutate returned None — no applicable mutations");
                next.push(child);
            }
        } // buffers dropped here, releasing the mutable borrow on gp_context

        gp_context.swap();
        population = next;
        fitness = compute_fitness(&population, &gp_context);
    }

    let best_idx = best_of(&fitness);

    let best = &population[best_idx];
    let arena = gp_context.source_arena();
    let root_node = arena.get_root(best.root).unwrap();
    let n_nodes = arena.iter_expr_nodes(root_node).count();
    let depth = tree_depth(best.root, arena);
    let eval = EvalContext::new(arena, &gp_context.operations);
    let raw_mse = mse(best, &eval, inputs.view(), targets.view());
    println!(
        "\nBest individual: MSE={raw_mse:.4e}  fitness={:.4e}  depth={depth}  nodes={n_nodes}  params={:.4?}",
        fitness[best_idx], best.parameters
    );
    println!(
        "Expression: {}",
        fmt_node(root_node, arena, &best.parameters, &gp_context.operations)
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
    let params_arr = make_params_array(&best.parameters, 5);
    let preds = eval.eval_batch(root_node, test_inputs.view(), params_arr.view());

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
            preds[i],
            preds[i] - t
        );
    }
}
