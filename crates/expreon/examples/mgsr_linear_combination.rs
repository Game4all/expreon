//! Multi-gene symbolic regression (MGSR) exampple, evolveing a weighted linear combination of `M_GENES` sub-expressions ("genes"), with
//! no intercept. This examples demonstrates decoupling of structure search from the parameter search (lin reg).
//!
//! GP can only mutate the mutable gene sub-expressions, the linear combination scaffold is 'frozen' and only its weight nodes are fit through gradient descent.
//!
//! Target: `2.5 * sin(x0) - 1.8 * (x1 * x1) + 0.7 * (x0 * x1)`.
//!
//! Run with:
//!   cargo run --release --example mgsr_linear_combination

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
use expreon::ops::builtin::{Add, Mul, Sin, Sub};
use expreon::{
    eval::{Buffer, CachedEvalContext, EvalBufferStack, EvalCache},
    prelude::*,
};
use fixedbitset::FixedBitSet;
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

const M_GENES: usize = 3;
const POP_SIZE: usize = 5_000;
const GEN_COUNT: usize = 400;
const N: usize = 256;
const K: usize = 15; // tournament size
const GD_STEPS: usize = 50;
const RIDGE_LAMBDA: f64 = 1e-4;
const DEGENERATE_SCALE_EPS: f64 = 1e-6;
const MAX_GENE_DEPTH: usize = 5;
const GENE_INIT_DEPTH: usize = 3;
const MAX_BREED_ATTEMPTS: usize = 10;
const CONST_RANGE: (Scalar, Scalar) = (-3.0, 3.0);

fn build_op_table() -> OperationTable {
    let mut b = OperationTableBuilder::new();
    b.register::<Add>();
    b.register::<Sub>();
    b.register::<Mul>();
    b.register::<Sin>();
    b.build()
}

/// Tags every node of an individual's expr tree as one of three following roles:
///
/// - `Scaffold`: the `+` chain and the `*` wrapping each gene. Can't be targetted by [`MgsrGenome::mutation_targets`].
/// - `Weight`: a parameter part of the immutable scaffold defining weight for a gene whose value is fit by gradient descent, never by mutation.
/// - `Gene`: everything inside a sub-expression. Fully mutable; this is the only tag [`MgsrGenome::mutation_targets`] can target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MgsrTag {
    Scaffold,
    Weight,
    Gene,
}

#[derive(Clone)]
struct MgsrGenome;

impl Genome for MgsrGenome {
    type Tag = MgsrTag;

    fn get_tag_for_node(_kind: NodeKind) -> MgsrTag {
        MgsrTag::Gene
    }

    fn mutation_targets(root: RootId, arena: &ExprArena<MgsrTag>) -> Vec<NodeId> {
        arena
            .walk_root(root)
            .into_iter()
            .flatten()
            .filter(|&id| arena.get_node(id).unwrap().tag == MgsrTag::Gene)
            .collect()
    }
}

/// Structural information about an individual expression tree, used to identify gene roots and optimizable weigths
struct GeneLayout {
    /// Holds the parameter IDs of the individual that are optimizable
    /// through gradient descent as a bitset
    optimizable: FixedBitSet,
    /// Root node ID of each gene
    genes: [NodeId; M_GENES],
}

impl MgsrGenome {
    /// Recovers the [`GeneLayout`] of an individual's tree, by walking
    /// the frozen `Scaffold` region and retrieving the gene roots and their weight `ParameterId`s.    
    fn get_expr_gene_layout(arena: &ExprArena<MgsrTag>, root: NodeId) -> GeneLayout {
        let mut terms: Vec<(ParameterId, NodeId)> = Vec::with_capacity(M_GENES);
        let mut stack = vec![root];
        while let Some(node_id) = stack.pop() {
            let NodeKind::Binary { left, right, .. } = arena.get_node(node_id).unwrap().kind else {
                unreachable!("scaffold is assumed to be an immutable Add chain of w * gene terms");
            };
            if let NodeKind::Parameter(weight) = arena.get_node(left).unwrap().kind {
                terms.push((weight, right));
            } else {
                stack.push(right);
                stack.push(left);
            }
        }
        terms.sort_by_key(|&(weight, _)| *weight);

        let mut optimizable = FixedBitSet::new();
        for &(weight, _) in &terms {
            optimizable.grow_and_insert(*weight as usize);
        }

        GeneLayout {
            optimizable,
            genes: std::array::from_fn(|k| terms[k].1),
        }
    }

    /// Reconstructs the full parameter vector after a fit through GD by writing `fitted` back
    /// into the marked slots and leaves every unmarked slot (the gene parameters) as is
    fn scatter_optimizable_parameters(
        mask: &FixedBitSet,
        fitted: &[Scalar],
        params: &mut [Scalar],
    ) {
        for (i, &v) in mask.ones().zip(fitted) {
            params[i] = v;
        }
    }
}

/// Builds one scaffolded individual: `M_GENES` freshly-grown genes, each
/// wrapped in `weight * gene` (weight initialised to 1.0), folded into a
/// balanced `Add` chain (so every gene sits at the same distance from the
/// root, keeping the depth budget fair across genes). Returns the root
/// `NodeId`; the caller registers it as a root via [`NodeBuilder`]'s owning
/// [`IndividualBuilder`].
fn build_multi_gene_individual<B: NodeBuilder<Genome = MgsrGenome>>(
    b: &mut B,
    gene_cfg: &TreeGenConfig,
    gene_init_depth: usize,
    m_genes: usize,
    mul_op: OperationId,
    add_op: OperationId,
) -> NodeId {
    let mut mul_nodes = Vec::with_capacity(m_genes);
    for _ in 0..m_genes {
        let w_pid = b.new_parameter(1.0); // lets init the gene to 1.0 so the GD fit has a reasonable starting point.
        let w_node = b.emit(ExprNode::new_parameter(w_pid, MgsrTag::Weight));
        let gene = gen_tree(b, gene_cfg, TreeMethod::Grow, gene_init_depth);
        let mul = b.emit(ExprNode::new_binary(
            w_node,
            gene,
            mul_op,
            MgsrTag::Scaffold,
        ));
        mul_nodes.push(mul);
    }
    balanced_add(b, &mul_nodes, add_op)
}

/// Folds `nodes` into a balanced binary `Add` tree.
fn balanced_add<B: NodeBuilder<Genome = MgsrGenome>>(
    b: &mut B,
    nodes: &[NodeId],
    add_op: OperationId,
) -> NodeId {
    match nodes {
        [] => unreachable!("at least one gene is required"),
        [only] => *only,
        _ => {
            let mid = nodes.len() / 2;
            let left = balanced_add(b, &nodes[..mid], add_op);
            let right = balanced_add(b, &nodes[mid..], add_op);
            b.emit(ExprNode::new_binary(left, right, add_op, MgsrTag::Scaffold))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MgsrFitness {
    mse: ScalarFitness,
    complexity: IntegerFitness,
}

impl MgsrFitness {
    const WORST: Self = Self {
        mse: ScalarFitness::WORST,
        complexity: IntegerFitness::WORST,
    };
}

impl Fitness for MgsrFitness {
    fn quality_cmp(&self, other: &Self) -> Option<Ordering> {
        let pareto = pareto_cmp([
            self.mse.quality_cmp(&other.mse),
            self.complexity.quality_cmp(&other.complexity),
        ]);
        Some(pareto.unwrap_or_else(|| self.mse.quality_cmp(&other.mse).unwrap()))
    }
}

/// Ridge-regularized gradient descent model
///
/// Columns are standardized to unit root-mean-square before descent.
/// Because we don't have an intercept, theres no centering done on the columns.
struct RidgeRegression {
    /// Standardized coefficients (`weights[i] = raw_weight[i] * scale[i]`),
    /// updated in place by [`Self::fit`].
    weights: Vec<f64>,
    /// Per-feature RMS scale used to standardize `z`. A scale below
    /// [`DEGENERATE_SCALE_EPS`] marks a column that collapsed to (near)
    /// zero everywhere.
    scale: Vec<f64>,
    /// Standardized feature columns: `z[i][j]` is feature `i`, sample `j`.
    z: Vec<Vec<f64>>,
    /// Raw (uncentered) targets.
    y: Vec<f64>,
    lambda: f64,
}

impl RidgeRegression {
    fn new(
        columns: &[&Buffer],
        targets: ArrayView1<Scalar>,
        warm_start: &[Scalar],
        lambda: f64,
    ) -> Self {
        let n = targets.len();
        let m = columns.len();

        let mut scale = vec![0.0f64; m];
        let mut z = vec![vec![0.0f64; n]; m];
        for i in 0..m {
            let rms =
                (columns[i].iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
            scale[i] = rms;
            if rms >= DEGENERATE_SCALE_EPS {
                for j in 0..n {
                    z[i][j] = columns[i][j] as f64 / rms;
                }
            }
        }

        let weights = (0..m).map(|i| warm_start[i] as f64 * scale[i]).collect();
        let y = targets.iter().map(|&t| t as f64).collect();

        Self {
            weights,
            scale,
            z,
            y,
            lambda,
        }
    }

    /// Runs `steps` iterations of gradient descent, in place.
    fn fit(&mut self, steps: usize) {
        let m = self.weights.len();
        let n = self.y.len();
        let eta = 1.0 / (2.0 * (m as f64 + self.lambda));

        let mut residual = vec![0.0f64; n];
        for _ in 0..steps {
            for j in 0..n {
                let mut pred = 0.0;
                for i in 0..m {
                    pred += self.z[i][j] * self.weights[i];
                }
                residual[j] = pred - self.y[j];
            }
            for i in 0..m {
                let mut grad = 0.0;
                for j in 0..n {
                    grad += self.z[i][j] * residual[j];
                }
                grad = grad * 2.0 / n as f64 + 2.0 * self.lambda * self.weights[i];
                self.weights[i] -= eta * grad;
            }
        }
    }

    /// Outputs de-transformed weights to be written back into the individual's parameter slots.
    fn weights(&self) -> Vec<Scalar> {
        self.weights
            .iter()
            .zip(&self.scale)
            .map(|(&a, &s)| {
                if s >= DEGENERATE_SCALE_EPS {
                    (a / s) as Scalar
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Fits `individual`'s weights in place and returns the fitness it achieves.
fn fit_and_score(
    individual: &mut Individual<MgsrGenome>,
    arena: &ExprArena<MgsrTag>,
    ops: &OperationTable,
    dataset: &ArrayDataset,
    targets: ArrayView1<Scalar>,
    cache: &mut EvalCache,
    stack: &mut EvalBufferStack,
) -> MgsrFitness {
    let Some(root_node) = arena.get_root(individual.root) else {
        return MgsrFitness::WORST;
    };
    let layout = MgsrGenome::get_expr_gene_layout(arena, root_node);

    cache.clear();
    let cached_ctx = CachedEvalContext::new(arena, ops, |id: NodeId, a: &ExprNode<MgsrTag>| {
        a.tag == MgsrTag::Gene && layout.genes.contains(&id)
    });

    // eval the whole tree once to cache the gene columns, so the GD fit can reuse them instead of re-evaluating the whole tree on every step.
    _ = cached_ctx.eval_batch(
        root_node,
        dataset.inputs(),
        &individual.parameters,
        cache,
        stack,
    );

    let n = dataset.len();

    let mut columns: Vec<&Buffer> = Vec::with_capacity(M_GENES);
    for &gene in &layout.genes {
        let Some(col) = cache.get(gene) else {
            return MgsrFitness::WORST; // gene root wasn't reached: dead individual.
        };

        // automatically reject any individual that produces non-finite values, because that'll wreck the gradient descent
        if !col.iter().all(|v| v.is_finite()) {
            return MgsrFitness::WORST;
        }

        columns.push(col);
    }

    // only the linear combination weights are the optimizer's, the rest of the parameter vector are constants, owned by the mutator.
    // use the individuals linear combination weights as a warm start for the fit.
    let warm_start: Vec<Scalar> = layout
        .optimizable
        .ones()
        .map(|i| individual.parameters[i])
        .collect();

    let mut ridge = RidgeRegression::new(&columns, targets, &warm_start, RIDGE_LAMBDA);
    ridge.fit(GD_STEPS);

    // set the fitted weights back into the individual's parameter vector, so the final MSE is computed with the fitted weights.
    MgsrGenome::scatter_optimizable_parameters(
        &layout.optimizable,
        &ridge.weights(),
        &mut individual.parameters,
    );

    // re-eval the whole tree but with the fitted weights for the linear combination for computing MSE
    let preds = cached_ctx.eval_batch(
        root_node,
        dataset.inputs(),
        &individual.parameters,
        cache,
        stack,
    );
    let mse: f64 = preds
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let d = (p - t) as f64;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    stack.reclaim(preds);

    let complexity: usize = layout.genes.iter().map(|&g| arena.node_count(g)).sum();
    MgsrFitness {
        mse: (mse as Scalar).into(),
        complexity: complexity.into(),
    }
}

// evaluate the whole population, fitting each individual gene weights and scoring them afterwards.
// The fitted weights are written back into the individual parameter vector.
fn evaluate_population(
    ctx: &mut Context<MgsrGenome, MgsrFitness>,
    dataset: &ArrayDataset,
    targets: &ArrayView1<Scalar>,
    cache: &mut EvalCache,
    stack: &mut EvalBufferStack,
) {
    let arena = &ctx.current.arena;
    let ops = &ctx.operations;
    ctx.current.population.score_unscored_mut(|ind| {
        fit_and_score(ind, arena, ops, dataset, targets.view(), cache, stack)
    });
}

fn fmt_node<Tag: Clone>(
    node_id: NodeId,
    arena: &ExprArena<Tag>,
    params: &[Scalar],
    ops: &OperationTable,
) -> String {
    let node = arena.get_node(node_id).unwrap();
    match node.kind {
        NodeKind::Variable(v) => format!("x{}", *v),
        NodeKind::Parameter(p) => format!("{:.4}", params[*p as usize]),
        NodeKind::Unary { value, op } => {
            let inner = fmt_node(value, arena, params, ops);
            let name = ops.lookup_by_id(op).map(|m| m.name).unwrap_or("??");
            format!("{name}({inner})")
        }
        NodeKind::Binary { left, right, op } => {
            let l = fmt_node(left, arena, params, ops);
            let r = fmt_node(right, arena, params, ops);
            match ops.lookup_by_id(op).map(|m| m.name).unwrap_or("??") {
                Add::NAME => format!("{l} + {r}"),
                Mul::NAME => format!("{l} * {r}"),
                Sub::NAME => format!("{l} - {r}"),
                name => format!("{name}({l}, {r})"),
            }
        }
    }
}

fn fmt_decomposition(
    arena: &ExprArena<MgsrTag>,
    root: NodeId,
    params: &[Scalar],
    ops: &OperationTable,
) -> String {
    let layout = MgsrGenome::get_expr_gene_layout(arena, root);
    let terms: Vec<String> = layout
        .optimizable
        .ones()
        .zip(layout.genes)
        .map(|(weight, gene)| {
            format!(
                "{:.4} * ({})",
                params[weight],
                fmt_node(gene, arena, params, ops)
            )
        })
        .collect();

    format!("f(x) = {}", terms.join(" + "))
}

fn main() {
    println!("Multi-gene symbolic regression (MGSR) example");
    println!("Target expression: 2.5 * sin(x0) - 1.8 * (x1 * x1) + 0.7 * (x0 * x1)");
    println!("pop={POP_SIZE}  gens={GEN_COUNT}  genes={M_GENES}  tournament k={K}\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Training data: N points, x0/x1 independently uniform in [-5, 5] (drawn from the RNG, so the two inputs aren't perfectly correlated).
    // Target expression = 2.5*sin(x0) - 1.8*x1^2 + 0.7*x0*x1.
    let xs0: Vec<Scalar> = (0..N).map(|_| rng.random_range(-5.0..5.0)).collect();
    let xs1: Vec<Scalar> = (0..N).map(|_| rng.random_range(-5.0..5.0)).collect();
    let targets: Vec<Scalar> = (0..N)
        .map(|i| 2.5 * xs0[i].sin() - 1.8 * xs1[i] * xs1[i] + 0.7 * xs0[i] * xs1[i])
        .collect();
    let dataset = ArrayDataset::new(Array2::from_shape_fn((N, 2), |(i, j)| {
        if j == 0 { xs0[i] } else { xs1[i] }
    }));
    let targets = Array1::from_vec(targets);

    let mut gp_context: Context<MgsrGenome, MgsrFitness> = Context::new(build_op_table(), &dataset);
    let mul_op = gp_context
        .operations
        .get_id_for_op::<Mul>()
        .expect("Mul isn't registered???");

    let add_op = gp_context
        .operations
        .get_id_for_op::<Add>()
        .expect("Add isn't registered???");

    let mut cache = EvalCache::new(N);
    let mut stack = EvalBufferStack::new(N);

    let gene_cfg = TreeGenConfig {
        p_terminal: 0.4,
        const_range: CONST_RANGE,
    };

    let mut mutator: Mutator<MgsrGenome> = Mutator::new();
    mutator
        .add(
            0.5,
            SubtreeMutation {
                grow: GrowSubtreeConfig {
                    tuning: gene_cfg,
                    max_depth: MAX_GENE_DEPTH,
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

    // build the initial population of random individuals with `M_GENES` genes each, wrapped in a frozen linear combination scafofld.
    for _ in 0..POP_SIZE {
        let mut b = gp_context.builder(&mut rng);
        let root = build_multi_gene_individual(
            &mut b,
            &gene_cfg,
            GENE_INIT_DEPTH,
            M_GENES,
            mul_op,
            add_op,
        );
        b.finish(root);
    }

    while gp_context.generation() < GEN_COUNT {
        evaluate_population(
            &mut gp_context,
            &dataset,
            &targets.view(),
            &mut cache,
            &mut stack,
        );

        // grab the best individual of the current generation for reporting and elitism
        let best: Scored<MgsrGenome, MgsrFitness> = k_best_of(&gp_context.current.population, 1)
            .first()
            .cloned()
            .cloned()
            .unwrap();

        let best_root_node_id = gp_context
            .current
            .arena
            .get_root(best.individual.root)
            .expect("no individual with matching root id");

        let best_fitness = best.fitness.unwrap_or(MgsrFitness::WORST);
        let generation = gp_context.generation();

        if generation % 10 == 0 {
            println!(
                "gen {:3}: MSE={:.4e}  gene_nodes={} | {}",
                generation,
                best_fitness.mse.0,
                best_fitness.complexity.0,
                fmt_decomposition(
                    &gp_context.current.arena,
                    best_root_node_id,
                    &best.individual.parameters,
                    &gp_context.operations
                )
            );
        }

        // Build the next generation through a gated breeder enforcing a hard per-gene depth limit (so the genes don't go brrr and make the expression explode in size)
        {
            let mut breeding = gp_context.gated_breeder(
                |ind: &Individual<MgsrGenome>, arena: &ExprArena<MgsrTag>| {
                    let Some(root_node) = arena.get_root(ind.root) else {
                        return false;
                    };
                    MgsrGenome::get_expr_gene_layout(arena, root_node)
                        .genes
                        .iter()
                        .all(|&gene| arena.depth_of(gene) <= MAX_GENE_DEPTH)
                },
            );

            //carry the best individual over unchanged for elitism (keeps its fitted weights and fitness).
            breeding
                .copy_individual_over(&best)
                .expect("elite individual unexpectedly rejected by depth hook");

            for _ in 1..POP_SIZE {
                let parent = k_tournament_selection(&breeding.source.population, K, &mut rng);

                let bred = (0..MAX_BREED_ATTEMPTS)
                    .any(|_| mutator.breed(&mut breeding, parent, &mut rng).is_some());
                if !bred {
                    breeding.copy_individual_over(parent);
                }
            }
        }

        gp_context.advance();
    }

    // we need to re-score the final generation to get the fitted weights and fitness of the best individual.
    evaluate_population(
        &mut gp_context,
        &dataset,
        &targets.view(),
        &mut cache,
        &mut stack,
    );
    let best = k_best_of(&gp_context.current.population, 1)
        .into_iter()
        .next()
        .unwrap();

    let arena = &gp_context.current.arena;
    let root_node = arena.get_root(best.individual.root).unwrap();
    let layout = MgsrGenome::get_expr_gene_layout(arena, root_node);
    let total_gene_nodes: usize = layout.genes.iter().map(|&g| arena.node_count(g)).sum();
    println!(
        "\nBest individual: MSE={:.4e}  gene_nodes={total_gene_nodes}",
        best.fitness.unwrap_or(MgsrFitness::WORST).mse.0
    );
    println!(
        "Expression: {}",
        fmt_decomposition(
            arena,
            root_node,
            &best.individual.parameters,
            &gp_context.operations
        )
    );

    let test_pts: [(f32, f32); 15] = [
        (-4.0, -3.0),
        (-2.0, 0.3),
        (0.0, 2.0),
        (2.0, -1.0),
        (4.0, 3.0),
        (-5.0, -5.0),
        (-3.5, 4.2),
        (-1.0, -2.5),
        (-0.5, 0.8),
        (1.3, 3.7),
        (2.8, -4.1),
        (3.3, 1.9),
        (4.7, -2.2),
        (-4.5, 0.6),
        (5.0, 5.0),
    ];
    let test_inputs = Array2::from_shape_fn((test_pts.len(), 2), |(i, j)| {
        if j == 0 { test_pts[i].0 } else { test_pts[i].1 }
    });

    let cached_ctx =
        CachedEvalContext::new(arena, &gp_context.operations, |_: NodeId, _: &_| false);
    let preds = cached_ctx.eval_batch(
        root_node,
        test_inputs.view(),
        &best.individual.parameters,
        &mut cache,
        &mut stack,
    );

    println!("\nTest predictions (target = 2.5*sin(x0) - 1.8*x1^2 + 0.7*x0*x1):");
    println!(
        "  {:>5}  {:>5}  {:>9}  {:>10}  {:>10}",
        "x0", "x1", "target", "predicted", "error"
    );
    for (i, &(x0, x1)) in test_pts.iter().enumerate() {
        let t = 2.5 * x0.sin() - 1.8 * x1 * x1 + 0.7 * x0 * x1;
        println!(
            "  {:>5.1}  {:>5.1}  {:>9.2}  {:>10.4}  {:>10.2e}",
            x0,
            x1,
            t,
            preds[i],
            preds[i] - t
        );
    }
}
