//! Demonstrates manual population initialization using [`Context::builder`] and
//! [`gen_tree`], then evaluates each individual on a single input sample.

use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use symbolic_rs::{
    ast::NodeKind,
    eval::EvalContext,
    gp::subtree::{TreeGenConfig, TreeMethod, gen_tree},
    gp::{Context, Genome, Population},
    ops::OperationTableBuilder,
    ops::builtin::MathBaseOps,
};

/// Minimal genome: no tags, all nodes are mutable.
#[derive(Clone)]
struct SimpleGenome;

impl Genome for SimpleGenome {
    type Tag = ();
    const INPUT_DIM: u16 = 2;

    fn get_tag_for_node(_kind: NodeKind) -> () {}
}

fn main() {
    let mut ops_builder = OperationTableBuilder::new();
    ops_builder.register_set::<MathBaseOps>();
    let ops = ops_builder.build();

    let mut ctx: Context<SimpleGenome> = Context::new(ops);
    let mut rng = StdRng::seed_from_u64(42);

    let tuning = TreeGenConfig {
        p_terminal: 0.3,
        const_range: (-1.0, 1.0),
    };

    let pop_size = 8;
    let max_depth = 4;

    // Build each individual by hand: get a builder from the context, generate a
    // random tree into it, then register the root and collect the individual.
    let mut individuals = Vec::with_capacity(pop_size);
    for _ in 0..pop_size {
        let mut b: symbolic_rs::gp::IndividualBuilder<'_, SimpleGenome> = ctx.builder(&mut rng);
        let root_node = gen_tree::<SimpleGenome, _>(&mut b, &tuning, TreeMethod::Grow, max_depth);
        let (ind, _) = b.finish(root_node);
        individuals.push(ind);
    }
    let pop = Population::new(individuals);

    println!("Built {} individuals\n", pop.len());

    // Evaluate each individual on input [x0=0.5, x1=1.0].
    let inputs = Array2::from_shape_vec((1, 2), vec![0.5_f32, 1.0_f32]).unwrap();

    let arena = ctx.source_arena();
    let eval = EvalContext::new(arena, &ctx.operations);

    for (i, ind) in pop.iter().enumerate() {
        let node_count = arena.iter_expr_nodes(ind.root).count();
        let param_count = ind.parameters.len();
        let root_node = arena.get_root(ind.root).unwrap();

        // eval_batch needs at least one parameter column even for param-free trees.
        let mut p = ind.parameters.clone();
        let n = p.len().max(1);
        if p.is_empty() {
            p.push(0.0);
        }
        let params = Array2::from_shape_vec((1, n), p).unwrap();
        let out = eval.eval_batch(root_node, inputs.view(), params.view());

        println!(
            "  [{i}] nodes={node_count:2}  params={param_count}  f(0.5, 1.0) = {:.4}",
            out[0]
        );
    }
}
