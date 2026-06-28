use rand::{Rng, RngCore};

use crate::{
    ast::{ExprArena, ExprNode, NodeKind},
    gp::{Genome, Individual},
    ops::OperationTable,
    types::{NodeId, ParameterId, RootId, Scalar},
};

use super::{Mutation, MutationContext};

fn copy_over_replacing<G: Genome + 'static>(
    src_node: NodeId,
    target: NodeId,
    replacement: Option<NodeId>,
    ctx: &mut MutationContext<'_, G>,
) -> NodeId {
    if src_node == target {
        // A passthrough mutation yields `None`; copy the target subtree verbatim
        // (preserving tags) in that case.
        return replacement.unwrap_or_else(|| ctx.copy_subtree(target));
    }

    // Bind arena as a copy of the shared ref before borrowing `ctx` mutably below.
    let arena: &ExprArena<G::Tag> = ctx.source;
    let node = arena
        .get_node(src_node)
        .expect("invalid source node in rebuild");

    let kind = node.kind;
    let tag = node.tag.clone();

    let new_kind = match kind {
        NodeKind::Unary { value, op } => {
            let v = copy_over_replacing(value, target, replacement, ctx);
            NodeKind::Unary { value: v, op }
        }
        NodeKind::Binary { left, right, op } => {
            let l = copy_over_replacing(left, target, replacement, ctx);
            let r = copy_over_replacing(right, target, replacement, ctx);
            NodeKind::Binary {
                left: l,
                right: r,
                op,
            }
        }
        leaf => leaf,
    };

    // Preserve tag: this node is on the unchanged path.
    ctx.dest.add(ExprNode::new(new_kind, tag))
}

/// Compact `params` to only the parameters still referenced by the tree at
/// `root`, renumbering survivors densely and rewriting the `Parameter` node ids
/// in place. Survivors are numbered in first-encounter (DFS pre-order) order.
fn eliminate_dead_params<Tag: Clone>(
    arena: &mut ExprArena<Tag>,
    params: &mut Vec<Scalar>,
    root: RootId,
) {
    // Collect node ids first; the walk borrows the arena immutably.
    let ids: Vec<NodeId> = arena.walk_expr(root).into_iter().flatten().collect();

    // First-encounter remap: old ParameterId -> new dense id, building new vec.
    let mut remap: Vec<Option<ParameterId>> = vec![None; params.len()];
    let mut new_params: Vec<Scalar> = Vec::new();
    for &id in &ids {
        if let NodeKind::Parameter(pid) = arena.get_node(id).unwrap().kind {
            if remap[*pid as usize].is_none() {
                remap[*pid as usize] = Some(ParameterId::from(new_params.len() as u16));
                new_params.push(params[*pid as usize]);
            }
        }
    }

    // Rewrite parameter node ids in place.
    for &id in &ids {
        if let NodeKind::Parameter(pid) = arena.get_node(id).unwrap().kind {
            let new = remap[*pid as usize].unwrap();
            arena.get_node_mut(id).unwrap().kind = NodeKind::Parameter(new);
        }
    }

    *params = new_params;
}

/// Applies the provided `mutation` to `target` within `parent`'s tree,
/// building the resulting offspring into `dest`.
///
/// This clones the parent's parameters, applies the mutation at `target`,
/// rebuilds the tree into `dest`, and runs dead-parameter elimination
/// if the mutation changed the expression structure.
/// Returns `None` if `parent` has no root in `source`.
pub fn apply_mutation<G: Genome + 'static>(
    mutation: &dyn Mutation<G>,
    target: NodeId,
    parent: &Individual<G>,
    source: &ExprArena<G::Tag>,
    dest: &mut ExprArena<G::Tag>,
    ops: &OperationTable,
    rng: &mut dyn RngCore,
) -> Option<Individual<G>> {
    let root_node: NodeId = source.get_root(parent.root)?;

    // Clone parent params so the offspring gets an owned copy
    let mut params = parent.parameters.clone();

    let mut ctx = MutationContext::new(source, ops, rng, dest, &mut params);

    let new_subtree_node = mutation.apply(target, &mut ctx);
    let changed_structure = new_subtree_node.is_some();

    let new_root_node = copy_over_replacing(root_node, target, new_subtree_node, &mut ctx);
    drop(ctx);

    let new_root = dest.add_root(new_root_node);

    // run dead param elimination if mutation changed expression structure
    if changed_structure {
        eliminate_dead_params(dest, &mut params, new_root);
    }

    Some(Individual::new(new_root, params))
}

/// Weighted, pluggable registry of mutations.
///
/// On each call to `mutate`, one mutation is selected (weighted by its
/// registered weight, restricted to mutations that have at least one valid
/// target) and applied to a randomly chosen qualifying node.
pub struct Mutator<G: Genome> {
    entries: Vec<(f32, Box<dyn Mutation<G>>)>,
    total_weight: f32,
}

impl<G: Genome + 'static> Mutator<G> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total_weight: 0.0,
        }
    }

    /// Register a mutation with the given selection weight.
    pub fn add(&mut self, weight: f32, m: impl Mutation<G> + 'static) -> &mut Self {
        assert!(weight > 0.0, "mutation weight must be positive");
        self.total_weight += weight;
        self.entries.push((weight, Box::new(m)));
        self
    }

    /// Apply one mutation to `parent`, building the offspring into `dest`.
    ///
    /// Returns the offspring `Individual`, or `None` if no registered mutation
    /// has a valid target node in the parent's tree.
    pub fn mutate(
        &self,
        parent: &Individual<G>,
        source: &ExprArena<G::Tag>,
        dest: &mut ExprArena<G::Tag>,
        ops: &OperationTable,
        rng: &mut dyn RngCore,
    ) -> Option<Individual<G>> {
        // Collect mutable candidate nodes from the genome of the invidivual.
        let candidates: Vec<NodeId> = G::mutation_targets(parent.root, source);
        if candidates.is_empty() {
            return None;
        }

        // Build (mutation_index, valid_targets) pairs.
        let applicable: Vec<(usize, Vec<NodeId>)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, (_, m))| {
                let targets: Vec<NodeId> = candidates
                    .iter()
                    .copied()
                    .filter(|&id| source.get_node(id).is_some_and(|n| m.applies_to(n.kind)))
                    .collect();
                if targets.is_empty() {
                    None
                } else {
                    Some((i, targets))
                }
            })
            .collect();

        if applicable.is_empty() {
            return None;
        }

        // Weighted selection restricted to applicable mutations.
        let applicable_weight: f32 = applicable.iter().map(|(i, _)| self.entries[*i].0).sum();

        let mut pick = rng.random::<f32>() * applicable_weight;
        let (chosen_idx, targets) = applicable
            .iter()
            .find(|(i, _)| {
                pick -= self.entries[*i].0;
                pick <= 0.0
            })
            .unwrap_or(applicable.last().unwrap());

        let mutation = &*self.entries[*chosen_idx].1;

        // Pick a target uniformly.
        let target = targets[rng.random_range(0..targets.len())];

        apply_mutation(mutation, target, parent, source, dest, ops, rng)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::{
        ast::{ExprArena, ExprNode, NodeKind},
        gp::{
            Individual,
            mutation::{
                Mutation, MutationContext, Mutator, apply_mutation,
                builtin::{ParamJitter, PointMutation, SubtreeMutation},
            },
            subtree::{GrowSubtreeConfig, TreeGenConfig},
            test_genome::TestSimpleGenome,
        },
        ops::{OperationTableBuilder, builtin::MathBaseOps},
        types::{NodeId, OperationId, ParameterId, RootId, Scalar},
    };

    fn base_ops() -> crate::ops::OperationTable {
        let mut b = OperationTableBuilder::new();
        b.register_set::<MathBaseOps>();
        b.build()
    }

    // Build a simple binary tree: add(param(0), param(1))
    fn build_two_param_tree(arena: &mut ExprArena<()>) -> (RootId, Vec<Scalar>) {
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1u16), ()));
        let add = arena.add(ExprNode::new_binary(p0, p1, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        (root, vec![1.0, 2.0])
    }

    /// Run eliminate_dead_params, collect surviving
    /// ParameterIds (in tree walk order) and the surviving param values.
    fn run_elim(
        arena: &mut ExprArena<()>,
        params: &mut Vec<Scalar>,
        root: RootId,
    ) -> (Vec<u16>, Vec<Scalar>) {
        super::eliminate_dead_params(arena, params, root);
        let ids: Vec<u16> = arena
            .iter_expr_nodes(root)
            .filter_map(|(_, n)| match n.kind {
                NodeKind::Parameter(pid) => Some(*pid),
                _ => None,
            })
            .collect();
        (ids, params.clone())
    }

    #[test]
    fn copy_subtree_round_trip() {
        let mut rng = StdRng::seed_from_u64(0);
        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        let (root, mut params) = build_two_param_tree(&mut src);
        let root_node = src.get_root(root).unwrap();

        {
            let mut ctx: MutationContext<'_, TestSimpleGenome> =
                MutationContext::new(&src, &ops, &mut rng, &mut dest, &mut params);

            let copied = ctx.copy_subtree(root_node);

            // Node counts should match.
            let src_ids: Vec<_> = src.iter_expr_nodes(root).map(|(id, _)| id).collect();
            let dest_root = dest.add_root(copied);
            let dest_ids: Vec<_> = dest.iter_expr_nodes(dest_root).map(|(id, _)| id).collect();
            assert_eq!(src_ids.len(), dest_ids.len());
        }
    }

    // ---------------------------------------------------------------------------
    // PointMutation — keeps structure, changes operator
    // ---------------------------------------------------------------------------
    #[test]
    fn point_mutation_preserves_arity() {
        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();
        let (root, params) = build_two_param_tree(&mut src);

        // PointMutation acts on an operator node — here the root `add`.
        let target = src.get_root(root).unwrap();
        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(1);

        let offspring = apply_mutation(
            &PointMutation,
            target,
            &parent,
            &src,
            &mut dest,
            &ops,
            &mut rng,
        )
        .unwrap();

        // Same number of nodes.
        let src_count = src.iter_expr_nodes(parent.root).count();
        let dest_count = dest.iter_expr_nodes(offspring.root).count();
        assert_eq!(src_count, dest_count);
    }

    // ---------------------------------------------------------------------------
    // ParamJitter — identical structure, one parameter shifted
    // ---------------------------------------------------------------------------
    #[test]
    fn param_jitter_changes_exactly_one_param() {
        let ops = base_ops();

        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        let (root, params) = build_two_param_tree(&mut src);
        let original_params = params.clone();

        // ParamJitter acts on a parameter node — pick the first one.
        let target = src
            .iter_expr_nodes(root)
            .find(|(_, n)| matches!(n.kind, NodeKind::Parameter(_)))
            .map(|(id, _)| id)
            .unwrap();

        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(2);

        let offspring = apply_mutation(
            &ParamJitter { stddev: 0.1 },
            target,
            &parent,
            &src,
            &mut dest,
            &ops,
            &mut rng,
        )
        .unwrap();

        // Same tree structure (same number of nodes).
        let src_count = src.iter_expr_nodes(parent.root).count();
        let dest_count = dest.iter_expr_nodes(offspring.root).count();
        assert_eq!(src_count, dest_count);

        // Exactly one parameter should have changed.
        let changed = offspring
            .parameters
            .iter()
            .zip(&original_params)
            .filter(|(a, b)| (*a - *b).abs() > 1e-9)
            .count();
        assert_eq!(changed, 1);
    }

    // ---------------------------------------------------------------------------
    // SubtreeMutation — offspring is valid and evaluates without panic
    // ---------------------------------------------------------------------------
    #[test]
    fn subtree_mutation_produces_valid_tree() {
        use crate::eval::EvalContext;
        use ndarray::array;

        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();
        let (root, params) = build_two_param_tree(&mut src);

        // SubtreeMutation can replace any node — target the root.
        let target = src.get_root(root).unwrap();
        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(3);

        let cfg = GrowSubtreeConfig {
            max_depth: 3,
            tuning: TreeGenConfig {
                p_terminal: 0.3,
                n_variables: 2,
                const_range: (-1.0, 1.0),
            },
        };

        let offspring = apply_mutation(
            &SubtreeMutation { grow: cfg },
            target,
            &parent,
            &src,
            &mut dest,
            &ops,
            &mut rng,
        )
        .unwrap();

        let root_node = dest.get_root(offspring.root).unwrap();
        let eval_ctx = EvalContext::new(&dest, &ops);

        let inputs = array![[0.5f32, 1.0f32]];
        let n_params = offspring.parameters.len();
        let params_arr = ndarray::Array2::from_shape_vec((1, n_params.max(1)), {
            let mut v = offspring.parameters.clone();
            if n_params == 0 {
                v.push(0.0);
            }
            v
        })
        .unwrap();

        // Must not panic.
        let _ = eval_ctx.eval_batch(root_node, inputs.view(), params_arr.view());
    }

    // ---------------------------------------------------------------------------
    // apply_mutation in isolation — drives a fixed mutation at a chosen target,
    // exercising the structural-change path (rebuild + dead-param elimination)
    // and confirming the parent is left untouched.
    // ---------------------------------------------------------------------------
    #[test]
    fn apply_mutation_runs_dead_param_elimination_and_clones_parent() {
        use crate::gp::builder::NodeBuilder;
        use crate::types::VariableId;

        // A mutation that swaps the target subtree for variable(0), making any
        // parameter the target referenced dead.
        struct ReplaceWithVariable;
        impl Mutation<TestSimpleGenome> for ReplaceWithVariable {
            fn applies_to(&self, _kind: NodeKind) -> bool {
                true
            }
            fn apply(
                &self,
                _target: NodeId,
                ctx: &mut MutationContext<'_, TestSimpleGenome>,
            ) -> Option<NodeId> {
                Some(ctx.emit(ExprNode::new_variable(VariableId::from(0u16), ())))
            }
        }

        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        // add(param(0), param(1)) with params [1.0, 2.0].
        let (root, params) = build_two_param_tree(&mut src);

        // Target param(1)'s node — replacing it yields add(param(0), variable(0)),
        // leaving param(1) (value 2.0) dead.
        let target = src
            .iter_expr_nodes(root)
            .find_map(|(id, n)| match n.kind {
                NodeKind::Parameter(pid) if *pid == 1 => Some(id),
                _ => None,
            })
            .unwrap();

        let parent = Individual::<TestSimpleGenome>::new(root, params);
        let mut rng = StdRng::seed_from_u64(7);

        let offspring = apply_mutation(
            &ReplaceWithVariable,
            target,
            &parent,
            &src,
            &mut dest,
            &ops,
            &mut rng,
        )
        .unwrap();

        // Dead-param elimination ran: only the surviving param(0) (1.0) remains,
        // renumbered densely from 0.
        assert_eq!(offspring.parameters, vec![1.0]);
        let surviving_pids: Vec<u16> = dest
            .iter_expr_nodes(offspring.root)
            .filter_map(|(_, n)| match n.kind {
                NodeKind::Parameter(pid) => Some(*pid),
                _ => None,
            })
            .collect();
        assert_eq!(surviving_pids, vec![0]);

        // The parent's parameters are untouched — apply_mutation works on a clone.
        assert_eq!(parent.parameters, vec![1.0, 2.0]);
    }

    #[test]
    fn dead_params_all_live_unchanged() {
        // add(param(0), param(1)) — both params are used, nothing to eliminate.
        let mut arena: ExprArena<()> = ExprArena::new();
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1u16), ()));
        let add = arena.add(ExprNode::new_binary(p0, p1, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        let mut params = vec![1.0_f32, 2.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(surviving, vec![1.0, 2.0]);
    }

    #[test]
    fn dead_params_trailing_param_eliminated() {
        // add(param(0), variable(0)) — param(1) is in the params vec but not used.
        let mut arena: ExprArena<()> = ExprArena::new();
        use crate::types::VariableId;
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let v0 = arena.add(ExprNode::new_variable(VariableId::from(0u16), ()));
        let add = arena.add(ExprNode::new_binary(p0, v0, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        let mut params = vec![1.0_f32, 99.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        assert_eq!(ids, vec![0]);
        assert_eq!(surviving, vec![1.0]);
    }

    #[test]
    fn dead_params_leading_param_removed() {
        // add(param(1), variable(0)) — param(0) is unused; param(1) should become param(0).
        let mut arena: ExprArena<()> = ExprArena::new();
        use crate::types::VariableId;
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1u16), ()));
        let v0 = arena.add(ExprNode::new_variable(VariableId::from(0u16), ()));
        let add = arena.add(ExprNode::new_binary(p1, v0, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        let mut params = vec![99.0_f32, 2.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        assert_eq!(ids, vec![0]);
        assert_eq!(surviving, vec![2.0]);
    }

    #[test]
    fn dead_params_middle_gap_eliminated() {
        // add(param(0), param(2)) — param(1) is unused; param(2) should become param(1).
        let mut arena: ExprArena<()> = ExprArena::new();
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let p2 = arena.add(ExprNode::new_parameter(ParameterId::from(2u16), ()));
        let add = arena.add(ExprNode::new_binary(p0, p2, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        let mut params = vec![1.0_f32, 99.0_f32, 3.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(surviving, vec![1.0, 3.0]);
    }

    #[test]
    fn dead_params_all_dead_clears_params() {
        // variable(0) — no parameters at all; params vec should be cleared.
        let mut arena: ExprArena<()> = ExprArena::new();
        use crate::types::VariableId;
        let v = arena.add(ExprNode::new_variable(VariableId::from(0u16), ()));
        let root = arena.add_root(v);
        let mut params = vec![1.0_f32, 2.0_f32, 3.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        assert!(ids.is_empty());
        assert!(surviving.is_empty());
    }

    #[test]
    fn dead_params_duplicate_reference_preserved_once() {
        // add(param(0), param(0)) — same param referenced twice; should appear once in params.
        let mut arena: ExprArena<()> = ExprArena::new();
        let p0a = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let p0b = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let add = arena.add(ExprNode::new_binary(p0a, p0b, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        let mut params = vec![42.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        // Both nodes still refer to pid 0; params shrinks to just the one entry.
        assert_eq!(ids, vec![0, 0]);
        assert_eq!(surviving, vec![42.0]);
    }

    #[test]
    fn dead_params_numbering_follows_dfs_preorder() {
        // add(param(1), param(0)) — left child is param(1), right is param(0).
        // DFS pre-order visits left before right, so param(1) is seen first
        // and should become new param(0); param(0) becomes new param(1).
        let mut arena: ExprArena<()> = ExprArena::new();
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1u16), ()));
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let add = arena.add(ExprNode::new_binary(p1, p0, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        let mut params = vec![10.0_f32, 20.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        // left param (old 1, value 20.0) → new id 0
        // right param (old 0, value 10.0) → new id 1
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(surviving, vec![20.0, 10.0]);
    }

    // ---------------------------------------------------------------------------
    // Determinism — same seed ⇒ identical offspring
    // ---------------------------------------------------------------------------
    #[test]
    fn mutator_is_deterministic() {
        let ops = base_ops();

        let run = |seed: u64| -> Vec<NodeId> {
            let mut src: ExprArena<()> = ExprArena::new();
            let mut dest: ExprArena<()> = ExprArena::new();
            let (root, params) = build_two_param_tree(&mut src);
            let parent = Individual::<TestSimpleGenome>::new(root, params);
            let mut rng = StdRng::seed_from_u64(seed);

            let mut mutator: Mutator<TestSimpleGenome> = Mutator::new();
            mutator.add(1.0, PointMutation);

            let offspring = mutator
                .mutate(&parent, &src, &mut dest, &ops, &mut rng)
                .unwrap();
            dest.iter_expr_nodes(offspring.root)
                .map(|(id, _)| id)
                .collect()
        };

        assert_eq!(run(99), run(99));
    }
}
