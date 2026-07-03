pub mod builtin;
pub mod mutator;

pub use mutator::Mutator;

use rand::RngCore;

use crate::gp::Individual;
use crate::types::{RootId, Scalar};
use crate::{
    ast::{ExprArena, ExprNode, NodeKind},
    ops::OperationTable,
    types::{NodeId, ParameterId},
};

use super::Genome;
use super::builder::NodeBuilder;

/// A pluggable tree mutation.
///
/// Implementations answer two questions: can this mutation act on `node`, and
/// what subtree should replace `target` in the offspring? Unchanged parts of
/// the tree are copied verbatim via `MutationContext::copy_subtree`.
pub trait Mutation<G: Genome>: 'static {
    /// Whether this mutation can act on a node of the given kind.
    fn applies_to(&self, kind: NodeKind) -> bool;

    /// Emit a replacement subtree for the target into the dest arena and return
    /// its root `NodeId`. `target` is the source node id (useful for splicing or
    /// copying the target's own subtree) and `node` is the resolved source node.
    ///
    /// ### Notes
    ///
    /// - Use `ctx.copy_subtree` to carry over unchanged children of an emitted node.
    /// - Returning `None` indicates that the mutation didn't affect structurally the expression subtree. Useful for mutations affecting parameters.
    fn apply(
        &self,
        target: NodeId,
        node: &ExprNode<G::Tag>,
        ctx: &mut MutationContext<'_, G>,
    ) -> Option<NodeId>;
}

/// Mutation context passed to a mutation
pub struct MutationContext<'a, G: Genome> {
    pub(crate) source: &'a ExprArena<G::Tag>,
    pub(crate) dest: &'a mut ExprArena<G::Tag>,
    pub(crate) ops: &'a OperationTable,
    pub(crate) rng: &'a mut dyn RngCore,
    pub(crate) params: &'a mut Vec<Scalar>,
}

impl<'a, G: Genome> MutationContext<'a, G> {
    pub const fn new(
        source: &'a ExprArena<G::Tag>,
        ops: &'a OperationTable,
        rng: &'a mut dyn RngCore,
        dest: &'a mut ExprArena<G::Tag>,
        params: &'a mut Vec<Scalar>,
    ) -> Self {
        Self {
            source,
            ops,
            rng,
            dest,
            params,
        }
    }

    /// Verbatim deep copy of a source subtree, preserving each node's tag.
    pub fn copy_subtree(&mut self, src: NodeId) -> NodeId {
        // Bind the shared ref before the mutable reborrow of `dest`.
        let source: &ExprArena<G::Tag> = self.source;
        source
            .copy_over(src, self.dest)
            .expect("invalid source node in copy_subtree")
    }

    /// Borrow the source arena (the parent's tree), for looking up nodes and
    /// walking subtrees while building the offspring.
    pub fn source(&self) -> &ExprArena<G::Tag> {
        self.source
    }

    /// Read the current value of a parameter.
    pub fn get_parameter(&self, id: ParameterId) -> Scalar {
        self.params[*id as usize]
    }

    /// Write a value back to an existing parameter slot.
    pub fn set_parameter(&mut self, id: ParameterId, value: Scalar) {
        self.params[*id as usize] = value;
    }
}

impl<'a, G: Genome> NodeBuilder<G> for MutationContext<'a, G> {
    fn rng(&mut self) -> &mut dyn RngCore {
        self.rng
    }

    fn ops(&self) -> &OperationTable {
        self.ops
    }

    fn emit(&mut self, node: ExprNode<G::Tag>) -> NodeId {
        self.dest.add(node)
    }

    /// Allocate a new parameter slot initialised to `value` and return its id.
    fn new_parameter(&mut self, value: Scalar) -> ParameterId {
        let id = ParameterId::from(self.params.len() as u16);
        self.params.push(value);
        id
    }
}

fn copy_over_replacing<G: Genome + 'static>(
    src_node: NodeId,
    target: NodeId,
    replacement: Option<NodeId>,
    ctx: &mut MutationContext<'_, G>,
) -> NodeId {
    if src_node == target {
        // A passthrough mutation yields `None`; copy the target subtree verbatim
        // (preserving tags) in that case.
        if let Some(node) = replacement {
            return node;
        }
        let source: &ExprArena<G::Tag> = ctx.source;
        return source
            .copy_over(target, ctx.dest)
            .expect("invalid target node in copy_over_replacing");
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
    let ids: Vec<NodeId> = arena.walk_root(root).into_iter().flatten().collect();

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
    let target_node = source.get_node(target)?;

    // Clone parent params so the offspring gets an owned copy
    let mut params = parent.parameters.clone();

    let mut ctx = MutationContext::new(source, ops, rng, dest, &mut params);

    let new_subtree_node = mutation.apply(target, target_node, &mut ctx);
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
            mutation::{Mutation, MutationContext, apply_mutation},
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

    fn build_two_param_tree(arena: &mut ExprArena<()>) -> (RootId, Vec<Scalar>) {
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1u16), ()));
        let add = arena.add(ExprNode::new_binary(p0, p1, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        (root, vec![1.0, 2.0])
    }

    fn run_elim(
        arena: &mut ExprArena<()>,
        params: &mut Vec<Scalar>,
        root: RootId,
    ) -> (Vec<u16>, Vec<Scalar>) {
        super::eliminate_dead_params(arena, params, root);
        let root_node = arena.get_root(root).unwrap();
        let ids: Vec<u16> = arena
            .iter_expr_nodes(root_node)
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

            let src_ids: Vec<_> = src.iter_expr_nodes(root_node).map(|(id, _)| id).collect();
            dest.add_root(copied);
            let dest_ids: Vec<_> = dest.iter_expr_nodes(copied).map(|(id, _)| id).collect();
            assert_eq!(src_ids.len(), dest_ids.len());
        }
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

        struct ReplaceWithVariable;
        impl Mutation<TestSimpleGenome> for ReplaceWithVariable {
            fn applies_to(&self, _kind: NodeKind) -> bool {
                true
            }
            fn apply(
                &self,
                _target: NodeId,
                _node: &ExprNode<()>,
                ctx: &mut MutationContext<'_, TestSimpleGenome>,
            ) -> Option<NodeId> {
                Some(ctx.emit(ExprNode::new_variable(VariableId::from(0u16), ())))
            }
        }

        let ops = base_ops();
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        let (root, params) = build_two_param_tree(&mut src);

        let root_node = src.get_root(root).unwrap();
        let target = src
            .iter_expr_nodes(root_node)
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

        assert_eq!(offspring.parameters, vec![1.0]);
        let offspring_root = dest.get_root(offspring.root).unwrap();
        let surviving_pids: Vec<u16> = dest
            .iter_expr_nodes(offspring_root)
            .filter_map(|(_, n)| match n.kind {
                NodeKind::Parameter(pid) => Some(*pid),
                _ => None,
            })
            .collect();
        assert_eq!(surviving_pids, vec![0]);

        assert_eq!(parent.parameters, vec![1.0, 2.0]);
    }

    #[test]
    fn dead_params_all_live_unchanged() {
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
        use crate::types::VariableId;
        let mut arena: ExprArena<()> = ExprArena::new();
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
        use crate::types::VariableId;
        let mut arena: ExprArena<()> = ExprArena::new();
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
        use crate::types::VariableId;
        let mut arena: ExprArena<()> = ExprArena::new();
        let v = arena.add(ExprNode::new_variable(VariableId::from(0u16), ()));
        let root = arena.add_root(v);
        let mut params = vec![1.0_f32, 2.0_f32, 3.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        assert!(ids.is_empty());
        assert!(surviving.is_empty());
    }

    #[test]
    fn dead_params_duplicate_reference_preserved_once() {
        let mut arena: ExprArena<()> = ExprArena::new();
        let p0a = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let p0b = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let add = arena.add(ExprNode::new_binary(p0a, p0b, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        let mut params = vec![42.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        assert_eq!(ids, vec![0, 0]);
        assert_eq!(surviving, vec![42.0]);
    }

    #[test]
    fn dead_params_numbering_follows_dfs_preorder() {
        // DFS pre-order visits left before right, so param(1) is seen first
        // and should become new param(0); param(0) becomes new param(1).
        let mut arena: ExprArena<()> = ExprArena::new();
        let p1 = arena.add(ExprNode::new_parameter(ParameterId::from(1u16), ()));
        let p0 = arena.add(ExprNode::new_parameter(ParameterId::from(0u16), ()));
        let add = arena.add(ExprNode::new_binary(p1, p0, OperationId::from(0u16), ()));
        let root = arena.add_root(add);
        let mut params = vec![10.0_f32, 20.0_f32];

        let (ids, surviving) = run_elim(&mut arena, &mut params, root);
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(surviving, vec![20.0, 10.0]);
    }
}
