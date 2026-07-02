use crate::types::{NodeId, OperationId, ParameterId, RootId, VariableId};

/// Type of node.
#[derive(PartialEq, PartialOrd, Debug, Clone, Copy)]
pub enum NodeKind {
    Unary {
        value: NodeId,
        op: OperationId,
    },
    Binary {
        left: NodeId,
        right: NodeId,
        op: OperationId,
    },
    /// Input
    Variable(VariableId),
    /// Constant or optimizable parameter
    Parameter(ParameterId),
}

/// A node in an epression AST
#[derive(PartialEq, PartialOrd, Debug)]
pub struct ExprNode<Tag: Clone> {
    /// Kind of node
    pub kind: NodeKind,
    /// Tag value attached to this node.
    pub tag: Tag,
}

/// An arena containing nodes for expression ASTs
pub struct ExprArena<Tag: Clone> {
    nodes: Vec<ExprNode<Tag>>,
    roots: Vec<NodeId>,
}

/// An iterator that iterates over the nodes of an expression.
/// Returns the node IDs
pub struct ExprNodeIter<'a, Tag: Clone> {
    arena: &'a ExprArena<Tag>,
    stack: Vec<NodeId>,
}

impl<Tag: Clone> ExprArena<Tag> {
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
        }
    }

    /// Appends the provided node to the arena and returns its node ID.
    pub fn add(&mut self, node: ExprNode<Tag>) -> NodeId {
        let id = NodeId::from(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Adds the provided node ID as a root
    pub fn add_root(&mut self, id: NodeId) -> RootId {
        let root = RootId::from(self.roots.len());
        self.roots.push(id);
        root
    }

    /// Returns the node for the provided node.
    pub fn get_node(&self, node_id: NodeId) -> Option<&ExprNode<Tag>> {
        self.nodes.get(usize::from(node_id))
    }

    /// Returns a mutable reference to the provided node.
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut ExprNode<Tag>> {
        self.nodes.get_mut(usize::from(node_id))
    }

    /// Returns the Node Id for the provided root id.
    pub fn get_root(&self, root_id: RootId) -> Option<NodeId> {
        self.roots.get(usize::from(root_id)).copied()
    }

    /// Returns an iterator that walks the Node IDs of the subtree rooted at
    /// `node` (including `node` itself) in DFS pre-order. An invalid `node`
    /// yields an empty iterator.
    pub fn walk_expr(&self, node: NodeId) -> ExprNodeIter<'_, Tag> {
        ExprNodeIter {
            arena: self,
            stack: vec![node],
        }
    }

    /// Returns an iterator that walks the Node IDs of the expression registered
    /// under `root_id`, or `None` if the root id is invalid.
    pub fn walk_root(&self, root_id: RootId) -> Option<ExprNodeIter<'_, Tag>> {
        self.get_root(root_id).map(|node| self.walk_expr(node))
    }

    /// Returns an iterator over the nodes of the subtree rooted at `node`.
    pub fn iter_expr_nodes(&self, node: NodeId) -> impl Iterator<Item = (NodeId, &ExprNode<Tag>)> {
        self.walk_expr(node)
            .map(move |id| (id, self.get_node(id).unwrap()))
    }

    /// Recursively deep-copies the subtree rooted at `node` from `self` into
    /// `dest`, preserving each node's tag, and returns the new root `NodeId`
    /// within `dest`.
    pub fn copy_over(&self, node: NodeId, dest: &mut ExprArena<Tag>) -> Option<NodeId> {
        let src = self.get_node(node)?;

        let tag = src.tag.clone();
        let new_kind = match src.kind {
            NodeKind::Unary { value, op } => NodeKind::Unary {
                value: self.copy_over(value, dest)?,
                op,
            },
            NodeKind::Binary { left, right, op } => {
                let left = self.copy_over(left, dest)?;
                let right = self.copy_over(right, dest)?;
                NodeKind::Binary { left, right, op }
            }
            leaf => leaf,
        };

        Some(dest.add(ExprNode::new(new_kind, tag)))
    }

    /// Deep-copies the subtree registered under `root_id` in `self` into `dest`
    /// and registers the copy as a new root in `dest`, returning its `RootId`.
    ///
    /// Returns `None` if `root_id` is not a valid root of `self`.
    pub fn copy_root_over(&self, root_id: RootId, dest: &mut ExprArena<Tag>) -> Option<RootId> {
        let node = self.get_root(root_id)?;
        let copied = self.copy_over(node, dest)?;
        Some(dest.add_root(copied))
    }

    /// Empties the contents of an arena and invalidates all existing references to it.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.roots.clear();
    }
}

impl<'a, Tag: Clone> Iterator for ExprNodeIter<'a, Tag> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let node_id = self.stack.pop()?;
        let node = self.arena.get_node(node_id)?;

        match &node.kind {
            NodeKind::Unary { value, .. } => {
                self.stack.push(*value);
            }
            NodeKind::Binary { left, right, .. } => {
                self.stack.push(*right);
                self.stack.push(*left);
            }
            NodeKind::Variable(_) | NodeKind::Parameter(_) => {}
        }

        Some(node_id)
    }
}

pub mod utils {}

impl<Tag: Clone> ExprNode<Tag> {
    pub const fn new(kind: NodeKind, tag: Tag) -> Self {
        Self { kind, tag }
    }

    /// Returns a node with a binary operation.
    pub const fn new_binary(left: NodeId, right: NodeId, op: OperationId, tag: Tag) -> Self {
        Self::new(NodeKind::Binary { left, right, op }, tag)
    }

    /// Returns a node with an unary operation.
    pub const fn new_unary(value: NodeId, op: OperationId, tag: Tag) -> Self {
        Self::new(NodeKind::Unary { value, op }, tag)
    }

    /// Returns a node which reads variables at index `var`.
    pub const fn new_variable(var: VariableId, tag: Tag) -> Self {
        Self::new(NodeKind::Variable(var), tag)
    }

    /// Returns a node which reads parameters at index `param`.
    pub const fn new_parameter(param: ParameterId, tag: Tag) -> Self {
        Self::new(NodeKind::Parameter(param), tag)
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{
        ExprArena, ExprNode, NodeId, NodeKind, OperationId, ParameterId, RootId, VariableId,
    };

    #[test]
    fn test_invalid_node_returns_none() {
        let arena: ExprArena<()> = ExprArena::new();

        assert!(arena.get_node(NodeId::from(0)).is_none());
    }

    #[test]
    pub fn test_arena_root_same_node_id() {
        let mut arena = ExprArena::new();

        let expr = arena.add(ExprNode::new_parameter(ParameterId::from(1), ()));
        let expr_root = arena.add_root(expr);

        assert_eq!(arena.get_root(expr_root).unwrap(), expr,);
    }

    #[test]
    fn test_invalid_root_returns_none() {
        let arena: ExprArena<()> = ExprArena::new();

        assert!(arena.get_root(RootId::from(0)).is_none());
    }

    #[test]
    fn test_walk_unary_expr() {
        let mut arena = ExprArena::new();

        let child = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let parent = arena.add(ExprNode::new_unary(child, OperationId::from(0), ()));

        arena.add_root(parent);
        let visited: Vec<NodeId> = arena.walk_expr(parent).collect();

        assert_eq!(visited, vec![parent, child]);
    }

    #[test]
    fn test_walk_binary_expr_preorder() {
        let mut arena = ExprArena::new();

        let left = arena.add(ExprNode::new_parameter(ParameterId::from(1), ()));
        let right = arena.add(ExprNode::new_parameter(ParameterId::from(2), ()));
        let root_node = arena.add(ExprNode::new_binary(left, right, OperationId::from(0), ()));

        arena.add_root(root_node);
        let visited: Vec<_> = arena.walk_expr(root_node).collect();

        assert_eq!(visited, vec![root_node, left, right]);
    }

    #[test]
    fn test_walk_expr_invalid_node() {
        let arena: ExprArena<()> = ExprArena::new();

        assert_eq!(arena.walk_expr(NodeId::from(0)).count(), 0);
    }

    #[test]
    fn test_walk_nested_expression() {
        let mut arena = ExprArena::new();

        let a = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let b = arena.add(ExprNode::new_parameter(ParameterId::from(1), ()));
        let c = arena.add(ExprNode::new_variable(VariableId::from(2), ()));
        let mul = arena.add(ExprNode::new_binary(a, b, OperationId::from(1), ()));
        let add = arena.add(ExprNode::new_binary(mul, c, OperationId::from(2), ()));

        arena.add_root(add);

        let visited: Vec<_> = arena.walk_expr(add).collect();

        assert_eq!(visited, vec![add, mul, a, b, c]);
    }

    #[test]
    fn test_iter_expr_nodes() {
        let mut arena = ExprArena::new();

        let p = arena.add(ExprNode::new_parameter(ParameterId::from(123), ()));

        arena.add_root(p);

        let items: Vec<_> = arena.iter_expr_nodes(p).collect();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, p);

        match items[0].1.kind {
            NodeKind::Parameter(id) => {
                assert_eq!(id, ParameterId::from(123));
            }
            _ => panic!("expected parameter node"),
        }
    }

    #[test]
    fn test_get_node_mut() {
        let mut arena = ExprArena::new();

        let p = arena.add(ExprNode::new_parameter(ParameterId::from(123), ()));

        assert!(
            matches!(arena.get_node(p).unwrap().kind, NodeKind::Parameter(i) if i == ParameterId::from(123))
        );

        {
            let node_mut = arena.get_node_mut(p).unwrap();
            node_mut.kind = NodeKind::Parameter(ParameterId::from(234));
        }

        assert!(
            matches!(arena.get_node(p).unwrap().kind, NodeKind::Parameter(i) if i == ParameterId::from(234))
        );
    }

    /// Builds `(p0 * v2) + p1` into `arena`, returning the root node id.
    fn build_nested(arena: &mut ExprArena<()>) -> NodeId {
        let a = arena.add(ExprNode::new_parameter(ParameterId::from(0), ()));
        let b = arena.add(ExprNode::new_parameter(ParameterId::from(1), ()));
        let c = arena.add(ExprNode::new_variable(VariableId::from(2), ()));
        let mul = arena.add(ExprNode::new_binary(a, c, OperationId::from(1), ()));
        arena.add(ExprNode::new_binary(mul, b, OperationId::from(2), ()))
    }

    /// Asserts two subtrees (possibly in different arenas) are structurally
    /// identical: same ops and leaf payloads, ignoring arena node ids.
    fn assert_tree_eq(a: &ExprArena<()>, an: NodeId, b: &ExprArena<()>, bn: NodeId) {
        match (a.get_node(an).unwrap().kind, b.get_node(bn).unwrap().kind) {
            (NodeKind::Unary { value: av, op: ao }, NodeKind::Unary { value: bv, op: bo }) => {
                assert_eq!(ao, bo);
                assert_tree_eq(a, av, b, bv);
            }
            (
                NodeKind::Binary {
                    left: al,
                    right: ar,
                    op: ao,
                },
                NodeKind::Binary {
                    left: bl,
                    right: br,
                    op: bo,
                },
            ) => {
                assert_eq!(ao, bo);
                assert_tree_eq(a, al, b, bl);
                assert_tree_eq(a, ar, b, br);
            }
            (ka, kb) => assert_eq!(ka, kb),
        }
    }

    #[test]
    fn copy_over_invalid_node_returns_none() {
        let src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        assert!(src.copy_over(NodeId::from(0), &mut dest).is_none());
        assert!(dest.get_node(NodeId::from(0)).is_none());
    }

    #[test]
    fn copy_over_leaf_preserves_kind() {
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        let p = src.add(ExprNode::new_parameter(ParameterId::from(7), ()));
        let copied = src.copy_over(p, &mut dest).unwrap();

        assert!(matches!(
            dest.get_node(copied).unwrap().kind,
            NodeKind::Parameter(id) if id == ParameterId::from(7)
        ));
    }

    #[test]
    fn copy_over_preserves_structure_into_empty_arena() {
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        let root = build_nested(&mut src);
        let copied = src.copy_over(root, &mut dest).unwrap();

        assert_tree_eq(&dest, copied, &src, root);
        assert_eq!(dest.walk_expr(copied).count(), 5);
    }

    #[test]
    fn copy_over_appends_after_existing_nodes() {
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        // Pre-populate dest so the copy must be offset past existing nodes.
        let existing = dest.add(ExprNode::new_variable(VariableId::from(9), ()));

        let root = build_nested(&mut src);
        let copied = src.copy_over(root, &mut dest).unwrap();

        // Existing node is untouched; the copy is structurally equal to the
        // source with its child ids remapped into dest (assert_tree_eq follows
        // those ids, so a bad remap would fail to resolve or mismatch).
        assert!(matches!(
            dest.get_node(existing).unwrap().kind,
            NodeKind::Variable(v) if v == VariableId::from(9)
        ));
        assert_tree_eq(&dest, copied, &src, root);
    }

    #[test]
    fn copy_root_over_registers_new_root() {
        let mut src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        let root_node = build_nested(&mut src);
        let src_root = src.add_root(root_node);

        let dest_root = src.copy_root_over(src_root, &mut dest).unwrap();

        let dest_root_node = dest.get_root(dest_root).unwrap();
        assert_tree_eq(&dest, dest_root_node, &src, root_node);
    }

    #[test]
    fn copy_root_over_invalid_root_returns_none() {
        let src: ExprArena<()> = ExprArena::new();
        let mut dest: ExprArena<()> = ExprArena::new();

        assert!(src.copy_root_over(RootId::from(0), &mut dest).is_none());
    }
}
