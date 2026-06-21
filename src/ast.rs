use crate::types::{NodeId, OperationId, ParameterId, RootId, VariableId};

/// Type of node.
#[derive(PartialEq, PartialOrd, Debug)]
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
pub struct ExprNode<Tag> {
    /// Kind of node
    pub kind: NodeKind,
    /// Tag value attached to this node.
    pub tag: Tag,
}

/// An arena containing nodes for expression ASTs
pub struct ExprArena<Tag> {
    nodes: Vec<ExprNode<Tag>>,
    roots: Vec<NodeId>,
}

/// An iterator that iterates over the nodes of an expression.
/// Returns the node IDs
pub struct ExprNodeIter<'a, Tag> {
    arena: &'a ExprArena<Tag>,
    stack: Vec<NodeId>,
}

impl<Tag> ExprArena<Tag> {
    pub fn new() -> Self {
        Self {
            nodes: Default::default(),
            roots: Default::default(),
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

    /// Returns the Node Id for the provided root id.
    pub fn get_root(&self, root_id: RootId) -> Option<NodeId> {
        self.roots.get(usize::from(root_id)).copied()
    }

    /// Returns an iterator that walks the Node IDs of an expression
    pub fn walk_expr<'a>(&'a self, root_id: RootId) -> Option<ExprNodeIter<'a, Tag>> {
        let root = self.get_root(root_id)?;

        Some(ExprNodeIter {
            arena: self,
            stack: vec![root],
        })
    }

    /// Returns an interator iterating over the nodes of an expression
    pub fn iter_expr_nodes(&self, root: RootId) -> impl Iterator<Item = (NodeId, &ExprNode<Tag>)> {
        self.walk_expr(root)
            .into_iter()
            .flatten()
            .map(move |id| (id, self.get_node(id).unwrap()))
    }
}

impl<'a, Tag> Iterator for ExprNodeIter<'a, Tag> {
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

impl<Tag> ExprNode<Tag> {
    pub const fn new(kind: NodeKind, tag: Tag) -> Self {
        Self { kind, tag }
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

        let expr = arena.add(ExprNode::new(NodeKind::Parameter(ParameterId::from(1)), ()));
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

        let child = arena.add(ExprNode::new(NodeKind::Parameter(ParameterId::from(0)), ()));
        let parent = arena.add(ExprNode::new(
            NodeKind::Unary {
                value: child,
                op: OperationId::from(0),
            },
            (),
        ));

        let root = arena.add_root(parent);
        let visited: Vec<_> = arena.walk_expr(root).unwrap().collect();

        assert_eq!(visited, vec![parent, child]);
    }

    #[test]
    fn test_walk_binary_expr_preorder() {
        let mut arena = ExprArena::new();

        let left = arena.add(ExprNode::new(NodeKind::Parameter(ParameterId::from(1)), ()));
        let right = arena.add(ExprNode::new(NodeKind::Parameter(ParameterId::from(2)), ()));
        let root_node = arena.add(ExprNode::new(
            NodeKind::Binary {
                left,
                right,
                op: OperationId::from(0),
            },
            (),
        ));

        let root = arena.add_root(root_node);
        let visited: Vec<_> = arena.walk_expr(root).unwrap().collect();

        assert_eq!(visited, vec![root_node, left, right]);
    }

    #[test]
    fn test_walk_expr_invalid_root() {
        let arena: ExprArena<()> = ExprArena::new();

        assert!(arena.walk_expr(RootId::from(0)).is_none());
    }

    #[test]
    fn test_walk_nested_expression() {
        let mut arena = ExprArena::new();

        let a = arena.add(ExprNode::new(NodeKind::Parameter(ParameterId::from(0)), ()));
        let b = arena.add(ExprNode::new(NodeKind::Parameter(ParameterId::from(1)), ()));
        let c = arena.add(ExprNode::new(NodeKind::Variable(VariableId::from(2)), ()));
        let mul = arena.add(ExprNode::new(
            NodeKind::Binary {
                left: a,
                right: b,
                op: OperationId::from(1),
            },
            (),
        ));

        let add = arena.add(ExprNode::new(
            NodeKind::Binary {
                left: mul,
                right: c,
                op: OperationId::from(2),
            },
            (),
        ));

        let root = arena.add_root(add);

        let visited: Vec<_> = arena.walk_expr(root).unwrap().collect();

        assert_eq!(visited, vec![add, mul, a, b, c]);
    }

    #[test]
    fn test_iter_expr_nodes() {
        let mut arena = ExprArena::new();

        let p = arena.add(ExprNode::new(
            NodeKind::Parameter(ParameterId::from(123)),
            (),
        ));

        let root = arena.add_root(p);

        let items: Vec<_> = arena.iter_expr_nodes(root).collect();

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, p);

        match items[0].1.kind {
            NodeKind::Parameter(id) => {
                assert_eq!(id, ParameterId::from(123));
            }
            _ => panic!("expected parameter node"),
        }
    }
}
