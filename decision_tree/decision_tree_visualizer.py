import graphviz

def visualize_children(dot, color_classes, label_classes, root_node):
    if root_node.left is not None and root_node.right is not None:
        
        dot.attr('node', shape='rectangle', color=color_classes[root_node.left.clazz])
        dot.node(root_node.left.graph_id(), root_node.left.displayName(label_classes))

        dot.attr('node', shape='rectangle', color=color_classes[root_node.right.clazz])
        dot.node(root_node.right.graph_id(), root_node.right.displayName(label_classes))

        dot.edge(root_node.graph_id(), root_node.left.graph_id())
        dot.edge(root_node.graph_id(), root_node.right.graph_id())

        visualize_children(dot, color_classes, label_classes, root_node.left)
        visualize_children(dot, color_classes, label_classes, root_node.right)

def visualize_result(root_node, color_classes, label_classes, save_pdf=False):

    dot = graphviz.Digraph(comment='decision_tree')
    dot.attr('node', shape='rectangle')
    dot.node(root_node.graph_id(), root_node.displayName())

    visualize_children(dot, color_classes, label_classes, root_node)

    if save_pdf:
        dot.render('decision_tree_output', view=True)  

    return dot
