import openseespy.opensees as ops


def find_void_nodes(remove: bool = False):
    """
    Finds free nodes in the model, i.e. nodes that are not attached to any element.

    Parameters
    ----------
    remove: bool, default=False
        If True, the function will remove the free nodes from the model.

    Returns
    -------
    free_node_tags: list, a list containing all free nodes.
    """
    ele_tags = ops.getEleTags()
    fixed_node_tags = ops.getFixedNodes()
    all_node_valid = []
    for etag in ele_tags:
        nodes = ops.eleNodes(etag)
        all_node_valid.extend(nodes)
    all_node_valid += fixed_node_tags
    node_tags = ops.getNodeTags()
    node_invalid = list(set(node_tags) - set(all_node_valid))
    if len(node_invalid) == 0:
        print("Info:: The model has no free nodes!")
    else:
        if remove:
            for ntag in node_invalid:
                ops.remove("node", ntag)
            print(f"Info:: Free nodes with tags {node_invalid} have been removed!")
    return node_invalid


def remove_void_nodes():
    """
    Removes free node from the model, i.e. nodes that are not attached to any element.

    Returns
    -------
    free_node_tags: list, a list containing all free nodes.
    """
    return find_void_nodes(remove=True)
