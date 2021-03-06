from collections import deque
from os import path
import random

import cv2
import numpy as np

# todo: add target family argument
# todo: add option to expand to all or only sides (e.g. large areas limit to sides)
# todo: can't neighbors be simplified?


def demo():
    image = cv2.imread(path.join('tests', 'hand.png'))
    trees, edges = treeify(image, ['on'])


def treeify(image, target_families=None):
    # family map is used everywhere to identify pixels, regions, etc.
    family_map = _make_family_map(image)
    target_families = target_families or np.unique(family_map)
    trees = list()
    rows, cols = family_map.shape
    remaining_points = set((r, c) for r in range(rows) for c in range(cols)
                           if family_map[r, c] in target_families)
    edges = np.empty_like(family_map, dtype=object)
    edges.flat = [set() for _ in edges.flat]
    # continue until all regions within the graph are handled
    while remaining_points:
        # grow a tree from any remaining point until complete
        start_p = remaining_points.pop()
        # todo: should be refactored from here into _full_tree or similar
        family = family_map[start_p]
        tree = {'family': family,
                'any_point': start_p}
        trees.append(tree)
        q = list()
        q.append((None, start_p))
        while q:
            # pushright + popleft --> breadth first expansion
            # random within left part of q - roughly BFS with less pattern
            source, p = q.pop(random.randrange(0, max(1, len(q)//2)))
            try:
                remaining_points.remove(p)
            except KeyError:
                pass  # tree start point is always already gone
            # send qualifying neighbors for expansion
            q_points = tuple(qp for sp, qp in q)
            expansion_points = [n for n in _neighbors(p, 'all', family_map)
                                if all((n != source,
                                        n in remaining_points,
                                        n not in q_points,
                                        family_map[n] == family))]
            expansion_pairs = [(p, n) for n in expansion_points]
            q.extend(expansion_pairs)
            # document all edges for this point
            if source is None:
                all_connections = list(expansion_points)
            else:
                all_connections = expansion_points + [source]
            edges[p].update(all_connections)

    # prune all but "best" branches within an area
    # work on each tree
    for tree in trees:
        # todo: should be refactored from here to _simplify_tree() or similar
        family = tree['family']
        # root graph at one end of the longest path in the graph
        distant_point = _most_distant_node(tree['any_point'], edges)
        # for choosing best paths: document the height of every pixel
        heights = _heights(distant_point, edges)
        remaining_leaves = set(_leaves(distant_point, edges))
        # repeatedly look for a leaf and decide to keep it or prune its branch
        # stop when no leaves are pruned
        while remaining_leaves:
            leaf = remaining_leaves.pop()
            # identify any degenerate path to next branching pixel
            # this path is ignored when testing for nearby branches
            ignore = set(_identify_degenerate_branch(leaf, edges))
            # BFS expansion to find nearby other branches
            expansion_q = deque()
            expansion_q.append(leaf)
            while expansion_q:
                p = expansion_q.popleft()  # pushright + popleft for BFS
                ignore.add(p)

# todo: this debug shows each step of prune testing
                # debug_draw_vectors(family_map,edges, tuple(), ignore, leaf)

                # decide what to do with each neighbor
                for n in _neighbors(p, 'sides', family_map):
                    if n in ignore:
                        continue  # already decided to ignore this point
                    elif n in expansion_q:
                        continue  # already slated for expansion testing
                    elif family_map[n] != family:
                        ignore.add(n)  # ignore other families
                        continue
                    elif len(edges[n]) == 0:
                        expansion_q.append(n)  # expand into empty spaces
                    elif _disqualified(leaf, n, edges, heights):

# todo: this debug shows the final result of each pruning
                        # debug_draw_vectors(family_map,edges, tuple(), ignore, distant_point)

                        _prune_branch_of_leaf(leaf, edges, heights)
                        expansion_q.clear()  # this leaf is done. stop looking
                        break
                    else:
                        expansion_q.append(n)

# todo: this debug shows the final result of the whole process
    debug_draw_vectors(family_map, edges, tuple(), tuple())

    return trees, edges


def _disqualified(leaf, compare_point, edges, heights):
    """Return True if compare point is "better" than leaf as a major path."""
    if not _is_leaf(leaf, edges):
        raise ValueError('{} does not seem to be a leaf'.format(leaf))
    if heights[compare_point] >= heights[leaf]:
        return True
    return False


def _make_family_map(image):
    """Return an ndarray with keys identifying family of each point in image."""
    rows, cols = image.shape[0:2]  # ignore channels if any
    family_map = np.ndarray((rows, cols), dtype=object)
    all_points = ((r, c) for r in range(rows) for c in range(cols))
    for p in all_points:
        family_map[p] = _identify_family(p, image)
    return family_map


# todo: replace this with whatever you want to break up image into regions
def _identify_family(point, image):
    """Return a key identifying the family of point within image.

    This can be replaced with any kind of neighbor-independent identification
    and any number of families.
    """
    maximum_off_intensity = 0
    if np.mean(image[point]) <= maximum_off_intensity:
        return 'off'
    else:
        return 'on'


def _make_incoming_map(image):
    rows, cols = image.shape[0:2]  # ignore channels if any
    incoming_map = np.ndarray((rows, cols), dtype=object)
    incoming_map.fill(None)
    return incoming_map


def _neighbors(center, which_neighbors, image, include_oob=False):
    """Return all the neighboring points of center.

    Arguments:
    which_neighbors: 'all', 'sides', or 'corners'
    image: anything that implements shape with rows, cols as first two items
    include_oob: include out of bounds points in result or not
    """
    # confirm valid point (fails with Value Error if invalid)
    _is_out_of_bounds(center, image)
    # identify horizontal and vertical rails
    row, col = center
    t, vmid, b = row-1, row, row+1
    l, hmid, r = col-1, col, col+1
    # build the neighbor coordinates
    topleft, top, topright = (t, l), (t, hmid), (t, r)
    left, right = (vmid, l), (vmid, r)
    botleft, bot, botright = (b, l), (b, hmid), (b, r)
    # build the right set
    if which_neighbors == 'sides':
        with_ob_set = top, right, bot, left
    elif which_neighbors == 'corners':
        with_ob_set = topleft, topright, botright, botleft
    elif which_neighbors == 'all':
        with_ob_set = (topleft, top, topright,
                       right, botright, bot, botleft, left)
    else:
        raise ValueError('Unknown which_neighbors switch: ' + which_neighbors)
    # build the final set (no real need to generate)
    final_neighbors = list()
    for neighbor_point in with_ob_set:
        try:
            oob = _is_out_of_bounds(neighbor_point, image)  # error --> ValueErr
        except ValueError:
            continue  # this neighbor is outside the valid out of bounds area
        if include_oob or (not oob):
            final_neighbors.append(neighbor_point)
    return final_neighbors


def _is_out_of_bounds(point, image):
    """Return True if the point is on the valid 1-point border around image."""
    r, c = point
    r_min, r_max = 0, image.shape[0]-1
    c_min, c_max = 0, image.shape[1]-1
    # confirm it is outside the image shape
    if any((r < r_min,
            r > r_max,
            c < c_min,
            c > c_max)):
        # it is outside the image shape. confirm it is not too far outside
        if any((r < r_min-1,
                r > r_max+1,
                c < c_min-1,
                c > c_max+1)):
            raise ValueError('Out of bounds but beyond 1-point valid border.')
        else:
            return True
    # by default it is in-bounds
    return False


def _traverse_tree(root, edges):
    q = deque()
    q.append((None, root, 0))  # start depth zero
    while q:
        parent, point, depth = q.pop()
        q.extend((point, n, depth+1) for n in edges[point] if n != parent)
        yield point, depth, parent


def _leaves(root, edges):
    for p, depth, _ in _traverse_tree(root, edges):
        if p == root:
            continue  # skip the root
        if _is_leaf(p, edges):
            yield p


def _is_leaf(node, edges):
    return len(edges[node]) <= 1


def _heights(root, edges):
    bottom_up_nodes = reversed(list(_traverse_tree(root, edges)))
    heights = dict()
    for point, depth, parent in bottom_up_nodes:
        # set subtree depth 0 for leaves
        if _is_leaf(point, edges):
            heights[point] = 0
        # ignore the root
        if parent is None:
            continue
        # set/reset/ignore parent
        current_height = heights[point]
        heights[parent] = max(1+current_height,
                              heights.get(parent, 0))
    return heights


def _most_distant_node(start_p, edges):
    max_depth_node, max_depth = start_p, 0
    for p, depth, _ in _traverse_tree(start_p, edges):
        if (max_depth is None) or (depth > max_depth):
            max_depth_node, max_depth = p, depth
    return max_depth_node


def _prune_branch_of_leaf(leaf, edges, depths):
    if not _is_leaf(leaf, edges):
        raise ValueError('{} does not seem to be a leaf'.format(leaf))
    degenerate_points = _identify_degenerate_branch(leaf, edges)
    # clear neighbors of references to this branch
    ends = degenerate_points[0], degenerate_points[-1]
    for end in ends:
        for n in edges[end]:
            try:
                edges[n].remove(end)
            except KeyError:
                pass
    # then clear all connections and depths for this branch
    for p in degenerate_points:
        edges[p].clear()
        depths[p] = 0


def _identify_degenerate_branch(leaf, edges):
    if not _is_leaf(leaf, edges):
        raise ValueError('{} does not seem to be a leaf'.format(leaf))
    degenerate_branch = [leaf]
    try:
        parent, point = leaf, next(iter(edges[leaf]))  # take the first step
    except StopIteration:
        # the leaf is actually just a point. still qualifies as d.branch
        return degenerate_branch
    else:
        # it wasn't just an isolated point leaf so look for the branch
        while len(edges[point]) in (1, 2):  # 2-->path; 1-->whole tree is degen.
            degenerate_branch.append(point)
            try:
                a, b = edges[point]
            except ValueError:
                break  # the whole tree is degenerate. return it as is
            old_point = point
            point = a if a != parent else b
            parent = old_point
    return degenerate_branch


def debug_draw_vectors(family_map, edges, remaining_points,
                       other_points=None, target_point=None):
    other_points = other_points or tuple()
    # common parts
    rows, cols = family_map.shape
    side = 13
    side_center = side//2
    assert side % 2  # must be odd
    resizer = lambda a: np.repeat(np.repeat(a, side, axis=0), side, axis=1)
    center_p_cv = lambda p: (side*p[1] + side_center,
                             side*p[0] + side_center)
    # map bases
    dark_gray = (100, 100, 100)
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    on_family_map = family_map == 'on'
    off_family_map = family_map == 'off'
    handled_map = np.empty_like(family_map)
    handled_map.fill(True)
    handled_map[zip(*remaining_points)] = False
    # family base color
    on_family_points = np.nonzero(on_family_map)
    image[on_family_points] = dark_gray
    # all handled points
    dark_green = (25, 100, 25)
    dark_red = (25, 25, 100)
    image[np.where(np.logical_and(handled_map, on_family_map))] = dark_green
    image[np.where(np.logical_and(handled_map, off_family_map))] = dark_red
    # all "other" points of interest
    yellow = (0, 255, 255)
    image[zip(*other_points)] = yellow
    # specific target point
    blue = (255, 0, 0)
    if target_point:
        image[target_point] = blue
    # scale up and start making detailed additions
    image_v = resizer(image)
    # connectors
    green = (0, 255, 0)
    red = (0, 0, 255)
    completed_edges = set()
    all_p = ((r, c) for r in range(rows) for c in range(cols))
    for center_p in all_p:
        for connected_point in edges[center_p]:
            edge_pair = tuple(sorted((center_p, connected_point)))
            if edge_pair in completed_edges:
                continue
            completed_edges.add(edge_pair)
            color = green if family_map[center_p] == 'on' else red
            cv2.line(image_v, center_p_cv(connected_point),
                     center_p_cv(center_p), color)
    # scale to useful size
    image_v_h, image_v_w = image_v.shape[0:2]
    h_limit, w_limit = 600, 1000
    ratio_center = float(h_limit) / w_limit
    ratio = float(image_v_h) / image_v_w
    if ratio > ratio_center:
        # height is the limiting factor
        target_h, target_w = h_limit, int(round(h_limit / ratio))
    else:
        target_h, target_w = int(round(w_limit * ratio)), w_limit
    image_v_scaled = cv2.resize(image_v, (target_w, target_h),
                                interpolation=cv2.INTER_AREA)
    cv2.imshow('asdf', image_v_scaled)
    cv2.waitKey(0)


if __name__ == '__main__':
    # import cProfile
    # r = cProfile.run('demo()')
    demo()
