from os import path
import random

import cv2
import numpy as np


# todo: first prune everything
#       deepest leaf --> traverse backwards and then forward on any branches
# todo: identify "bottom" leaf as the arm/wrist.
# todo: identify up to five more leaves as fingertips
#       node is the wrist. then all fingertip angles measured relative to wrist


def demo():
    image = cv2.imread(path.join('tests', 'hand.png'))
    family_map = _make_family_map(image)
    trees, edges = treeify(family_map)

    for tree in trees:
        print '{family} ends: {count}'.format(family=tree['family'],
                                              count=len(tree['ends']))


def treeify(family_map):
    trees = list()
    rows, cols = family_map.shape
    remaining_points = set((r, c) for r in range(rows) for c in range(cols))
    edges = np.empty_like(family_map, dtype=object)
    for p in remaining_points:
        edges[p] = set()  # todo: better way to initialize with sets?
    # continue until all regions within the graph are handled
    while remaining_points:
        # grow a tree from any remaining point until complete
        start_p = remaining_points.pop()
        family = family_map[start_p]
        tree = {'family': family,
                'ends': set()}
        trees.append(tree)
        q = list()
        q.append((None, start_p))
        while q:

            # todo: remove debug
            try:
                debug_counter += 1
            except NameError:
                debug_counter = 1
            draw_step = int(round(rows * cols * 0.05))
            if not debug_counter % draw_step:
                debug_draw_vectors(family_map, edges, remaining_points)

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
            random.shuffle(expansion_pairs)  # further effort to avoid patterns
            q.extend(expansion_pairs)
            # store ends
            if not expansion_points:
                tree['ends'].add(p)
            # document all edges for this point
            if source is None:
                all_connections = list(expansion_points)
            else:
                all_connections = expansion_points + [source]
            edges[p].update(all_connections)

    # todo: remove debug
    debug_draw_vectors(family_map, edges, remaining_points)
    cv2.waitKey(0)

    # prune completely parallel branches
    pass
    return trees, edges


def _make_family_map(image):
    """Return an ndarray with keys identifying family of each point in image."""
    rows, cols = image.shape[0:2]  # ignore channels if any
    family_map = np.ndarray((rows, cols), dtype=object)
    all_points = ((r, c) for r in range(rows) for c in range(cols))
    for p in all_points:
        family_map[p] = _identify_family(p, image)
    return family_map


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


def debug_draw_vectors(family_map, edges, remaining_points):
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
    cv2.waitKey(1)


if __name__ == '__main__':
    # import cProfile
    # r = cProfile.run('demo()')
    demo()
