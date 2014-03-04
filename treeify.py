from collections import deque
from os import path
import random

import cv2
import numpy as np

# todo: just pruning will remove all branches...???
# todo: speed-up: avoid for loops. use where, logical_and etc.


# todo O: sides before corners
# todo: randomize sides / corners to avoid biased travel
# todo: deepest leaf --> traverse backwards and then forward on any branches
# todo: seems like need to store links both ways to do trim traversal


# todo: yes, store each node as all outgoing rather than parent/children
#       so tree can be emergent depending on direction
# todo: after pruning, identify "bottom" leaf as the arm/wrist. The next
# todo: identify up to five more leaves as fingertips
#       node is the wrist. then all fingertip angles measured relative to wrist

ROOT = '<< root >>'


def demo():
    image = cv2.imread(path.join('tests', 'test.png'))
    family_map = _make_family_map(image)
    tree = treeify(family_map)


def treeify(family_map, specified_roots=None):
    incoming = dict()
    roots = list()
    joints = list()
    leaves = list()
    rows, cols = family_map.shape
    remaining_points = set((r, c) for r in range(rows) for c in range(cols))
    # continue until all regions within the graph are handled
    while remaining_points:
        # grow a tree from root points or any remaining point if not povided
        if specified_roots:
            root_p = specified_roots.pop()
            remaining_points.remove(root_p)
        else:
            root_p = remaining_points.pop()
        roots.append(root_p)
        incoming[root_p] = ROOT
        q = deque()
        q.append(root_p)
        while q:

            # todo: remove debug
            # try:
            #     debug_counter += 1
            # except NameError:
            #     debug_counter = 1
            # draw_step = int(round(rows * cols * 0.05))
            # if not debug_counter % draw_step:
            #     debug_draw_vectors(family_map, incoming, roots, joints, leaves)

            p = q.popleft()  # pushright + popleft --> breadth first expansion
            family = family_map[p]
            try:
                remaining_points.remove(p)
            except KeyError:
                pass  # already removed
            # handle each neighbor of this point
            qualified_n = lambda n_point: all((n_point not in incoming,
                                               n_point not in q,
                                               family_map[n_point] == family))
            # expansion_sides = [n for n in _neighbors(p, 'sides', family_map,
            #                                          include_oob=False)
            #                    if qualified_n(n)]
            # expansion_corners = [n for n in _neighbors(p, 'corners', family_map,
            #                                            include_oob=False)
            #                      if qualified_n(n)]
            # random.shuffle(expansion_sides)
            # random.shuffle(expansion_corners)
            expansion = [n for n in _neighbors(p, 'all', family_map,
                                               include_oob=False)
                         if qualified_n(n)]
            random.shuffle(expansion)
            expansion_count = len(expansion)
            if expansion_count == 0:
                # this branch is done (tree may still have more to go)
                leaves.append(p)
            elif expansion_count > 1:
                joints.append(p)
            # build the tree and add expansion
            for n in expansion:
                incoming[n] = p
                q.append(n)

    # todo: remove debug
    debug_draw_vectors(family_map, incoming, roots, joints, leaves)
    print 'roots: {}'.format(len(roots))
    print 'joints: {}'.format(len(joints))
    print 'leaves: {}'.format(len(leaves))
    # prune completely parallel branches
    pass
    raw_input('done')


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


def debug_draw(family_map, incoming, roots, joints, leaves):
    # common parts
    rows, cols = family_map.shape
    # map bases
    dark_gray = (100, 100, 100)
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    on_family_map = family_map == 'on'
    off_family_map = family_map == 'off'
    handled_map = np.empty_like(family_map)
    handled_map[zip(*incoming)] = True
    # family base color
    on_family_points = np.nonzero(on_family_map)
    image[on_family_points] = dark_gray
    # all handled points
    dark_green = (25, 100, 25)
    dark_red = (25, 25, 100)
    image[np.where(np.logical_and(handled_map, on_family_map))] = dark_green
    image[np.where(np.logical_and(handled_map, off_family_map))] = dark_red
    # all special points
    white = (255, 255, 255)
    for joint in joints:
        image[joint] = white
    green = (0, 255, 0)
    red = (0, 0, 255)
    for leaf in leaves:
        color = green if family_map[leaf] == 'on' else red
        image[leaf] = color
    yellow = (0, 255, 255)
    blue = (255, 0, 0)
    for root in roots:
        color = blue if family_map[root] == 'on' else yellow
        image[root] = color
    # scale for ease of inspection
    scale = 8
    image = cv2.resize(image, (cols*scale, rows*scale),
                       interpolation=cv2.INTER_NEAREST)
    cv2.imshow('asdf', image)
    cv2.waitKey(1)


def debug_draw_vectors(family_map, incoming, roots, joints, leaves):
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
    handled_map[zip(*incoming)] = True
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
    for child, parent in incoming.items():
        color = green if family_map[child] == 'on' else red
        if parent == ROOT:
            continue  # skip root
        cv2.line(image_v, center_p_cv(child), center_p_cv(parent), color)

    # # joints
    # white = (255, 255, 255)
    # for joint in joints:
    #     image_v[center_p(joint)] = white
    # # leaves
    # green = (0, 255, 0)
    # red = (0, 0, 255)
    # for leaf in leaves:
    #     color = green if family_map[leaf] == 'on' else red
    #     image_v[center_p(leaf)] = color
    # # roots
    # yellow = (0, 255, 255)
    # blue = (255, 0, 0)
    # for root in roots:
    #     color = blue if family_map[root] == 'on' else yellow
    #     image_v[center_p(root)] = color

    # scale to useful size
    image_v_h, image_v_w = image_v.shape[0:2]
    h_limit, w_limit = 600, 1000
    ratio_center = float(h_limit) / w_limit
    ratio = float(image_v_h) / image_v_w
    print ratio_center, ratio
    if ratio > ratio_center:
        # height is the limiting factor
        target_h, target_w = h_limit, int(round(h_limit / ratio))
    else:
        target_h, target_w = int(round(w_limit / ratio)), w_limit
    print target_h, target_w
    image_v_scaled = cv2.resize(image_v, (target_w, target_h),
                                interpolation=cv2.INTER_AREA)
    cv2.imshow('asdf', image_v_scaled)
    cv2.waitKey(1)


if __name__ == '__main__':
    import cProfile
    r = cProfile.run('demo()')
    print r
    #
    # demo()
