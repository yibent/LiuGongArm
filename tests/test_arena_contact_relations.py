import numpy as np

from mr_liu.arena.contact_relations import fixture_feature, contact_pose_candidates


def plane(xmin=-.07, xmax=.07, ymin=-.06, ymax=.06, z=.01):
    x, y = np.meshgrid(np.arange(xmin, xmax, .004), np.arange(ymin, ymax, .004))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]


def cylinder(cx, cy, radius=.007, low=.01, high=.075):
    angle, z = np.meshgrid(np.linspace(0, 2*np.pi, 48), np.linspace(low, high, 25))
    return np.c_[cx+radius*np.cos(angle.ravel()), cy+radius*np.sin(angle.ravel()), z.ravel()]


def test_peg_feature_selects_one_observed_component_near_preference():
    parent = np.r_[plane(), cylinder(-.04, 0), cylinder(.04, 0)]
    feature = fixture_feature(parent, 'sleeve_on_peg', [.06, 0, .2])
    assert feature['feature_count'] == 2
    assert np.linalg.norm(np.asarray(feature['centre'])[:2]-[.04, 0]) < .01


def test_insert_snaps_anyplace_orientation_to_observed_socket_centre_and_floor():
    angles, z = np.meshgrid(np.linspace(0, 2*np.pi, 80), np.linspace(.01, .05, 20))
    walls = np.c_[.03*np.cos(angles.ravel()), .03*np.sin(angles.ravel()), z.ravel()]
    parent = np.r_[plane(), walls]
    child = cylinder(.12, -.08, radius=.016, low=.10, high=.16)
    rows = contact_pose_candidates([np.eye(4)], child, parent, np.eye(4), np.eye(4),
                                   'insert', [0, 0, .3])
    placed = rows[0]['placed']
    centre = np.quantile(placed, [.02, .98], axis=0).mean(0)
    assert np.linalg.norm(centre[:2]) < .006
    assert abs(np.quantile(placed[:, 2], .02)-rows[0]['feature']['base_z']-.002) < .002


def test_sleeve_stops_at_partial_engagement_to_keep_fingers_above_fixture():
    parent = np.r_[plane(), cylinder(.04, 0)]
    child = cylinder(.10, -.10, radius=.02, low=.10, high=.14)
    rows = contact_pose_candidates([np.eye(4)], child, parent, np.eye(4), np.eye(4),
                                   'sleeve_on_peg', [.04, 0, .2])
    bottom = np.quantile(rows[0]['placed'][:, 2], .02)
    assert abs(bottom - (rows[0]['feature']['top_z']-.018)) < .002
    assert bottom > rows[0]['feature']['base_z'] + .025
