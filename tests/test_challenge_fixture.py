"""The audit must start from a physical layout, not interpenetrating assets."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def test_fallen_bin_part_spawns_clear_of_dividers_before_gravity_settles_it():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/arena_panda_challenge_qa.json').read_text())
    rows={r['name']:r for r in config['entities']}
    part=rows['qa_body_11'];bin_=rows['qa_body_12']
    # Sample the fixture's sleeve shell (used by this independent fixture
    # audit only, never passed into runtime perception or grasp selection).
    a=np.linspace(0,2*np.pi,96);z=np.linspace(0,.060,30)
    zz,aa=np.meshgrid(z,a)
    shell=np.c_[.014*np.cos(aa.ravel()),.014*np.sin(aa.ravel()),zz.ravel()]
    world=Rotation.from_quat(part['orientation']).apply(shell)+part['position']
    assert world[:,2].min()>bin_['position'][2]+.040
