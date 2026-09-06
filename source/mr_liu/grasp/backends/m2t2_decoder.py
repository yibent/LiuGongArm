"""M2T2 published grasp formula with an unambiguous vector dimension.

torch.cross without dim chooses the first dimension of size 3. For exactly
three predictions this crosses different samples instead of XYZ directions.
All model axes, contact points, widths and scores remain unchanged.
"""


def build_6d_grasp(contact_pt, contact_dir, approach_dir, offset, gripper_depth=0.1034):
    import torch
    grasp = torch.stack((contact_dir,
                         torch.cross(approach_dir, contact_dir, dim=-1),
                         approach_dir,
                         contact_pt + contact_dir * offset.unsqueeze(-1) / 2
                         - gripper_depth * approach_dir), dim=-1)
    last = grasp.new_tensor([0, 0, 0, 1]).expand(*grasp.shape[:-2], 1, 4)
    return torch.cat((grasp, last), dim=-2)
