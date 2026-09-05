import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from mr_liu.place.anyplace import AnyPlaceBackend
from mr_liu.place.contracts import PlaceRequest,PlaceError
from mr_liu.grasp.transforms import invert_transform
from test_fine_place import Rig


class AnyPlaceBackendTests(unittest.TestCase):
    def rig(self):
        from dataclasses import replace
        rig=Rig();rng=np.random.default_rng(3)
        rig.held=replace(rig.held,reference_points_object=rng.uniform(-.015,.015,(1200,3)))
        evidence=rig.destination(PlaceRequest('region'))
        backend=AnyPlaceBackend()
        backend.parent_cloud=lambda e:e.scene_points
        return rig,evidence,backend

    def test_relative_transform_maps_input_object_pose_not_ee_pose(self):
        rig,evidence,backend=self.rig()
        pose=rig.pose@rig.held.T_ee_object;final=pose.copy();final[:3,3]=[.03,0,1.021]
        relative=final@invert_transform(pose)
        backend.client=SimpleNamespace(infer=lambda **kw:{'protocol':1,'sequence':evidence.observation.sequence,
                                                         'transforms':[relative.tolist()]})
        result=backend.select(PlaceRequest('region'),rig.held,evidence,rig.pose,.013)
        np.testing.assert_allclose(result,[.03,0])
        np.testing.assert_allclose(backend.last_metrics['selected_object_pose'],final)

    def test_orientation_cannot_be_silently_discarded(self):
        rig,evidence,backend=self.rig()
        relative=np.eye(4);relative[:2,:2]=[[0,-1],[1,0]]
        backend.client=SimpleNamespace(infer=lambda **kw:{'protocol':1,'sequence':evidence.observation.sequence,
                                                         'transforms':[relative.tolist()]})
        with self.assertRaisesRegex(PlaceError,'rotation_requires_pose_controller'):
            backend.select(PlaceRequest('region'),rig.held,evidence,rig.pose,.013)

    def test_stale_response_rejected(self):
        rig,evidence,backend=self.rig()
        backend.client=SimpleNamespace(infer=lambda **kw:{'protocol':1,'sequence':-1,'transforms':[]})
        with self.assertRaisesRegex(PlaceError,'sequence_mismatch'):
            backend.select(PlaceRequest('region'),rig.held,evidence,rig.pose,.013)

    def test_model_failure_does_not_fall_back_or_release(self):
        rig=Rig();node=rig.node();node.backend=Mock()
        node.backend.select.side_effect=PlaceError('no_executable_anyplace_candidate')
        node.backend.last_metrics={'place_backend':'anyplace'}
        result=node.execute(PlaceRequest('region'),rig.held)
        self.assertFalse(result.success);self.assertFalse(rig.opened)
        self.assertEqual(result.metrics['place_backend'],'anyplace')

    def test_parent_uses_only_destination_segmentation(self):
        from unittest.mock import patch
        points=np.array([[0,0,0],[1,1,1],[2,2,2]])
        evidence=SimpleNamespace(observation=None,mask=np.array([[True,False],[False,True]]))
        with patch('mr_liu.place.anyplace.scene_cloud',return_value=(points,np.array([0,0,1]),np.array([0,1,1]))):
            result=AnyPlaceBackend().parent_cloud(evidence)
        np.testing.assert_array_equal(result,points[[0,2]])


if __name__=='__main__':unittest.main()
