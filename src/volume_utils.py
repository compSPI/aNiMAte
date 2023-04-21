import torch
from pytorch3d.transforms import Rotate, Transform3d
from pot_utils import Potential

''' Volumes '''

class ExplicitAtomicVolume(torch.nn.Module):
    def __init__(self, atomic_model, sidelen, pixel_size, log_dir):
        super(ExplicitAtomicVolume, self).__init__()
        self.sidelen = sidelen
        self.atomic_model = atomic_model

        self.potential = Potential(sidelen=sidelen, pixel_sz=pixel_size, log_dir=log_dir)

    def forward(self, rotmat, nma_coordinates={}, global_pose={}, global_nma=None):
        nma_coords = nma_coordinates
        if global_nma is not None:
            nma_coords = global_nma + nma_coordinates
        deformed_coords, ff_a, ff_b = self.atomic_model(nma_coords)

        if 'global_rotmat' in global_pose and 'global_shift' in global_pose:
            t = Transform3d(device=deformed_coords.device).\
                compose(Rotate(global_pose['global_rotmat'], device=deformed_coords.device)).\
                translate(global_pose['global_shift'])
            deformed_coords = t.transform_points(deformed_coords)

        t = Rotate(rotmat, device=deformed_coords.device)
        rotated_coords = t.inverse().transform_points(deformed_coords) # rotation of atoms is the reverse of the grid

        potential_2d =  self.potential(rotated_coords, ff_a, ff_b)
        return potential_2d

    def make_volume(self):
        coords, ff_a, ff_b = self.atomic_model.get_atomic_model()
        potential_3d = self.potential.calculate_3d(coords, ff_a, ff_b)
        return potential_3d
