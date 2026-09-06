"""Build original, metre-scale USD parts for the Arena industrial workbench.

Run with a Python containing usd-core (or Isaac Sim's Python). Each file has
one rigid root and compound convex colliders; holes remain physically open.
No class tags, grasp annotations, controller poses or semantic mappings.
"""
from pathlib import Path
import math
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/scenes/industrial'


class Part:
    def __init__(self, number, color, metallic=.6, roughness=.32):
        self.path = OUT / f'asset_{number:02d}.usda'
        self.stage = Usd.Stage.CreateNew(str(self.path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.)
        self.root = UsdGeom.Xform.Define(self.stage, '/Asset').GetPrim()
        self.stage.SetDefaultPrim(self.root)
        UsdPhysics.RigidBodyAPI.Apply(self.root)
        UsdPhysics.MassAPI.Apply(self.root).CreateMassAttr(.08)
        self.material = UsdShade.Material.Define(self.stage, '/Asset/Material')
        shader = UsdShade.Shader.Define(self.stage, '/Asset/Material/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
        self.material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        physics = UsdPhysics.MaterialAPI.Apply(self.material.GetPrim())
        physics.CreateStaticFrictionAttr(1.2)
        physics.CreateDynamicFrictionAttr(1.)
        physics.CreateRestitutionAttr(0.)
        self.count = 0

    def name(self):
        self.count += 1
        return f'/Asset/piece_{self.count:03d}'

    def finish(self, geom, position=(0, 0, 0), rotate=(0, 0, 0)):
        xform = UsdGeom.Xformable(geom)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotate))
        prim = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(self.material)
        binding.Bind(self.material, materialPurpose='physics')
        return xform

    def box(self, size, position, rotate=(0, 0, 0)):
        geom = UsdGeom.Cube.Define(self.stage, self.name())
        geom.CreateSizeAttr(1.)
        self.finish(geom, position, rotate).AddScaleOp().Set(Gf.Vec3f(*size))

    def cylinder(self, radius, height, position, rotate=(0, 0, 0)):
        geom = UsdGeom.Cylinder.Define(self.stage, self.name())
        geom.CreateRadiusAttr(radius); geom.CreateHeightAttr(height)
        geom.CreateAxisAttr('Z')
        self.finish(geom, position, rotate)

    def prism(self, outline, height, z=0):
        """A convex polygon extruded upward; each ring sector is convex."""
        n = len(outline)
        points = [(x, y, level) for level in (z, z+height) for x, y in outline]
        faces = [list(reversed(range(n))), list(range(n, 2*n))]
        faces += [[i, (i+1)%n, (i+1)%n+n, i+n] for i in range(n)]
        geom = UsdGeom.Mesh.Define(self.stage, self.name())
        geom.CreatePointsAttr(points)
        geom.CreateFaceVertexCountsAttr([len(f) for f in faces])
        geom.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        geom.CreateSubdivisionSchemeAttr('none')
        self.finish(geom)

    def ring(self, outer, inner, height, z=0, hexagon=False):
        count = 24
        outer_points = []
        for i in range(count):
            if hexagon:
                side, t = divmod(i, 4)
                a, b = 2*math.pi*side/6, 2*math.pi*(side+1)/6
                outer_points.append((outer*((1-t/4)*math.cos(a)+t/4*math.cos(b)),
                                     outer*((1-t/4)*math.sin(a)+t/4*math.sin(b))))
            else:
                a = 2*math.pi*i/count
                outer_points.append((outer*math.cos(a), outer*math.sin(a)))
        for i in range(count):
            j = (i+1)%count
            a, b = 2*math.pi*i/count, 2*math.pi*j/count
            self.prism([outer_points[i], outer_points[j],
                        (inner*math.cos(b), inner*math.sin(b)),
                        (inner*math.cos(a), inner*math.sin(a))], height, z)

    def save(self):
        self.stage.GetRootLayer().Save()
        print(f'{self.path.relative_to(ROOT)}: {self.count} colliders')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Painted angle bracket with an open rectangular mounting slot.
    p = Part(1, (.035, .42, .08), .15)
    p.box((.064, .05, .008), (0, 0, .004))
    p.box((.018, .008, .042), (-.023, .021, .029))
    p.box((.018, .008, .042), (.023, .021, .029))
    p.box((.028, .008, .012), (0, .021, .044)); p.save()
    # Large hex nut with a through-hole.
    p = Part(2, (.52, .55, .58), .85)
    p.ring(.029, .014, .026, hexagon=True); p.save()
    # Coated spur gear, 16 teeth, open central bore.
    p = Part(3, (.75, .22, .035), .45)
    p.ring(.029, .011, .016)
    for i in range(16):
        angle = 360*i/16; a = math.radians(angle)
        p.box((.009, .008, .016), (.031*math.cos(a), .031*math.sin(a), .008), (0, 0, angle))
    p.save()
    # Stepped shaft with two shoulders, lying along X.
    p = Part(4, (.40, .44, .48), .9)
    p.cylinder(.012, .095, (0, 0, .02), (0, 90, 0))
    p.cylinder(.020, .014, (-.019, 0, .02), (0, 90, 0))
    p.cylinder(.020, .014, (.019, 0, .02), (0, 90, 0)); p.save()
    # Brass hex-head bolt, standing on its head, with modeled thread ridges.
    p = Part(5, (.68, .45, .10), .8)
    p.prism([(.023*math.cos(i*math.pi/3), .023*math.sin(i*math.pi/3)) for i in range(6)], .013)
    p.cylinder(.009, .046, (0, 0, .036))
    for i in range(8): p.cylinder(.0104, .0015, (0, 0, .018+i*.005))
    p.save()
    # Blue-green bearing sleeve with a visible hollow bore.
    p = Part(6, (.035, .38, .43), .5)
    p.ring(.023, .013, .04); p.save()
    # Dark washer, placed close to the silver nut for a crowded region.
    p = Part(7, (.075, .085, .10), .75)
    p.ring(.025, .012, .008); p.save()
    # Open parts trays; base and walls belong to one rigid body.
    for number, color in [(8, (.025, .15, .48)), (9, (.85, .40, .025))]:
        p = Part(number, color, .05, .48)
        p.box((.18, .16, .008), (0, 0, .004))
        for x in (-.087, .087): p.box((.006, .16, .024), (x, 0, .020))
        for y in (-.077, .077): p.box((.18, .006, .024), (0, y, .020))
        p.save()
    # Fixture plate with three pegs and a hook; geometry for future tasks.
    p = Part(10, (.27, .30, .33), .75)
    p.box((.12, .11, .012), (0, 0, .006))
    for x in (-.038, 0, .038): p.cylinder(.007, .075, (x, .026, .0495))
    p.box((.012, .012, .10), (0, -.028, .062))
    p.box((.012, .035, .012), (0, -.044, .106)); p.save()


if __name__ == '__main__': main()
