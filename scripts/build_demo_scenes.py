"""Reproducible industrial workbenches, with physical compound colliders.

Scene metadata is for the editor. Robot execution still grounds visual targets.
Run with usd-core; no simulation or model services are needed to build assets.
"""
import json
from pathlib import Path
import build_industrial_assets as assets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/scenes/demo'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assets.OUT = OUT
    # Low walls leave both the contents and the support visible to RGB-D.
    for number, color in [(1, (.03, .18, .65)), (2, (.75, .10, .08)), (3, (.08, .45, .16))]:
        p = assets.Part(number, color, .05, .5)
        p.box((.17, .14, .008), (0, 0, .004))
        for x in (-.082, .082): p.box((.006, .14, .020), (x, 0, .018))
        for y in (-.067, .067): p.box((.17, .006, .020), (0, y, .018))
        p.save()
    p = assets.Part(4, (.72, .73, .75), .65, .35)
    p.cylinder(.085, .008, (0, 0, .004)); p.ring(.085, .078, .024, z=.008); p.save()
    # Through sockets and pegs stay separate colliders, never convexified closed.
    p = assets.Part(5, (.16, .31, .56), .3, .4)
    p.box((.13, .12, .010), (0, 0, .005))
    p.ring(.030, .020, .035, z=.010); p.save()
    p = assets.Part(6, (.55, .57, .59), .8, .3)
    p.box((.14, .12, .010), (0, 0, .005))
    for x in (-.042, .042): p.cylinder(.007, .065, (x, 0, .0425))
    p.save()
    # A wrench with two open jaws, plus a screwdriver with an exposed handle.
    p = assets.Part(7, (.56, .59, .62), .8, .32)
    p.box((.070, .018, .012), (0, 0, .006))
    for x in (-.040, .040):
        p.box((.016, .034, .012), (x, 0, .006))
        for y in (-.014, .014): p.box((.026, .009, .012), (x + (.012 if x > 0 else -.012), y, .006))
    p.save()
    p = assets.Part(8, (.88, .48, .035), .1, .5)
    p.cylinder(.014, .055, (-.027, 0, .014), (0, 90, 0))
    p.box((.055, .007, .007), (.027, 0, .014)); p.save()

    base = json.loads((ROOT / 'configs/arena_panda_industrial.json').read_text())
    base['camera'].update(width=960, height=720)
    base['camera']['scene'].update(position=[.72, -.46, .78], lookat=[.51, 0, .02])
    base['camera']['side'].update(position=[.80, .55, .68], lookat=[.51, 0, .02])
    base['vision']['vocabulary'] = ['block', 'cylinder', 'nut', 'gear', 'shaft', 'bolt', 'bracket', 'sleeve', 'washer', 'wrench', 'screwdriver', 'tray', 'bowl', 'box', 'table']

    def usd(asset, xy, label, *, library='demo', dynamic=True, mass=.08):
        return {'usd_path': f'assets/scenes/{library}/asset_{asset:02d}.usda', 'position': [*xy, .001],
                'dynamic': dynamic, 'mass': mass, 'label': label}

    def solid(shape, xyz, size, color, label, dynamic=True):
        return {'shape': shape, 'position': xyz, 'size': size, 'color': color, 'label': label, 'dynamic': dynamic, 'mass': .08 if dynamic else 1.}

    block = lambda xy, color, label: solid('cuboid', [*xy, .020], [.038, .038, .040], color, label)
    cylinder = lambda xy, color, label: solid('cylinder', [*xy, .030], [.032, .032, .060], color, label)
    blue, red, green, silver = [.035, .18, .7], [.8, .07, .035], [.035, .55, .10], [.60, .63, .66]
    definitions = [
        ('sorting', '零件分拣', '彩色工件按颜色或形状分拣，配有红蓝料盘和开放桌面。', 'sort',
         ['将红色方块放到蓝色料盘里。', '拿起绿色方块，保持拿着。', '把手里的物体放到桌面空处。'],
         [block((.43, -.13), red, '红色方块'), block((.55, -.13), green, '绿色方块'),
          cylinder((.65, -.13), silver, '金属圆柱'), cylinder((.38, .01), blue, '蓝色圆柱'),
          usd(1, (.44, .19), '蓝色料盘', dynamic=False), usd(2, (.65, .19), '红色料盘', dynamic=False)]),
        ('machining', '机加工零件整理', '齿轮、轴、螺栓和轴套，配有料盘、圆盘和低支撑台。', 'parts',
         ['将橙色齿轮放到蓝色料盘里。', '把金属圆柱放到桌面空处。'],
         [usd(3, (.41, -.15), '橙色齿轮', library='industrial'), usd(4, (.56, -.14), '阶梯轴', library='industrial'),
          usd(5, (.66, -.06), '黄铜螺栓', library='industrial'), usd(6, (.38, .01), '青色轴套', library='industrial'),
          cylinder((.50, -.02), silver, '金属圆柱'), usd(1, (.40, .19), '蓝色料盘', dynamic=False),
          usd(4, (.62, .19), '金属圆盘', dynamic=False), solid('cuboid', [.70, -.19, .012], [.10, .10, .024], [.4, .42, .45], '低支撑台', False)]),
        ('packing', '分格装箱', '金属圆柱和方形工件，配有六格料箱、周转料盘和叠放底箱。', 'pack',
         ['拿起一个金属圆柱放到绿色料盘里。', '观察蓝色料箱有哪些空格。'],
         [cylinder((x, -.14), silver, '金属圆柱') for x in (.38, .47, .56)] +
         [block((.66, -.14), red, '红色方块'), usd(2, (.42, .17), '蓝色分格料箱', library='challenge_qa', dynamic=True, mass=.3),
          usd(3, (.65, .16), '绿色周转盘', dynamic=False), usd(2, (.70, -.01), '蓝色底箱', library='challenge_qa', dynamic=False)]),
        ('assembly', '工装装配', '圆柱销、轴套、垫圈和支架，配有插孔座、定位销架、挂钩及暂存盘。', 'fixture',
         ['把金属圆柱放到蓝色料盘里。', '拿起青色轴套，保持拿着。'],
         [cylinder((.43, -.14), silver, '金属圆柱'), usd(6, (.55, -.14), '青色轴套', library='industrial'),
          usd(7, (.66, -.14), '垫圈', library='industrial'), usd(1, (.37, -.02), '绿色支架', library='industrial'),
          usd(5, (.42, .17), '插孔座', dynamic=False), usd(6, (.59, .16), '定位销架', dynamic=False),
          usd(10, (.74, .15), '挂钩工装', library='industrial', dynamic=False), usd(1, (.59, .0), '蓝色暂存盘', dynamic=False)]),
        ('tools', '工具归位', '扳手、螺丝刀、螺母和工件，配有双料盘及圆柱支撑台。', 'tools',
         ['将红色方块放到蓝色料盘里。', '拿起黄色螺丝刀，保持拿着。'],
         [usd(7, (.42, -.15), '扳手'), usd(8, (.59, -.15), '黄色螺丝刀'),
          usd(2, (.38, -.02), '六角螺母', library='industrial'), block((.52, -.02), red, '红色方块'),
          usd(1, (.42, .19), '蓝色料盘', dynamic=False), usd(3, (.65, .19), '绿色料盘', dynamic=False),
          solid('cylinder', [.70, -.01, .025], [.08, .08, .05], [.85, .67, .05], '黄色支撑柱', False)]),
    ]
    catalog = []
    directory = ROOT / 'configs/scenes'; directory.mkdir(exist_ok=True)
    for ident, name, description, kind, examples, rows in definitions:
        meta = {'id': ident, 'name': name, 'description': description, 'kind': kind, 'examples': examples, 'count': len(rows), 'config': f'configs/scenes/{ident}.json'}
        config = {**base, 'scene': {k: v for k, v in meta.items() if k != 'config'}, 'entities': [{**row, 'name': f'part_{i:02d}'} for i, row in enumerate(rows, 1)]}
        (directory / f'{ident}.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        catalog.append(meta)
    qa = json.loads((ROOT / 'configs/arena_panda_challenge_qa.json').read_text())
    meta = {'id': 'challenge_qa', 'name': '挑战杯综合验收', 'description': '多区域密集来料、翻正、六格装箱及整箱叠放，用于综合能力验证。', 'kind': 'pack', 'examples': ['观察来料最多的区域。'], 'count': len(qa['entities']), 'config': 'configs/scenes/challenge_qa.json'}
    qa['scene'] = {k: v for k, v in meta.items() if k != 'config'}
    (directory / 'challenge_qa.json').write_text(json.dumps(qa, ensure_ascii=False, indent=2) + '\n')
    catalog.append(meta)
    (ROOT / 'configs/industrial_scenes.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__': main()
