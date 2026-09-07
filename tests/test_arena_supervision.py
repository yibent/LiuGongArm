"""Boot configuration contracts; no GPU or process signalling in these tests."""
import configparser
import importlib.util
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('arena_stack', ROOT/'ops/arena_stack.py')
stack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stack)


def test_saved_scene_survives_an_empty_boot_environment(tmp_path):
    with patch.object(stack, 'STATE', tmp_path):
        chosen = stack.scene_environment({}, 'configs/arena_panda_industrial.json')
        boot = stack.scene_environment({})
        assert boot['ARENA_PANDA_CONFIG'] == chosen['ARENA_PANDA_CONFIG']
        assert Path(boot['ARENA_PANDA_CONFIG']).is_absolute()


def test_explicit_scene_change_is_saved_for_next_boot(tmp_path):
    with patch.object(stack, 'STATE', tmp_path):
        stack.scene_environment({}, 'configs/arena_panda_industrial.json')
        changed = stack.scene_environment({}, 'configs/arena_panda.json')
        assert stack.scene_environment({}) == changed


def test_every_service_has_one_foreground_supervisor_owner():
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT/'ops/arena-supervisord.conf')
    names = {s.split(':', 1)[1] for s in config.sections() if s.startswith('program:')}
    assert names == {'database', *stack.SERVICES}
    for name in names:
        program = config['program:'+name]
        assert program.getboolean('autostart')
        assert program.getboolean('autorestart')
        assert program.getboolean('stopasgroup')
        assert program.getboolean('killasgroup')
        assert program['command'].endswith('arena_supervisor.py '+name)
    assert '0700' == config['unix_http_server']['chmod']


def test_cloud_boot_owns_only_the_project_manager():
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT/'ops/arena-cloud-boot.conf')
    assert config.sections() == ['program:liugong-stack']
    program = config['program:liugong-stack']
    assert 'supervisord -n -c ' in program['command']
    assert program.getboolean('autostart') and program.getboolean('autorestart')
