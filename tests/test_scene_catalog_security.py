"""Scene catalog and public-route safety checks."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scene_catalog_has_distinct_complete_industrial_scenes():
    scenes = json.loads((ROOT / "configs/industrial_scenes.json").read_text())
    assert len(scenes) >= 5
    configurations = []
    for scene in scenes:
        path = (ROOT / scene["config"]).resolve()
        assert path.is_relative_to((ROOT / "configs/scenes").resolve())
        config = json.loads(path.read_text())
        assert config["scene"]["id"] == scene["id"]
        assert any(row.get("dynamic", True) for row in config["entities"])
        assert any(not row.get("dynamic", True) for row in config["entities"])
        configurations.append(json.dumps(config["entities"], sort_keys=True))
    assert len(set(configurations)) == len(scenes)


def test_external_nginx_configs_block_system_namespace():
    for name in ("arena-nginx.conf", "arena-ui-preview-nginx.conf"):
        text = (ROOT / "ops" / name).read_text()
        assert "location ^~ /api/system/ { return 404; }" in text
        assert "proxy_pass http://127.0.0.1:7862" not in text
        assert "allow 127.0.0.1" not in text
        assert "location /v1/" in text


def test_supervisor_has_no_lifecycle_manager():
    text = (ROOT / "ops/arena-supervisord.conf").read_text()
    assert "[program:workspace]" not in text
    assert "workspace_manager.py" not in text


def test_workspace_status_proxy_is_read_only_and_independent_of_busagent():
    for name in ("arena-nginx.conf", "arena-ui-preview-nginx.conf"):
        text = (ROOT / "ops" / name).read_text()
        assert "location = /v1/workspace/status" in text
        assert "limit_except GET { deny all; }" in text
        assert "proxy_pass http://127.0.0.1:5599/status;" in text
    supervisor = (ROOT / "ops/arena-supervisord.conf").read_text()
    assert "[program:workspace_status]" in supervisor
    server = (ROOT / "ops/workspace_status_server.py").read_text()
    assert '("127.0.0.1", 5599)' in server
    assert "do_POST" in server
    assert "send_error(405)" in server


def test_lifecycle_worker_has_no_network_listener_and_narrow_runtime_targets():
    text = (ROOT / "ops/workspace_lifecycle_worker.py").read_text()
    assert "HTTPServer" not in text
    assert "BaseHTTPRequestHandler" not in text
    assert "output/arena" in text
    assert "reset_vision()" in text
    assert "BusAgent/backend/.local/mastra" in text
    assert 'ROOT / "output/services",' not in text
    assert 'ROOT / "output/supervisor",' not in text
    assert "intelligence.json" not in text
