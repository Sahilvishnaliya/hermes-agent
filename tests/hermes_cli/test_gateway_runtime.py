"""Runtime discovery must distinguish absence from an unsafe/live owner."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest


@asynccontextmanager
async def control_peer(home: Path, payload: dict):
    from gateway.control_socket import GatewayControlServer

    home.mkdir(mode=0o700)
    server = GatewayControlServer(home, verb_handlers={"identify": lambda: payload})
    assert await server.start()
    pointer = home / "gateway.sock.path"
    if pointer.exists():
        pointer.chmod(0o600)
    try:
        yield server
    finally:
        await server.stop()


@pytest.mark.platforms("linux")
def test_live_discovery_does_not_treat_an_unusable_owner_as_absent(tmp_path):
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    async def probe():
        home = tmp_path / "profile"
        identity = str(home.resolve())
        payload = {
            "protocol": 1,
            "runtime_protocol": 1,
            "instance_id": "fixture-instance",
            "authority_epoch": 1,
            "hermes_home": identity,
            "state": "ready",
            "api_origin": "http://127.0.0.1:41817",
            "supervisor": "none",
            "served_profiles": [{"profile_id": identity, "home": identity}],
            "capabilities": ["session-authority-v1"],
        }
        absent = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
        assert absent.state == "absent"
        async with control_peer(home, payload):
            ready = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
            assert ready.state == "ready"
            assert ready.endpoint.profile_id == identity
            assert ready.endpoint.instance_id == "fixture-instance"
            assert ready.endpoint.api_origin == "http://127.0.0.1:41817"
            for key, value, expected in [
                ("runtime_protocol", 99, "incompatible"),
                ("runtime_protocol", True, "incompatible"),
                ("served_profiles", [], "inaccessible"),
                ("state", "draining", "draining"),
                ("api_origin", "http://secret:credential@127.0.0.1:41817", "inaccessible"),
                ("api_origin", "http://example.com:41817", "inaccessible"),
                ("authority_epoch", True, "inaccessible"),
                ("capabilities", [], "incompatible"),
            ]:
                old = payload[key]
                payload[key] = value
                observed = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
                assert observed.state == expected, (key, observed)
                assert "credential" not in repr(observed)
                payload[key] = old
        # A forged pointer is not permission to connect somewhere else or spawn.
        (home / "gateway.sock.path").write_text(str(tmp_path / "foreign.sock"))
        (home / "gateway.sock.path").chmod(0o666)
        refused = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1)
        assert refused.state == "inaccessible"

    asyncio.run(probe())


@pytest.mark.platforms("linux")
def test_fifo_control_pointer_is_rejected_without_a_writer(tmp_path):
    import os
    import subprocess
    import sys

    home = tmp_path / 'fifo-profile'
    home.mkdir(mode=0o700)
    os.mkfifo(home / 'gateway.sock.path', 0o600)
    code = (
        'import sys; from pathlib import Path; '
        'from hermes_cli.gateway_runtime import discover_gateway_endpoint; '
        'r = discover_gateway_endpoint(Path(sys.argv[1]), timeout=0.1); '
        'print(r.state, r.reason_code)'
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', code, str(home)],
            cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=3,
        )
    except subprocess.TimeoutExpired:
        pytest.fail('discovery blocked on a FIFO instead of rejecting its file type')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'inaccessible unsafe_control_pointer'


def test_start_decisions_do_not_turn_uncertainty_into_another_owner():
    from gateway.runtime_contract import RuntimeObservation, next_start_action

    cases = [
        (RuntimeObservation("ready"), "attach"),
        (RuntimeObservation("starting"), "wait-owner"),
        (RuntimeObservation("draining"), "wait-owner"),
        (RuntimeObservation("incompatible"), "reject-version"),
        (RuntimeObservation("inaccessible"), "reject-access"),
        (RuntimeObservation("conflict"), "reject-conflict"),
        (RuntimeObservation("absent"), "spawn-unmanaged"),
        (RuntimeObservation("absent", installed_service=True), "start-service"),
        (RuntimeObservation("absent", service_may_start=True), "wait-service"),
        (RuntimeObservation("ready", update_paused=True), "wait-update"),
    ]
    for observation, expected in cases:
        assert next_start_action(observation) == expected
    with pytest.raises(ValueError, match="unknown runtime state"):
        next_start_action(RuntimeObservation("invalid"))


def test_discovery_reaches_sibling_host_gateway(tmp_path, monkeypatch):
    """A named profile served by a sibling-host gateway must attach, not wait out the deadline.

    Regression for PR #106742 review: discovery only asked the requested home and the
    default root, so siblings of a ``profiles/<host>`` owner reported "starting" until
    the deadline. The host record names the live owner whatever home launched it.
    Socket-free: fakes ``query_identify`` so it runs on every platform.
    """
    import os
    from pathlib import Path

    from gateway.host_attach import HostGateway
    import gateway.host_attach as host_attach
    import hermes_cli.gateway_runtime_discovery as discovery

    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    host = tmp_path / "host-profile"
    client = tmp_path / "client-profile"
    client.mkdir(mode=0o700)
    host_id = str(host.resolve())
    client_id = str(client.resolve())
    payload = {
        "protocol": 1,
        "runtime_protocol": 1,
        "instance_id": "sibling-host-instance",
        "authority_epoch": 3,
        "hermes_home": host_id,
        "pid": os.getpid(),
        "state": "ready",
        "api_origin": "http://127.0.0.1:41819",
        "supervisor": "none",
        "served_profiles": [
            {"profile_id": host_id, "home": host_id},
            {"profile_id": client_id, "home": client_id},
        ],
        "capabilities": ["session-authority-v1"],
    }

    def fake_identify(home, timeout=2.0):
        if os.path.normcase(str(Path(home).resolve())) == os.path.normcase(host_id):
            return dict(payload)
        raise FileNotFoundError(str(home))

    monkeypatch.setattr(discovery, "query_identify", fake_identify)
    monkeypatch.setattr(
        host_attach,
        "host_gateway",
        lambda wait_for_channel=0.0: HostGateway(
            pid=os.getpid(), home=Path(host), profiles=("host-profile", "client-profile"),
        ),
    )
    observed = discover_gateway_endpoint(client, timeout=2)
    assert observed.state == "ready", observed
    assert observed.endpoint.profile_id == client_id
    assert observed.endpoint.control_home == str(host)
    assert observed.endpoint.instance_id == "sibling-host-instance"


def test_discovery_ignores_host_gateway_that_does_not_serve_us(tmp_path, monkeypatch):
    """A host gateway serving someone else must not mask an absent profile."""
    import os
    from pathlib import Path

    from gateway.host_attach import HostGateway
    import gateway.host_attach as host_attach
    import hermes_cli.gateway_runtime_discovery as discovery

    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    host = tmp_path / "host-profile"
    client = tmp_path / "client-profile"
    client.mkdir(mode=0o700)
    host_id = str(host.resolve())
    payload = {
        "protocol": 1,
        "runtime_protocol": 1,
        "instance_id": "other-host-instance",
        "authority_epoch": 3,
        "hermes_home": host_id,
        "pid": os.getpid(),
        "state": "ready",
        "api_origin": "http://127.0.0.1:41819",
        "supervisor": "none",
        "served_profiles": [{"profile_id": host_id, "home": host_id}],
        "capabilities": ["session-authority-v1"],
    }

    def fake_identify(home, timeout=2.0):
        if os.path.normcase(str(Path(home).resolve())) == os.path.normcase(host_id):
            return dict(payload)
        raise FileNotFoundError(str(home))

    monkeypatch.setattr(discovery, "query_identify", fake_identify)
    monkeypatch.setattr(
        host_attach,
        "host_gateway",
        lambda wait_for_channel=0.0: HostGateway(
            pid=os.getpid(), home=Path(host), profiles=("host-profile",),
        ),
    )
    observed = discover_gateway_endpoint(client, timeout=2)
    assert observed.state == "absent", observed
