import pytest

from utils import ServerPreset, parallel_function_calls


server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 3


def complete(prompt, headers, n_predict=1):
    return server.make_request(
        "POST",
        "/completion",
        data={"prompt": prompt, "n_predict": n_predict},
        headers=headers,
    )


def test_claude_affinity_precedence_and_typed_keys():
    server.start()

    main = complete("main request", {"x-claude-code-session-id": "shared-id"})
    sub = complete("sub-agent request", {
        "x-claude-code-session-id": "shared-id",
        "x-claude-code-agent-id": "shared-id",
    })
    other_sub = complete("other sub-agent request", {
        "x-claude-code-session-id": "shared-id",
        "x-claude-code-agent-id": "other-agent",
    })

    assert len({main.body["id_slot"], sub.body["id_slot"], other_sub.body["id_slot"]}) == 3

    main_again = complete("unrelated main follow-up", {"x-claude-code-session-id": "shared-id"})
    sub_again = complete("unrelated sub-agent follow-up", {
        "x-claude-code-session-id": "shared-id",
        "x-claude-code-agent-id": "shared-id",
    })

    assert main_again.body["id_slot"] == main.body["id_slot"]
    assert sub_again.body["id_slot"] == sub.body["id_slot"]


def test_same_session_concurrent_requests_are_serialized():
    server.n_slots = 2
    server.start()

    headers = {"x-claude-code-session-id": "concurrent-session"}
    results = parallel_function_calls([
        (complete, ("first concurrent request", headers, 32)),
        (complete, ("second concurrent request", headers, 32)),
    ])

    assert all(result.status_code == 200 for result in results)
    assert results[0].body["id_slot"] == results[1].body["id_slot"]


def test_invalid_affinity_header_falls_back_without_sticky_mapping():
    server.n_slots = 2
    server.start()

    invalid = {"x-claude-code-session-id": "contains whitespace"}
    first = complete("first invalid request", invalid)
    second = complete("completely different invalid request", invalid)

    assert first.status_code == 200
    assert second.status_code == 200


def test_slot_erase_releases_affinity_mapping():
    server.slot_save_path = "./tmp"
    server.start()

    assigned = [
        complete(f"request {idx}", {"x-claude-code-session-id": f"session-{idx}"})
        for idx in range(3)
    ]
    assert len({res.body["id_slot"] for res in assigned}) == 3

    erased_slot = assigned[1].body["id_slot"]
    erased = server.make_request("POST", f"/slots/{erased_slot}?action=erase")
    assert erased.status_code == 200

    replacement = complete(
        "replacement request",
        {"x-claude-code-session-id": "replacement-session"},
    )
    assert replacement.status_code == 200
    assert replacement.body["id_slot"] == erased_slot


def test_affinity_metrics_by_key_type():
    server.server_metrics = True
    server.start()

    session_headers = {"x-claude-code-session-id": "metrics-session"}
    agent_headers = {"x-claude-code-agent-id": "metrics-agent"}
    complete("session assignment", session_headers)
    complete("session hit", session_headers)
    complete("agent assignment", agent_headers)
    complete("agent hit", agent_headers)

    response = server.make_request("GET", "/metrics")
    assert response.status_code == 200
    assert 'llamacpp:affinity_assigned_total{key_type="session"} 1' in response.body
    assert 'llamacpp:affinity_hit_total{key_type="session"} 1' in response.body
    assert 'llamacpp:affinity_assigned_total{key_type="agent"} 1' in response.body
    assert 'llamacpp:affinity_hit_total{key_type="agent"} 1' in response.body
    assert "metrics-session" not in response.body
    assert "metrics-agent" not in response.body
