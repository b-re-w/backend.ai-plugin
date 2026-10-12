import json

from labgpu.spotstatus import (
    GpuLending,
    SpotSessionState,
    parse_spot_sessions,
    parse_status,
    session_lending,
    spot_session_states,
    uuids_from_env,
)

NOW = 1_000_000.0


def status(updated=NOW - 5, gpus=()):
    return json.dumps({"updated_at": updated, "gpus": list(gpus)})


def test_parse_status_fresh_stale_and_malformed():
    text = status(gpus=[
        {"uuid": "GPU-a", "state": "LENT", "lent_job": 3, "lent_since": NOW - 600},
        {"uuid": "GPU-b", "state": "IDLE", "lent_job": None, "lent_since": None},
        {"state": "BUSY"},  # no uuid: skipped
    ])
    got = parse_status(text, NOW)
    # A lent GPU from an older monitor (no spot_share) counts as fully taken.
    assert got == {"GPU-a": GpuLending(True, NOW - 600, "LENT", spot_share=1.0), "GPU-b": GpuLending(False, None, "IDLE")}
    assert parse_status(status(updated=NOW - 61), NOW) is None  # stale: report nothing
    assert parse_status("not json", NOW) is None
    assert parse_status(json.dumps({"gpus": []}), NOW) is None


def test_session_lending_counts_only_own_gpus():
    st = {"GPU-a": GpuLending(True, NOW - 600), "GPU-b": GpuLending(False, None),
          "GPU-c": GpuLending(True, NOW - 60)}
    figures = session_lending(
        {"c1": ["GPU-a", "GPU-b"], "c2": ["GPU-c"], "c3": ["GPU-x"], "c4": []},
        own_uuids={"GPU-a", "GPU-b", "GPU-c"},
        status=st,
    )
    assert set(figures) == {"c1", "c2"}  # c3 holds another plugin's GPU, c4 none
    assert (figures["c1"].lent, figures["c1"].total, figures["c1"].since) == (1, 2, NOW - 600)
    assert (figures["c2"].lent, figures["c2"].total) == (1, 1)
    quiet = session_lending({"c1": ["GPU-b"]}, {"GPU-b"}, st)["c1"]
    assert (quiet.lent, quiet.total, quiet.since) == (0, 1, 0.0)


def test_uuids_from_env():
    assert uuids_from_env(["PATH=/bin", "LABGPU_DEVICE_UUIDS=GPU-a,GPU-b"]) == ["GPU-a", "GPU-b"]
    assert uuids_from_env(["PATH=/bin"]) == []


def test_session_lending_counts_gpus_that_could_take_a_spot():
    st = {"GPU-a": GpuLending(True, NOW - 600, "LENT"), "GPU-b": GpuLending(False, None, "LENDABLE"),
          "GPU-c": GpuLending(False, None, "BUSY"), "GPU-d": GpuLending(False, None, "")}  # "" = spot off
    f = session_lending({"c1": ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]}, {"GPU-a", "GPU-b", "GPU-c", "GPU-d"}, st)["c1"]
    assert (f.lent, f.lendable, f.total) == (1, 2, 4)


def test_uuids_from_env_reads_the_spot_list_too():
    env = ["LABGPU_SPOT_UUIDS=GPU-a,GPU-b", "LABGPU_DEVICE_UUIDS=GPU-x"]
    assert uuids_from_env(env, "LABGPU_SPOT_UUIDS") == ["GPU-a", "GPU-b"]


def test_parse_spot_sessions():
    text = json.dumps({"updated_at": NOW - 5, "gpus": [], "spot_sessions": {
        "aaaaaaaaaaaa": {"state": "parked", "gpu": "GPU-a", "since": NOW - 300},
        "bbbbbbbbbbbb": {"state": "idle", "gpu": "GPU-a", "since": None},
        "cccccccccccc": {"state": "dancing"},  # unknown state: skipped
    }})
    assert parse_spot_sessions(text, NOW) == {
        "aaaaaaaaaaaa": SpotSessionState("parked", NOW - 300),
        "bbbbbbbbbbbb": SpotSessionState("idle", 0.0),
    }
    assert parse_spot_sessions(json.dumps({"updated_at": NOW - 5, "gpus": []}), NOW) == {}  # older monitor
    assert parse_spot_sessions(json.dumps({"updated_at": NOW - 61, "spot_sessions": {}}), NOW) is None
    assert parse_spot_sessions("not json", NOW) is None


def test_spot_session_states_only_for_this_model():
    states = {"a" * 12: SpotSessionState("running", NOW - 60), "b" * 12: SpotSessionState("parked", NOW - 9)}
    got = spot_session_states(
        {"a" * 64: ["GPU-a", "GPU-b"], "b" * 64: ["GPU-x"], "c" * 64: ["GPU-a"], "d" * 64: []},
        own_uuids={"GPU-a", "GPU-b"},
        states=states,
    )
    # b runs on another model's GPUs, c is not known to the monitor yet, d is not a spot session
    assert got == {"a" * 64: SpotSessionState("running", NOW - 60)}
