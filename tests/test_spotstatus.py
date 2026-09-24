import json

from labgpu.spotstatus import GpuLending, parse_status, session_lending, uuids_from_env

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
    assert got == {"GPU-a": GpuLending(True, NOW - 600), "GPU-b": GpuLending(False, None)}
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
