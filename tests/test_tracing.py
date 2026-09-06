from llm_serving_lab.tracing import TraceRecorder


def test_trace_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path)
    recorder.emit("scheduled", step=2, request_id="r1", tokens=[1, 2])
    recorder.write()
    events = TraceRecorder.read(path)
    assert len(events) == 1
    assert events[0].kind == "scheduled"
    assert events[0].request_id == "r1"
    assert events[0].fields == {"tokens": [1, 2]}
