from llm_serving_lab.mini_serving import (
    BlockManager,
    DeterministicModelRunner,
    Engine,
    Request,
    Scheduler,
    SchedulerConfig,
)


def make_engine(*, budget: int, chunked_prefill: bool = True) -> Engine:
    manager = BlockManager(block_size=4, num_blocks=32)
    scheduler = Scheduler(
        manager,
        SchedulerConfig(
            max_num_batched_tokens=budget,
            max_num_seqs=4,
            chunked_prefill=chunked_prefill,
        ),
    )
    return Engine(scheduler, DeterministicModelRunner())


def test_scheduler_respects_token_budget_and_chunks_prefill() -> None:
    engine = make_engine(budget=4)
    request = Request("long", list(range(1, 10)), 2)
    engine.add_request(request)
    first = engine.step()
    second = engine.step()
    third = engine.step()
    assert first.token_budget_used == 4
    assert second.token_budget_used == 4
    assert third.token_budget_used == 1
    assert request.num_computed_tokens == 9
    scheduled = [
        event for event in engine.recorder.events if event.kind == "request_scheduled"
    ]
    assert [event.fields["phase"] for event in scheduled] == [
        "prefill",
        "prefill",
        "prefill",
    ]


def test_engine_finishes_requests_and_releases_blocks() -> None:
    engine = make_engine(budget=6)
    first = Request("a", [1, 2, 3], 3)
    second = Request("b", [1, 7, 8, 9, 10], 2)
    engine.add_request(first)
    engine.add_request(second)
    outputs = engine.run()
    assert outputs
    assert first.is_finished and second.is_finished
    assert len(first.output_token_ids) == 3
    assert len(second.output_token_ids) == 2
    assert engine.scheduler.block_manager.num_free_blocks == 32


def test_chunking_does_not_change_deterministic_model_output() -> None:
    chunked = make_engine(budget=3, chunked_prefill=True)
    whole = make_engine(budget=32, chunked_prefill=False)
    left = Request("same", list(range(1, 12)), 4)
    right = Request("same", list(range(1, 12)), 4)
    chunked.add_request(left)
    whole.add_request(right)
    chunked.run()
    whole.run()
    assert left.output_token_ids == right.output_token_ids
