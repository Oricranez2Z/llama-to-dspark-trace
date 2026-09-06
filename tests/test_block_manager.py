from llm_serving_lab.mini_serving import BlockManager, Request


def test_block_allocation_slot_mapping_and_release() -> None:
    manager = BlockManager(block_size=4, num_blocks=4)
    request = Request("r1", [1, 2, 3, 4, 5], 2)
    allocated = manager.allocate(request, 5)
    assert allocated == [0, 1]
    assert manager.slots_for_range(request, 0, 5) == [0, 1, 2, 3, 4]
    assert manager.num_free_blocks == 2
    released = manager.free(request)
    assert released == [0, 1]
    assert manager.num_free_blocks == 4
    manager.validate([request])


def test_allocation_fails_without_mutating_request() -> None:
    manager = BlockManager(block_size=2, num_blocks=1)
    request = Request("r1", [1, 2, 3], 1)
    assert manager.allocate(request, 3) is None
    assert request.block_ids == []
    assert manager.num_free_blocks == 1
