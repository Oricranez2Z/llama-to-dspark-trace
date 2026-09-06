from collections import deque
from math import ceil

from .request import Request


class BlockManager:
    """Allocate fixed-size physical blocks to logical request token ranges."""

    def __init__(self, *, block_size: int, num_blocks: int):
        if block_size <= 0 or num_blocks <= 0:
            raise ValueError("block_size and num_blocks must be positive")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free_blocks = deque(range(num_blocks))
        self._owners: dict[int, str] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def owners(self) -> dict[int, str]:
        return self._owners.copy()

    def blocks_required(self, num_tokens: int) -> int:
        return 0 if num_tokens == 0 else ceil(num_tokens / self.block_size)

    def can_allocate(self, request: Request, num_new_tokens: int) -> bool:
        total_tokens = request.num_computed_tokens + num_new_tokens
        required = self.blocks_required(total_tokens)
        return required - len(request.block_ids) <= self.num_free_blocks

    def allocate(self, request: Request, num_new_tokens: int) -> list[int] | None:
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens cannot be negative")
        total_tokens = request.num_computed_tokens + num_new_tokens
        required = self.blocks_required(total_tokens)
        missing = required - len(request.block_ids)
        if missing > self.num_free_blocks:
            return None
        new_blocks: list[int] = []
        for _ in range(missing):
            block_id = self._free_blocks.popleft()
            self._owners[block_id] = request.request_id
            request.block_ids.append(block_id)
            new_blocks.append(block_id)
        return new_blocks

    def slots_for_range(
        self,
        request: Request,
        start_token: int,
        num_tokens: int,
    ) -> list[int]:
        slots: list[int] = []
        for token_index in range(start_token, start_token + num_tokens):
            logical_block = token_index // self.block_size
            offset = token_index % self.block_size
            if logical_block >= len(request.block_ids):
                raise RuntimeError("request has no physical block for token")
            physical_block = request.block_ids[logical_block]
            slots.append(physical_block * self.block_size + offset)
        return slots

    def free(self, request: Request) -> list[int]:
        released = request.block_ids.copy()
        for block_id in released:
            owner = self._owners.pop(block_id, None)
            if owner != request.request_id:
                raise RuntimeError("block ownership invariant violated")
            self._free_blocks.append(block_id)
        request.block_ids.clear()
        return released

    def validate(self, requests: list[Request]) -> None:
        request_blocks = {
            block_id: request.request_id
            for request in requests
            for block_id in request.block_ids
        }
        if request_blocks != self._owners:
            raise RuntimeError("request block tables and ownership map disagree")
        if len(self._owners) + len(self._free_blocks) != self.num_blocks:
            raise RuntimeError("allocated and free block counts do not add up")
        if set(self._owners).intersection(self._free_blocks):
            raise RuntimeError("a block cannot be both allocated and free")
