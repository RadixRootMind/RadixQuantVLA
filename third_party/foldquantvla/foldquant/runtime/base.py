# Copyright (c) 2026 The FoldQuant Authors.
# Licensed under the Apache License, Version 2.0; see LICENSE.

"""Common interface for compiled-engine wrappers used by model adapters."""

from __future__ import annotations

from typing import Protocol, Sequence, Tuple, runtime_checkable

import torch

TensorMeta = Tuple[str, Tuple[int, ...], torch.dtype]


@runtime_checkable
class RuntimeEngine(Protocol):
    """Structural protocol for one deserialized, ready-to-run compiled engine."""

    in_meta: Sequence[TensorMeta]
    """``(name, shape, dtype)`` per input, in binding order."""

    out_meta: Sequence[TensorMeta]
    """``(name, shape, dtype)`` per output, in binding order."""

    def set_runtime_tensor_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        """Declare the concrete shape of a dynamic input before running.

        Raises:
            RuntimeError: If *shape* is outside this engine's compiled
                optimization-profile bounds for *name*.
        """
        ...

    def forward(
        self,
        *args: torch.Tensor,
        return_list: bool = False,
        skip_checks: bool = False,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor] | list[torch.Tensor]:
        """Run one inference call, binding *args*/*kwargs* to :attr:`in_meta` by position or name."""
        ...

    def __call__(
        self,
        *args: torch.Tensor,
        return_list: bool = False,
        skip_checks: bool = False,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor] | list[torch.Tensor]: ...

    def close(self) -> None:
        """Release engine resources. Safe to call more than once."""
        ...
