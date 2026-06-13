"""AutoGen adapter.

AutoGen (AG2) passes a list of message dicts between agents. Each dict already
looks like ``{"role": ..., "content": ...}``, which is the format the TEA core
uses, so the adapter is thin.

Two entry points.

1. A transform function for a list of messages::

       from tea.integrations.autogen_hook import optimize_autogen_messages
       optimized, report = optimize_autogen_messages(messages, model_name="gpt-4o")

2. A ``transform_messages`` hook compatible with AutoGen's capability system,
   so the optimisation runs automatically before each model call::

       from tea.integrations.autogen_hook import TEAMessageTransform
       from autogen.agentchat.contrib.capabilities import transform_messages

       handler = transform_messages.TransformMessages(
           transforms=[TEAMessageTransform(model_name="gpt-4o")]
       )
       handler.add_to_agent(assistant)

The transform leaves the most recent message untouched so the live turn is
never altered, and only compresses earlier context.
"""

from __future__ import annotations

from typing import Any, Optional

from ..optimizer import Compressor, OptimizeResult, optimize_messages


def optimize_autogen_messages(
    messages: list[dict],
    *,
    model_name: str = "gpt-4o",
    enable: Optional[set[str]] = None,
    compressor: Optional[Compressor] = None,
    **opt_kwargs: Any,
) -> tuple[list[dict], OptimizeResult]:
    """Optimise a list of AutoGen message dicts. Returns (messages, report)."""
    result = optimize_messages(
        messages, model=model_name, enable=enable, compressor=compressor, **opt_kwargs
    )
    return result.optimized, result


class TEAMessageTransform:
    """An AutoGen-compatible message transform.

    Implements ``apply_transform(messages) -> messages`` and
    ``get_logs(pre, post) -> (str, bool)``, which is the interface AutoGen's
    TransformMessages capability expects. AutoGen itself is not imported, so
    this class is safe to construct even without AutoGen installed; it only
    needs the framework when added to an agent.
    """

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o",
        enable: Optional[set[str]] = None,
        compressor: Optional[Compressor] = None,
        protect_last: int = 1,
        **opt_kwargs: Any,
    ):
        self.model_name = model_name
        self.enable = enable
        self.compressor = compressor
        self.protect_last = max(0, protect_last)
        self.opt_kwargs = opt_kwargs
        self._last_saved = 0

    def apply_transform(self, messages: list[dict]) -> list[dict]:
        if not messages:
            return messages
        # Protect the most recent N messages from any change.
        cut = len(messages) - self.protect_last if self.protect_last else len(messages)
        head, tail = messages[:cut], messages[cut:]
        if not head:
            self._last_saved = 0
            return messages
        result = optimize_messages(
            head, model=self.model_name, enable=self.enable,
            compressor=self.compressor, **self.opt_kwargs
        )
        self._last_saved = result.tokens_saved
        return list(result.optimized) + list(tail)

    def get_logs(self, pre_transform_messages: list, post_transform_messages: list):
        if self._last_saved > 0:
            return (f"TEA saved {self._last_saved:,} tokens.", True)
        return ("", False)
