"""Framework adapters for TEA.

Each adapter wires the TEA optimiser into a specific framework or SDK. None are
imported here, so importing ``tea`` never requires any framework to be present.
Import the adapter you need directly, for example::

    from tea.integrations.openai_wrap import optimize_openai_kwargs
    from tea.integrations.langchain_cb import TEAOptimizer

Available adapters:

- ``openai_wrap``    : optimise the ``messages`` you pass to the OpenAI SDK.
- ``anthropic_wrap`` : optimise ``system`` + ``messages`` for the Anthropic SDK.
- ``langchain_cb``   : a LangChain Runnable middleware and prompt helper.
- ``crewai_hook``    : optimise task descriptions and agent backstories in CrewAI.
- ``autogen_hook``   : a message transform for AutoGen conversations.
"""
