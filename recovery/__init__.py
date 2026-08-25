"""
Day-12 end-to-end recovery orchestration.

    failure_event -> classification -> policy-v4 -> compliance gate
    -> payment action -> LLM communication (if allowed) -> audit trail

See recovery/orchestrator.py for the single entry point
(`orchestrate_recovery`) and recovery/schemas.py for the structured
`RecoveryExecutionResult` every call returns.
"""
