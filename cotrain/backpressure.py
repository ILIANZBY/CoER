"""Small, dependency-free helpers for co-training rollout backpressure."""


def both_role_queues_full(
    attacker_depth: int,
    defender_depth: int,
    max_queue_size: int,
) -> bool:
    """Return whether generation should pause for two independently drained roles."""

    if max_queue_size <= 0:
        return False
    return (
        int(attacker_depth) >= int(max_queue_size)
        and int(defender_depth) >= int(max_queue_size)
    )
