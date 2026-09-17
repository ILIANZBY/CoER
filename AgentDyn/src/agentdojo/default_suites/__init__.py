"""Backward-compatible shim: import_default_suites returns all suites across versions."""


def import_default_suites(benchmark_version: str = "v1.1") -> dict:
    """Return suites dict for the given benchmark version.

    Merges all versions into a single flat dict when no version is specified,
    for backward compatibility with code that does:
        suites = import_default_suites()
        suite = suites[suite_name]
    """
    from agentdojo.task_suite.load_suites import get_suites
    return get_suites(benchmark_version)


def get_suites(benchmark_version: str = "v1.1") -> dict:
    from agentdojo.task_suite.load_suites import get_suites as _get_suites
    return _get_suites(benchmark_version)


def _get_SUITES():
    from agentdojo.task_suite.load_suites import _SUITES
    return _SUITES
