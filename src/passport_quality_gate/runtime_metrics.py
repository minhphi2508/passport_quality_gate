"""Portable best-effort process memory metric for opt-in research tools."""


def memory_mib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        # Sandboxed PID namespaces can make /proc/<getpid()> unavailable even
        # though getrusage still works. Unix fallback is the peak RSS.
        try:
            import resource
            import sys
            value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return value/(1024**2 if sys.platform=='darwin' else 1024)
        except (ImportError, OSError):
            return None
