"""Portable process memory metric for opt-in research tools."""


def memory_mib():
    try:
        # /proc/self follows the calling process even in nested PID namespaces.
        # psutil.Process(getpid()) can silently refer to an unrelated host PID.
        from pathlib import Path
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return float(line.split()[1])/1024
    except OSError:
        pass
    try:
        import resource
        import sys
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value/(1024**2 if sys.platform=='darwin' else 1024)
    except (ImportError, OSError):
        try:
            import psutil
            return psutil.Process().memory_info().rss/1024**2
        except Exception:
            return None
