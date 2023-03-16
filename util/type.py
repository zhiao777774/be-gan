def str2int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except:
        return default


def str2float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except:
        return default
