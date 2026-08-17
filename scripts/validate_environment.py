from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = ["numpy", "scipy", "networkx", "yaml", "pandas", "PIL", "cv2", "typer"]


def main() -> int:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    missing: list[str] = []
    for module_name in REQUIRED:
        try:
            module = importlib.import_module(module_name)
            print(f"{module_name}: {getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            missing.append(f"{module_name}: {exc}")
    if missing:
        print("Missing dependencies:")
        print("\n".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
