from __future__ import annotations
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    # Support both dashboard/ and weld_project_template/ layouts
    template = root / "weld_project_template"
    if template.exists():
        app_path = template / "src" / "weldml" / "dashboard" / "app.py"
        config = template / "configs" / "default.yaml"
        cwd = template
    else:
        app_path = root / "src" / "weldml" / "dashboard" / "app.py"
        config = root / "configs" / "default.yaml"
        cwd = root
    cmd = ["python", "-m", "streamlit", "run", str(app_path), "--server.headless", "true", "--", "--config", str(config)]
    raise SystemExit(subprocess.call(cmd, cwd=str(cwd)))

if __name__ == "__main__":
    main()
