"""User entrypoint. All cloud operations are read-only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from experiment_workbench import main
if __name__=='__main__':raise SystemExit(main())
