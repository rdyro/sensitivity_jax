import sys
from pathlib import Path

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if path not in sys.path:
        sys.path.append(str(path))
    
from .tour import main

def test_tour():
    main()

if __name__ == "__main__":
    test_tour()