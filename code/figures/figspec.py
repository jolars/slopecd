from pathlib import Path
from pyprojroot import here

FULL_WIDTH = 6.75
HALF_WIDTH = 3.25

def fig_path(x):
    code_dir = Path(here())
    root_dir = code_dir.parent
    fig_dir = root_dir / "figures"

    return fig_dir / x
    
    
