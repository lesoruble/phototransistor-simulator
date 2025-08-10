import logging
import sys
from pathlib import Path

# --- Project Path Configuration ---
# Ensure the project's root directory is in the Python path.
# This allows for consistent relative imports (e.g., `from src.gui ...`)
# and helps prevent common import errors.
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# --- The Crucial Logging Configuration ---
# This MUST be done before importing any of our application modules (like gui or kernel)
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level of messages to show. INFO will show everything we need.
    format='%(asctime)s - %(name)-12s - %(levelname)-8s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Now that logging is configured, we can import our application
from src.gui import launch_gui

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.info("--- Application Starting ---")
    log.info(f"Python executable: {sys.executable}")
    log.info(f"Project root: {project_root}")
    
    launch_gui()