import typer
from pathlib import Path
import logging
import sys
from src.ingestion.pipeline import run_pipeline
from src.common.config import settings
logging.basicConfig(
    level=logging.DEBUG,  # or INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
logger.info("✅ Logging configured")
app = typer.Typer()

@app.command()
def all(input_dir: str = settings.DATA_DIR):
    run_pipeline(Path(input_dir))
    print("ended")

if __name__ == "__main__":
    app()
