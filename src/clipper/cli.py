"""
CLI interface for clipper.
"""

import typer
from typing import Optional
from pathlib import Path

from .config import Config
from .db import Database
from .devices import DeviceManager

app = typer.Typer()

def get_config() -> Config:
    """Get the configuration."""
    return Config.load_config()

def get_db(config: Config) -> Database:
    """Get the database connection."""
    return Database(config)

def get_devices(config: Config) -> DeviceManager:
    """Get device manager."""
    return DeviceManager(config)

@app.command()
def doctor():
    """
    Check system for clipper requirements.
    """
    typer.echo("Checking clipper system requirements...")
    
    # Load configuration
    try:
        config = get_config()
        typer.echo("✓ Configuration loaded successfully")
    except Exception as e:
        typer.echo(f"✗ Failed to load configuration: {e}")
        raise typer.Exit(1)
    
    # Check data directory
    try:
        if not config.resolved_data_dir.exists():
            config.resolved_data_dir.mkdir(parents=True, exist_ok=True)
        typer.echo("✓ Data directory is accessible")
    except Exception as e:
        typer.echo(f"✗ Data directory inaccessible: {e}")
        raise typer.Exit(1)
    
    # Check database
    try:
        db = get_db(config)
        version = db.get_schema_version()
        typer.echo(f"✓ Database initialized (schema version {version})")
    except Exception as e:
        typer.echo(f"✗ Failed to connect to database: {e}")
        raise typer.Exit(1)
    
    # Check devices
    try:
        devices = get_devices(config)
        device_info = devices.get_all_devices()
        
        if "nvidia" in device_info and "gpus" in device_info["nvidia"]:
            gpus = device_info["nvidia"]["gpus"]
            typer.echo(f"✓ NVIDIA GPUs detected: {len(gpus)} GPU(s)")
            for gpu in gpus:
                typer.echo(f"  - GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']} MB)")
        else:
            typer.echo("⚠ No NVIDIA GPUs available (this is okay for CPU-only processing)")
            
        if "ffmpeg" in device_info and device_info["ffmpeg"].get("ffmpeg_available"):
            typer.echo("✓ FFmpeg detected")
        else:
            typer.echo("⚠ FFmpeg not found (will be required for video processing)")
            
        if "ffmpeg" in device_info and device_info["ffmpeg"].get("error"):
            typer.echo(f"  Error: {device_info['ffmpeg']['error']}")
            
    except Exception as e:
        typer.echo(f"✗ Device detection error: {e}")
        raise typer.Exit(1)
    
    # Check fonts directory
    fonts_dir = Path("./assets/fonts")
    if fonts_dir.exists():
        typer.echo("✓ Fonts directory found")
    else:
        typer.echo("⚠ Fonts directory not found")
        
    # Check Python version
    import sys
    typer.echo(f"✓ Python {sys.version}")
    
    typer.echo("System check completed successfully!")

@app.command()
def ingest(
    source_path: str = typer.Option(..., help="Path to source video file"),
    client: str = typer.Option(..., help="Client name"),
    niche: str = typer.Option(..., help="Niche/topic") 
):
    """
    Ingest a new video for processing.
    """
    typer.echo("Not implemented until M1")
    raise typer.Exit(0)

@app.command()
def run(
    job_id: str = typer.Option(..., help="Job ID to process"),
):
    """
    Run the processing pipeline for a specific job.
    """
    typer.echo("Not implemented until M1")
    raise typer.Exit(0)

@app.command()
def status(
    job_id: Optional[str] = typer.Option(None, help="Job ID to check status of")
):
    """
    Show current status of jobs.
    """
    typer.echo("Not implemented until M1")
    raise typer.Exit(0)

@app.command()
def ls():
    """
    List all jobs in the system.
    """
    typer.echo("Not implemented until M1")
    raise typer.Exit(0)

@app.command()
def approve(
    clip_id: str = typer.Option(..., help="Clip ID to approve"),
):
    """
    Approve a clip for final export.
    """
    typer.echo("Not implemented until M10")
    raise typer.Exit(0)

@app.command()
def reject(
    clip_id: str = typer.Option(..., help="Clip ID to reject"),
):
    """
    Reject a clip.
    """
    typer.echo("Not implemented until M10") 
    raise typer.Exit(0)

@app.command()
def rerender(
    clip_id: str = typer.Option(..., help="Clip ID to re-render"),
):
    """
    Re-render a clip.
    """
    typer.echo("Not implemented until M10")
    raise typer.Exit(0)

@app.command()
def export():
    """
    Export final clips.
    """
    typer.echo("Not implemented until M10")
    raise typer.Exit(0)

@app.command()
def report():
    """
    Generate system report.
    """
    typer.echo("Not implemented until M10")
    raise typer.Exit(0)

if __name__ == "__main__":
    app()