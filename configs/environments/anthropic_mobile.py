"""Anthropic Mobile environment configuration."""

# Anthropic Mobile environment configuration
environment = dict(
    base_dir="workdir/anthropic_mobile_agent",
    device_id=None,  # Use first connected device
    fps=2,
    bitrate=50000000,
    chunk_duration=60
)