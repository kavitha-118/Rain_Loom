"""Data ingestion and feature engineering for the Monsoon-Textile Volatility system."""

try:
    from .pipeline import (
        Config,
        DataPipeline,
        IMDDataLoader,
        MCXCottonLoader,
        MacroDataLoader,
        NSEDataLoader,
        main,
    )

    __all__ = [
        "Config",
        "DataPipeline",
        "IMDDataLoader",
        "MCXCottonLoader",
        "MacroDataLoader",
        "NSEDataLoader",
        "main",
    ]
except ImportError:
    # pipeline.py requires loguru, imdlib, yaml — gracefully skip if missing
    __all__ = []
