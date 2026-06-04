"""Utility functions for the dataset importer workflow."""

from mixtrain.types import ALL_MIX_TYPES, MixType


def column_types_for_save(
    column_types: dict[str, str] | str,
) -> dict[str, type[MixType]]:
    """Convert JSON-safe column type names for ``Dataset.save()``."""
    if not isinstance(column_types, dict):
        raise ValueError("column_types must be a dict of overrides")
    sdk_types = {typ._type: typ for typ in ALL_MIX_TYPES}
    invalid_types = {
        col: typ for col, typ in column_types.items() if typ not in sdk_types
    }
    if invalid_types:
        raise ValueError(
            f"Invalid column types: {invalid_types}. Must be one of {sorted(sdk_types)}"
        )

    return {col: sdk_types[typ] for col, typ in column_types.items()}
