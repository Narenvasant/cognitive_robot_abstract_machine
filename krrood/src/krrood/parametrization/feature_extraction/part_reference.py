"""
Marking a field that refers to a part of another collection of the same instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.patterns.field_metadata import FieldMetadata


@dataclass
class PartReference(FieldMetadata):
    """
    Marks a field of a part, such as the first piece a contact joins, that refers to a
    part of another collection of the same instance. Feature extraction does not follow
    it, so the referring part's template never models the referred part's attributes.
    """
