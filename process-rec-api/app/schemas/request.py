# app/schemas/request.py
#
# Pydantic request schemas for the process recommendation API.
# These define exactly what the Quanta/Optima frontend must send
# and what gets validated before any model inference runs.
#
# Validation errors return 422 with a structured error body — not a 500.
# This was an intentional design decision: bad input from the CAD parser
# should surface as a validation error, not a model error.

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum


class SupportedFormat(str, Enum):
    STEP = "STEP"
    STL = "STL"
    IGES = "IGES"


class SurfaceFinishPreset(str, Enum):
    AS_MACHINED = "as_machined"      # Ra 3.2
    FINE = "fine"                     # Ra 1.6
    VERY_FINE = "very_fine"           # Ra 0.8
    MIRROR = "mirror"                 # Ra 0.1


class ToleranceBand(str, Enum):
    IT4 = "IT4"
    IT5 = "IT5"
    IT6 = "IT6"
    IT7 = "IT7"
    IT8 = "IT8"    # Default: general precision machining
    IT9 = "IT9"
    IT10 = "IT10"
    IT11 = "IT11"
    IT12 = "IT12"


class BoundingBox(BaseModel):
    x: float = Field(..., gt=0, le=3000, description="X dimension in mm")
    y: float = Field(..., gt=0, le=3000, description="Y dimension in mm")
    z: float = Field(..., gt=0, le=1000, description="Z dimension in mm")


class HoleFeature(BaseModel):
    diameter: float = Field(..., gt=0, le=500, description="Hole diameter in mm")
    depth: float = Field(..., gt=0, le=2000, description="Hole depth in mm")


class PartGeometry(BaseModel):
    """
    Geometry metadata extracted from a CAD part upload.
    Produced by the upstream CAD parsing service before this API is called.
    """
    bounding_box: BoundingBox
    volume_mm3: float = Field(..., gt=0, description="Part volume in mm³")
    surface_area_mm2: float = Field(..., gt=0, description="Total surface area in mm²")
    wall_thicknesses: List[float] = Field(
        default_factory=list,
        description="Sampled wall thickness measurements in mm"
    )
    holes: List[HoleFeature] = Field(
        default_factory=list,
        description="Detected hole features"
    )
    has_undercuts: bool = Field(
        False,
        description="True if geometry contains undercut features"
    )
    curvature_samples: List[float] = Field(
        default_factory=list,
        description="Per-face normalized curvature values (0–1)"
    )
    symmetry_axes: Dict[str, float] = Field(
        default_factory=dict,
        description="Symmetry score per axis: {'x': 0-1, 'y': 0-1, 'z': 0-1}"
    )
    source_format: SupportedFormat = SupportedFormat.STEP

    @validator("wall_thicknesses")
    def wall_thicknesses_must_be_positive(cls, v):
        if any(t <= 0 for t in v):
            raise ValueError("All wall thickness values must be positive")
        return v

    @validator("curvature_samples")
    def curvature_must_be_normalized(cls, v):
        if any(not (0.0 <= c <= 1.0) for c in v):
            raise ValueError("Curvature samples must be in range [0, 1]")
        return v


class PartRequirements(BaseModel):
    """
    User-specified manufacturing requirements for the part.
    These come from the form the user fills in Quanta or Optima.
    """
    material_code: str = Field(
        ...,
        description="Material identifier, e.g. 'AL6061', 'SS316L', 'PEEK'"
    )
    quantity: int = Field(
        ...,
        gt=0,
        le=1_000_000,
        description="Number of parts to manufacture"
    )
    surface_finish_spec: Optional[float] = Field(
        None,
        gt=0,
        le=25,
        description="Surface finish Ra value in microns. If null, preset is used."
    )
    surface_finish_preset: Optional[SurfaceFinishPreset] = Field(
        SurfaceFinishPreset.AS_MACHINED,
        description="Named surface finish preset (used if surface_finish_spec is null)"
    )
    tolerance_band: ToleranceBand = Field(
        ToleranceBand.IT8,
        description="ISO tolerance grade. Tighter = more expensive."
    )
    max_lead_time_days: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum acceptable lead time. Used to filter slow processes."
    )
    batch_size: Optional[int] = Field(
        None,
        gt=0,
        description="Production batch size if different from quantity (for amortization)"
    )


class RecommendRequest(BaseModel):
    """
    Full inference request payload.
    """
    part_id: str = Field(..., description="Unique identifier for this part")
    geometry: PartGeometry
    requirements: PartRequirements
    top_k: int = Field(3, ge=1, le=5, description="Number of process options to return")
    include_debug: bool = Field(
        False,
        description="If true, include SHAP feature importance in response"
    )

    def to_feature_dict(self) -> Dict[str, float]:
        """Flattens the nested Pydantic geometry features into a flat dict for ML model inference."""
        geom = self.geometry
        reqs = self.requirements
        
        # Dimensions
        x = geom.bounding_box.x
        y = geom.bounding_box.y
        z = geom.bounding_box.z
        
        # Wall thicknesses
        wt = geom.wall_thicknesses
        wt_min = min(wt) if wt else 2.0
        wt_avg = sum(wt) / len(wt) if wt else 3.0
        
        # Holes
        holes = geom.holes
        hole_count = len(holes)
        hole_min = min(h.diameter for h in holes) if holes else 0.0

        # Curvature
        curv = geom.curvature_samples
        curv_avg = sum(curv) / len(curv) if curv else 0.0

        # Symmetry
        sym_axes = geom.symmetry_axes
        sym_score = sum(sym_axes.values()) / len(sym_axes) if sym_axes else 0.5

        # Thin wall flag derivation
        thin_wall = 1 if wt_min < 1.5 else 0

        # Aspect ratio
        dims = [x, y, z]
        aspect = max(dims) / min(dims) if min(dims) > 0 else 1.0

        return {
            "bounding_box_x": float(x),
            "bounding_box_y": float(y),
            "bounding_box_z": float(z),
            "volume": float(geom.volume_mm3),
            "surface_area": float(geom.surface_area_mm2),
            "wall_thickness_min": float(wt_min),
            "wall_thickness_avg": float(wt_avg),
            "aspect_ratio": float(aspect),
            "hole_count": float(hole_count),
            "hole_diameter_min": float(hole_min),
            "undercut_flag": float(1 if geom.has_undercuts else 0),
            "thin_wall_flag": float(thin_wall),
            "curvature_complexity": float(curv_avg),
            "symmetry_score": float(sym_score),
            # Requirements fields that cost models consume
            "material_code": reqs.material_code,
            "quantity": float(reqs.quantity),
            "surface_finish_ra": float(reqs.surface_finish_spec) if reqs.surface_finish_spec is not None else 3.2,
            "tolerance_band": reqs.tolerance_band.value if hasattr(reqs.tolerance_band, "value") else str(reqs.tolerance_band),
            "batch_size": float(reqs.batch_size) if reqs.batch_size is not None else float(reqs.quantity)
        }

    class Config:
        schema_extra = {
            "example": {
                "part_id": "PART-20241114-001",
                "geometry": {
                    "bounding_box": {"x": 45.2, "y": 30.1, "z": 12.8},
                    "volume_mm3": 8420.0,
                    "surface_area_mm2": 5210.0,
                    "wall_thicknesses": [2.1, 2.4, 1.9, 3.2, 2.0],
                    "holes": [{"diameter": 3.0, "depth": 12.0}],
                    "has_undercuts": False,
                    "curvature_samples": [0.02, 0.01, 0.04],
                    "symmetry_axes": {"x": 0.2, "y": 0.7, "z": 0.1},
                    "source_format": "STEP"
                },
                "requirements": {
                    "material_code": "AL6061",
                    "quantity": 200,
                    "surface_finish_preset": "fine",
                    "tolerance_band": "IT8"
                },
                "top_k": 3
            }
        }
