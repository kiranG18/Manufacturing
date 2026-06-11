# src/features/definitions.py
#
# Central registry of all features used in the process classification pipeline.
# Each entry documents: name, unit, expected range, domain rationale, and
# which manufacturing constraints it informs.
#
# When adding a new feature:
#   1. Add an entry to FEATURE_REGISTRY below
#   2. Add the extraction logic in extractor.py
#   3. Add a validation rule in validators.py
#   4. Re-run notebooks/02_feature_engineering.ipynb to update correlation + importance plots

FEATURE_REGISTRY = {

    # --- Dimensional features ---
    "bounding_box_x": {
        "unit": "mm",
        "range": (0.5, 3000.0),
        "description": "Longest bounding box dimension along X axis.",
        "manufacturing_relevance": (
            "Machine bed / chuck size limit. Parts >500 mm in any axis are "
            "infeasible for most injection molding machines and small CNC cells."
        ),
        "process_sensitivity": ["injection_molding", "sheet_metal_stamping", "forging"],
    },
    "bounding_box_y": {
        "unit": "mm",
        "range": (0.5, 3000.0),
        "description": "Bounding box Y axis.",
        "manufacturing_relevance": "See bounding_box_x.",
        "process_sensitivity": ["injection_molding", "sheet_metal_stamping"],
    },
    "bounding_box_z": {
        "unit": "mm",
        "range": (0.5, 1000.0),
        "description": "Bounding box Z axis (depth/height).",
        "manufacturing_relevance": (
            "Deep-draw sheet metal viability. Tall parts with uniform cross-section "
            "are candidates for extrusion."
        ),
        "process_sensitivity": ["sheet_metal_deep_draw", "extrusion"],
    },

    # --- Volume / mass proxy ---
    "volume": {
        "unit": "mm³",
        "range": (1.0, 1e9),
        "description": "Estimated part volume from convex hull or mesh integration.",
        "manufacturing_relevance": (
            "Raw material cost proxy. Also separates additive (viable up to ~1e5 mm³ "
            "economically) from casting/forging (better at high volume)."
        ),
        "process_sensitivity": ["additive_fdm", "additive_slm", "casting_sand", "forging"],
    },
    "surface_area": {
        "unit": "mm²",
        "range": (1.0, 5e7),
        "description": "Total surface area of the part geometry.",
        "manufacturing_relevance": (
            "Finishing cost driver (painting, coating, anodizing). "
            "High SA/Volume ratio signals thin-walled or complex geometry."
        ),
        "process_sensitivity": ["anodizing", "powder_coating", "sheet_metal_bending"],
    },

    # --- Wall geometry ---
    "wall_thickness_min": {
        "unit": "mm",
        "range": (0.1, 50.0),
        "description": (
            "Minimum wall thickness detected across the geometry. "
            "Computed by sampling ray-cast distances across the mesh."
        ),
        "manufacturing_relevance": (
            "Hard constraint: injection molding requires min wall ~1.0–1.5 mm depending "
            "on material. Sheet metal has a minimum bend radius tied to thickness. "
            "Values < 0.8 mm typically only viable for additive or photochemical etching."
        ),
        "process_sensitivity": [
            "injection_molding", "sheet_metal_bending", "additive_slm", "die_casting"
        ],
        "critical": True,
    },
    "wall_thickness_avg": {
        "unit": "mm",
        "range": (0.5, 100.0),
        "description": "Mean wall thickness across sampled cross-sections.",
        "manufacturing_relevance": (
            "Overall material distribution. Uniform average thickness favors molding. "
            "High variance between min and avg signals complex internal geometry."
        ),
        "process_sensitivity": ["injection_molding", "die_casting", "cnc_milling"],
    },

    # --- Aspect and shape ---
    "aspect_ratio": {
        "unit": "dimensionless",
        "range": (1.0, 50.0),
        "description": "Ratio of longest to shortest bounding box dimension.",
        "manufacturing_relevance": (
            "High aspect ratio (>8) favors turning or extrusion. "
            "Low aspect ratio (<2) with volume favors casting or molding. "
            "Mid-range is the CNC/sheet metal zone."
        ),
        "process_sensitivity": ["turning", "extrusion", "cnc_milling"],
    },

    # --- Feature counts ---
    "hole_count": {
        "unit": "count",
        "range": (0, 200),
        "description": "Number of cylindrical hole features detected in the geometry.",
        "manufacturing_relevance": (
            "Drilling and tapping operations. >20 holes strongly favor CNC. "
            "Many small holes in thin sheet: punching/laser cutting."
        ),
        "process_sensitivity": ["cnc_milling", "laser_cutting", "sheet_metal_punching"],
    },
    "hole_diameter_min": {
        "unit": "mm",
        "range": (0.5, 100.0),
        "description": "Smallest hole diameter present. None → NaN, filled with part mean.",
        "manufacturing_relevance": (
            "Tight holes (<1 mm) require EDM or precision CNC. "
            "Holes in casting must meet draft constraints."
        ),
        "process_sensitivity": ["edm", "cnc_milling", "casting_die"],
    },

    # --- Binary flags (manufacturability constraints) ---
    "undercut_flag": {
        "unit": "bool",
        "range": (0, 1),
        "description": (
            "1 if any undercut geometry detected. "
            "Undercuts are regions inaccessible from a single pull direction."
        ),
        "manufacturing_relevance": (
            "Hard exclusion for standard injection molding and die casting unless "
            "side-actions are used (cost increase). Flags part for engineer review."
        ),
        "process_sensitivity": ["injection_molding", "die_casting"],
        "critical": True,
    },
    "thin_wall_flag": {
        "unit": "bool",
        "range": (0, 1),
        "description": "1 if wall_thickness_min < 1.5 mm.",
        "manufacturing_relevance": (
            "Thin walls restrict process options significantly. "
            "Often correlated with higher additive or precision sheet metal likelihood."
        ),
        "process_sensitivity": ["additive_slm", "sheet_metal_bending", "photochemical_etching"],
        "critical": True,
    },

    # --- Shape complexity scores ---
    "curvature_complexity": {
        "unit": "score 0-1",
        "range": (0.0, 1.0),
        "description": (
            "Normalized score for mean surface curvature variance. "
            "0 = flat/prismatic, 1 = highly organic / freeform."
        ),
        "manufacturing_relevance": (
            "High curvature (>0.6) makes subtractive methods expensive (5-axis CNC) "
            "or infeasible. Favors additive manufacturing or investment casting."
        ),
        "process_sensitivity": ["additive_fdm", "additive_slm", "investment_casting", "cnc_5axis"],
    },
    "symmetry_score": {
        "unit": "score 0-1",
        "range": (0.0, 1.0),
        "description": (
            "Degree of rotational/bilateral symmetry. "
            "Computed by comparing mirrored mesh overlap ratio."
        ),
        "manufacturing_relevance": (
            "High symmetry (>0.8) indicates turning or forging viability. "
            "Asymmetric parts typically require milling or additive."
        ),
        "process_sensitivity": ["turning", "forging", "casting_sand"],
    },
}

# Ordered list for training data column alignment
FEATURE_COLUMNS = list(FEATURE_REGISTRY.keys())

# Features that are hard constraints: if violated, certain processes must be excluded
# regardless of model probability output
HARD_CONSTRAINT_FEATURES = [
    "undercut_flag",
    "thin_wall_flag",
    "wall_thickness_min",
    "bounding_box_x",
    "bounding_box_y",
]
