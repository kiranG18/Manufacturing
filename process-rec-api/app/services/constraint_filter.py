# app/services/constraint_filter.py
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

ALL_PROCESSES = {
    "cnc_milling",
    "cnc_turning",
    "sheet_metal_bending",
    "sheet_metal_stamping",
    "injection_molding",
    "die_casting",
    "sand_casting",
    "investment_casting",
    "additive_fdm",
    "additive_slm",
    "laser_cutting",
    "forging",
    "extrusion",
    "5axis_cnc",
    "turning_milling",
    "edm",
    "vacuum_forming",
    "powder_coating",
}


def apply_hard_constraints(features: Dict[str, float]) -> List[str]:
    """
    Applies physical and process manufacturing rules to filter the list of
    feasible process codes before ML classification or ranking is finalized.
    """
    feasible = set(ALL_PROCESSES)

    # 1. Undercuts -> molding and casting constraints
    # undercut_flag is binary (1 or 0)
    undercut = features.get("undercut_flag", 0)
    if undercut == 1:
        feasible.discard("injection_molding")
        feasible.discard("die_casting")
        logger.debug("Constraint: undercut_flag=1. Discarded injection_molding, die_casting.")

    # 2. Part too large for standard injection molding bed
    bb_x = features.get("bounding_box_x", 0.0)
    bb_y = features.get("bounding_box_y", 0.0)
    bb_z = features.get("bounding_box_z", 0.0)
    if bb_x > 600.0 or bb_y > 600.0 or bb_z > 600.0:
        feasible.discard("injection_molding")
        logger.debug("Constraint: Part bounding box > 600mm. Discarded injection_molding.")

    # 3. Thin walls -> Casting and Milling limitations
    wall_min = features.get("wall_thickness_min", 10.0)
    if wall_min < 2.0:
        feasible.discard("sand_casting")
        logger.debug("Constraint: wall_thickness_min < 2mm. Discarded sand_casting.")

    if wall_min < 0.8:
        # Extreme thin walls cannot be injection molded, cast, or forged
        feasible.discard("injection_molding")
        feasible.discard("die_casting")
        feasible.discard("forging")
        logger.debug("Constraint: wall_thickness_min < 0.8mm. Discarded molding/die-casting/forging.")

    if wall_min < 0.5:
        # Cannot CNC machine walls thinner than 0.5mm without breaking
        feasible.discard("cnc_milling")
        feasible.discard("cnc_turning")
        feasible.discard("5axis_cnc")
        feasible.discard("turning_milling")
        logger.debug("Constraint: wall_thickness_min < 0.5mm. Discarded CNC milling and turning.")

    # 4. Sheet metal aspect ratio & thin wall checks
    # Sheet metal is only feasible for thin, flat components (usually wall thickness < 6.0mm)
    wall_avg = features.get("wall_thickness_avg", 5.0)
    if wall_avg > 12.0:
        feasible.discard("sheet_metal_bending")
        feasible.discard("sheet_metal_stamping")
        feasible.discard("laser_cutting")
        logger.debug("Constraint: wall_thickness_avg > 12mm. Discarded sheet metal processes.")

    return list(feasible)
