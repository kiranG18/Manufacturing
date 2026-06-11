# tests/test_constraint_filter.py
import pytest
from app.services.constraint_filter import apply_hard_constraints

def test_no_constraints():
    features = {
        "bounding_box_x": 50.0,
        "bounding_box_y": 50.0,
        "bounding_box_z": 10.0,
        "wall_thickness_min": 5.0,
        "wall_thickness_avg": 6.0,
        "undercut_flag": 0
    }
    allowed = apply_hard_constraints(features)
    assert "injection_molding" in allowed
    assert "cnc_milling" in allowed
    assert "sand_casting" in allowed

def test_undercut_constraint():
    features = {
        "bounding_box_x": 50.0,
        "bounding_box_y": 50.0,
        "bounding_box_z": 10.0,
        "wall_thickness_min": 5.0,
        "wall_thickness_avg": 6.0,
        "undercut_flag": 1
    }
    allowed = apply_hard_constraints(features)
    assert "injection_molding" not in allowed
    assert "die_casting" not in allowed
    assert "cnc_milling" in allowed

def test_large_part_constraint():
    features = {
        "bounding_box_x": 650.0,  # > 600mm
        "bounding_box_y": 50.0,
        "bounding_box_z": 10.0,
        "wall_thickness_min": 5.0,
        "wall_thickness_avg": 6.0,
        "undercut_flag": 0
    }
    allowed = apply_hard_constraints(features)
    assert "injection_molding" not in allowed
    assert "cnc_milling" in allowed

def test_thin_wall_constraint():
    features = {
        "bounding_box_x": 50.0,
        "bounding_box_y": 50.0,
        "bounding_box_z": 10.0,
        "wall_thickness_min": 1.8,  # < 2.0mm
        "wall_thickness_avg": 3.0,
        "undercut_flag": 0
    }
    allowed = apply_hard_constraints(features)
    assert "sand_casting" not in allowed
    assert "cnc_milling" in allowed
    
    # Extreme thin wall (< 0.8mm)
    features["wall_thickness_min"] = 0.7
    allowed = apply_hard_constraints(features)
    assert "injection_molding" not in allowed
    assert "die_casting" not in allowed
    assert "forging" not in allowed
    
    # CNC thin wall limit (< 0.5mm)
    features["wall_thickness_min"] = 0.4
    allowed = apply_hard_constraints(features)
    assert "cnc_milling" not in allowed
    assert "cnc_turning" not in allowed
    assert "5axis_cnc" not in allowed

def test_sheet_metal_thick_wall_constraint():
    features = {
        "bounding_box_x": 100.0,
        "bounding_box_y": 100.0,
        "bounding_box_z": 100.0,
        "wall_thickness_min": 8.0,
        "wall_thickness_avg": 15.0,  # too thick for sheet metal
        "undercut_flag": 0
    }
    allowed = apply_hard_constraints(features)
    assert "sheet_metal_bending" not in allowed
    assert "sheet_metal_stamping" not in allowed
    assert "cnc_milling" in allowed
