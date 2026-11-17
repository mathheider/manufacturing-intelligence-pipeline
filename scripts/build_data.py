"""Canonical source of truth for the MIP rule table, feature schema, and the
seed component set. Running this module regenerates the frozen JSON artifacts
that the (zero-dependency) engine loads at runtime:

    data/rules.json        feature schema + 17 processes + cited forbidden_if rules
    data/components.json   the 14 hand-labeled seed components
    data/rules.md          a human-readable mirror of the rule table

Why a build step instead of shipping YAML directly:

  * The engine must load with the Python standard library only (json), so the
    reproducible offline benchmark has no third-party dependency. YAML would
    add PyYAML.
  * Citations are promoted from YAML *comments* to first-class DATA. Every
    forbidden_if clause carries `cite` (the source) and `reason` (the physical
    justification). The pruner surfaces both in each elimination, so the
    provenance the thesis is built on is machine-checkable, not prose.

The original 71 rule values are transcribed verbatim from
mip_agent_prototype/rules.yaml (17-process, literature-cited table). See
docs/rules.md / LIT_REVIEW.md for full page-level citations.

Phase-1 densification (CLAUDE_CODE_GOALS.md goals 1-2): three discriminative
features were added (max_dimension_mm, production_volume,
continuous_fiber_required) and 30 clauses were drafted against the RULE POLICY
(fire only on physical/kinematic infeasibility; build-envelope and
rate-infeasibility clauses per the policy's own examples). Every draft then
went through an adversarial counterexample review (2026-07-01, six independent
domain reviewers): 9 clauses were REFUTED and removed (see the NOTE comments
in place -- e.g. HP-RTM mass production, organosheet overmolding, SLA/SLS
print-farm mass production, one-off molding services), and 7 thresholds were
raised to real documented machine envelopes. 21 proposed clauses survived,
for 92 total. Every surviving NEW clause carries a cite beginning with
"[VERIFY:" -- a proposed source Mathias must check against the actual text
before it counts as cited. Run `python scripts/citation_status.py` for the
proposed-vs-verified table.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# --------------------------------------------------------------------------- #
# Feature schema (11 features). Each carries a type and, for numerics, a unit
# and a plausible range used by the synthetic instance generator. A feature may
# carry an explicit VLM `question`; otherwise planner.question_for builds one
# from the name.
#
# production_volume is an ordinal integer class: 0 = one-off prototype,
# 1 = low-volume production, 2 = high-volume mass production. Encoding it as an
# integer keeps the zero-dependency engine unchanged (eq/gt/lt work) and lets
# MockVLM's +/-1 integer noise model a plausible mis-read of volume class.
# --------------------------------------------------------------------------- #
FEATURES = {
    # The boolean questions are explicit because the templated phrasing was a
    # measured failure mode on real photos: "is there a visible hollow"
    # reads as "any cavity" and caused 58% of the Hybrid's false eliminations
    # (per-feature provenance tally, 2026-07-02, 150-photo run).
    "wall_thickness_mm": {"type": "numeric", "unit": "mm", "range": [0.2, 30.0]},
    "undercut_present": {
        "type": "boolean",
        "question": ("Does this part have an undercut: a feature that would "
                     "lock it inside a rigid one-piece mold, such as an "
                     "internal lip, side hole, or reentrant snap feature? "
                     "Surface recesses that release along the demolding "
                     "direction do not count. Answer yes or no.")},
    "draft_angle_deg": {"type": "numeric", "unit": "deg", "range": [0.0, 10.0]},
    "rib_count": {"type": "integer", "range": [0, 50]},
    "projected_area_cm2": {"type": "numeric", "unit": "cm^2", "range": [1, 50000]},
    "hollow": {
        "type": "boolean",
        "question": ("Is this part a CLOSED hollow body: an enclosed vessel, "
                     "tank, duct, or container whose interior is sealed or "
                     "nearly sealed (like a bottle or fuel tank)? Open trays, "
                     "shells, panels with recesses, and open-backed housings "
                     "are NOT closed hollow bodies. Answer yes or no.")},
    "constant_cross_section": {
        "type": "boolean",
        "question": ("Does this part have one constant cross-section along "
                     "its whole length, like an extruded beam, channel, or "
                     "tube of uniform profile? Answer yes or no.")},
    "axisymmetric": {
        "type": "boolean",
        "question": ("Is this part a body of revolution, rotationally "
                     "symmetric about one axis like a cylinder, cone, or "
                     "tank? Answer yes or no.")},
    # ---- Phase-1 discriminative features (goal 1) -------------------------
    "max_dimension_mm": {
        "type": "numeric", "unit": "mm", "range": [10.0, 20000.0],
        "question": ("Estimate the largest overall dimension of the part "
                     "(its longest bounding-box edge) in millimetres.")},
    "production_volume": {
        "type": "integer", "range": [0, 2],
        "question": ("Classify the production volume this part was made for: "
                     "answer 0 for a one-off prototype, 1 for low-volume "
                     "production, 2 for high-volume mass production.")},
    "continuous_fiber_required": {
        "type": "boolean",
        "question": ("Does this part structurally require continuous "
                     "(unbroken, oriented) fiber reinforcement, as opposed to "
                     "short chopped fiber or unreinforced plastic? "
                     "Answer yes or no.")},
}

PROCESSES = [
    "Injection_Molding", "Thermoforming", "Compression_Molding",
    "Resin_Transfer_Molding", "Vacuum_Assisted_RTM",
    "Structural_Reaction_Injection_Molding", "Sheet_Molding_Compound",
    "Bulk_Molding_Compound", "Hand_Lay_up", "Blow_Molding",
    "Pultrusion", "Filament_Winding", "Overmolding",
    "Fused_Filament_Fabrication", "Stereolithography",
    "Selective_Laser_Sintering", "Laminated_Object_Manufacturing",
]

# --------------------------------------------------------------------------- #
# forbidden_if rules. A tuple is (feature, op, value, cite, reason).
# op in {eq, ne, gt, lt, gte, lte}.
#
# The first block of each process is transcribed from rules.yaml (verified
# citations). Clauses whose cite begins with "[VERIFY:" are Phase-1 proposals
# awaiting Mathias's source check (RULE POLICY: physical/kinematic
# infeasibility only -- build envelopes, definitional material forms, and
# serial-rate infeasibility at mass-production volume).
# --------------------------------------------------------------------------- #
_R = {
    "Injection_Molding": [
        ("wall_thickness_mm", "gt", 8.0, "Malloy 1994 p.151; Rosato 2000 ch.3", "sink/void regime above ~8 mm"),
        ("wall_thickness_mm", "lt", 0.3, "Kutz 2011 ch.11", "below conventional IM fill capability"),
        ("draft_angle_deg", "lt", 0.5, "Malloy 1994 p.205", "matched-tool ejection minimum draft"),
        ("projected_area_cm2", "gt", 50000, "Rosato & Rosato 2000 ch.4", "~4000-ton clamp / platen ceiling"),
        ("continuous_fiber_required", "eq", True, "[VERIFY: Mallick 2007 ch.2; Rosato 2000 ch.1]", "melt injection carries only short/long chopped fiber; cannot place continuous reinforcement"),
        ("max_dimension_mm", "gt", 4000, "[VERIFY: Rosato & Rosato 2000 ch.4; Haitian JU 8800t specs]", "beyond the largest commercial platen/shot envelope (~3.5 m tie-bar daylight; 3.4 m hulls are molded)"),
    ],
    "Thermoforming": [
        ("hollow", "eq", True, "Throne 1996 ch.1 pp.4-6; Throne 2008 ch.1", "single-face draping cannot close a cavity"),
        ("wall_thickness_mm", "gt", 8.0, "Throne 1996 ch.4 pp.150-160", "through-thickness heating / sag limit"),
        ("wall_thickness_mm", "lt", 0.2, "Throne 2008 ch.1", "below commercial sheet gauge"),
        ("rib_count", "gt", 3, "Throne 1996 ch.8", "single-face draping cannot form transverse ribs"),
        ("undercut_present", "eq", True, "Throne 1996 ch.6", "single-face tool cannot release undercuts without side-action"),
        ("max_dimension_mm", "gt", 6000, "[VERIFY: Throne 2008 ch.9; MAAC/Geiss large-format specs]", "beyond the largest commercial sheet/oven envelope (~3.8 x 5.9 m formers; 5.8 m spa shells exist)"),
    ],
    "Compression_Molding": [
        ("wall_thickness_mm", "lt", 0.8, "Mallick 2008 p.376", "charge cannot fill/knit below ~0.8 mm"),
        ("wall_thickness_mm", "gt", 30.0, "Campbell 2004 p.367", "exotherm voids; schema ceiling"),
        ("projected_area_cm2", "gt", 30000, "Campbell 2004 p.366", "~4000-ton commercial press ceiling"),
        ("hollow", "eq", True, "Mallick 2008 p.376; Campbell 2004 p.367", "matched-die closed cavity requires core"),
        ("undercut_present", "eq", True, "Strong 2008 p.331", "matched steel dies cannot release undercuts"),
        ("draft_angle_deg", "lt", 0.5, "Strong 2008 p.331", "matched-die release minimum"),
        ("max_dimension_mm", "gt", 4000, "[VERIFY: Campbell 2004 p.366]", "press platen envelope ceiling"),
    ],
    "Resin_Transfer_Molding": [
        ("wall_thickness_mm", "lt", 1.0, "Potter 1997 p.47; Advani & Sozer 2010 sec.5.4", "preform fill floor"),
        ("wall_thickness_mm", "gt", 25.0, "Campbell 2004 p.158; Mallick 2007 p.383", "exotherm in thick sections"),
        ("draft_angle_deg", "lt", 0.5, "Strong 2008 p.298", "matched steel-tool release requirement"),
        ("hollow", "eq", True, "Potter 1997 ch.6; Campbell 2004 p.160", "no closed cavity without core"),
        ("projected_area_cm2", "gt", 100000, "Mallick 2007 pp.381-384", "practical RTM tool/clamp ceiling"),
        # NOTE: a proposed "forbidden at mass-production volume" clause was
        # REFUTED in adversarial review: HP-RTM ran BMW i3 / 7-Series CFRP
        # structures at automotive series rates (2-5 min cycles).
    ],
    "Vacuum_Assisted_RTM": [
        ("wall_thickness_mm", "lt", 1.5, "Williams et al. 1996 Composites A 27(7); Campbell 2004 p.165", "bag-side tolerance floor"),
        ("undercut_present", "eq", True, "Strong 2008 p.303; Campbell 2004 p.165", "single rigid face cannot release undercuts"),
        ("hollow", "eq", True, "Campbell 2004 p.166", "no closed cavity without mandrel"),
        ("production_volume", "eq", 2, "[VERIFY: Strong 2008 p.303]", "manual bagging + hours-scale infusion per part; mass production infeasible"),
    ],
    "Structural_Reaction_Injection_Molding": [
        ("wall_thickness_mm", "lt", 2.0, "Macosko 1989 pp.205-207; Mallick 2007 p.385", "gel-before-fill in thin cavities"),
        ("wall_thickness_mm", "gt", 15.0, "Macosko 1989 p.210; Campbell 2004 p.169", "runaway exotherm in thick sections"),
        ("draft_angle_deg", "lt", 0.5, "Strong 2008 p.312", "high-rate matched-tool release"),
        ("hollow", "eq", True, "Campbell 2004 p.169", "no closed cavity in a single shot"),
        ("projected_area_cm2", "gt", 30000, "Campbell 2004 p.168", "SRIM press / flow-path ceiling"),
        ("max_dimension_mm", "gt", 3500, "[VERIFY: Campbell 2004 p.168]", "press envelope consistent with the projected-area ceiling"),
        # NOTE: a proposed "forbidden for one-off prototypes" clause was
        # REFUTED in adversarial review: low cavity pressures let prototype
        # RIM shops run one-offs on cast-aluminum/epoxy tools.
    ],
    "Sheet_Molding_Compound": [
        ("wall_thickness_mm", "lt", 1.2, "Mallick 2008 p.371", "25-mm chopped-fiber flow limit"),
        ("wall_thickness_mm", "gt", 10.0, "Orgeas & Dumont 2012 Composites A 43(4) p.630", "exotherm / segregation"),
        ("projected_area_cm2", "lt", 100, "Strong 2008 p.334", "SMC uneconomic below panel scale"),
        ("projected_area_cm2", "gt", 35000, "Campbell 2004 p.369", "~4500-ton press ceiling"),
        ("hollow", "eq", True, "Strong 2008 p.334; Campbell 2004 p.369", "matched-die closed cavity requires core"),
        ("undercut_present", "eq", True, "Strong 2008 p.334", "matched steel dies cannot release undercuts"),
        ("max_dimension_mm", "gt", 4000, "[VERIFY: Campbell 2004 p.369]", "press platen ceiling"),
        # NOTE: a proposed "forbidden for one-off prototypes" clause was
        # REFUTED in adversarial review: molders buy SMC sheet from
        # compounders and mold one-off development parts on prototype tools.
    ],
    "Bulk_Molding_Compound": [
        ("wall_thickness_mm", "lt", 0.6, "Rosato 2004 p.420", "viscosity / filler-loading floor"),
        ("wall_thickness_mm", "gt", 12.0, "Mallick 2008 p.376", "exotherm porosity"),
        ("projected_area_cm2", "gt", 3000, "Strong 2008 p.335", "SMC preferred above panel scale"),
        ("hollow", "eq", True, "Rosato 2004 p.420; Mallick 2008 p.376", "matched-die closed cavity requires core"),
        ("continuous_fiber_required", "eq", True, "[VERIFY: Rosato 2004 ch.11]", "bulk compound is chopped-fiber dough by definition; cannot deliver continuous reinforcement"),
        ("max_dimension_mm", "gt", 2000, "[VERIFY: Rosato 2004 ch.11]", "small-part envelope consistent with the 3000 cm^2 area ceiling"),
        # NOTE: a proposed "forbidden for one-off prototypes" clause was
        # REFUTED in adversarial review: single BMC specimens are lab-press
        # molded routinely.
    ],
    "Hand_Lay_up": [
        ("wall_thickness_mm", "lt", 1.5, "Strong 2008 p.246; Gurit Guide 2020 p.27", "manual consolidation scatter"),
        ("undercut_present", "eq", True, "Strong 2008 p.247; Campbell 2004 p.119", "open mold cannot release undercuts"),
        ("hollow", "eq", True, "Campbell 2004 p.120; Mallick 2007 p.375", "no closed cavity from a single open mold"),
        ("rib_count", "gt", 2, "Strong 2008 p.247", "ribs require secondary moldings beyond simple stiffeners"),
        ("production_volume", "eq", 2, "[VERIFY: Strong 2008 p.247; Mallick 2007 p.375]", "hours of serial manual labor per part; mass production infeasible"),
    ],
    "Blow_Molding": [
        ("hollow", "eq", False, "Rosato 2003 ch.1; Lee 2006 ch.1; Strong 2006 ch.12", "parison inflation is definitional (part must be hollow)"),
        ("wall_thickness_mm", "gt", 10.0, "Rosato 2003 ch.3 pp.60-72; Lee 2006 ch.2", "parison programming / single-side cooling"),
        ("wall_thickness_mm", "lt", 0.25, "Lee 2006 ch.2 pp.20-23", "blowout / pinhole regime"),
        ("rib_count", "gt", 0, "Lee 2006 ch.2; Rosato 2003 ch.2", "parison inflation cannot form transverse ribs"),
        ("undercut_present", "eq", True, "Rosato 2003 ch.2", "split-mold-half release cannot clear undercuts"),
        ("continuous_fiber_required", "eq", True, "[VERIFY: Rosato 2003 ch.1]", "an inflated parison cannot carry continuous fiber reinforcement"),
        ("max_dimension_mm", "gt", 4000, "[VERIFY: Lee 2006 ch.2; accumulator-head kayak practice]", "industrial accumulator-head envelope (~4 m canoes are serially blow-molded)"),
    ],
    "Pultrusion": [
        ("constant_cross_section", "eq", False, "Starr 2000 p.3", "constant cross-section is the defining kinematic"),
        ("undercut_present", "eq", True, "Starr 2000 p.55; Campbell 2004 p.381", "cannot exit die"),
        ("wall_thickness_mm", "lt", 1.5, "Starr 2000 p.50", "impregnation / die-pressure floor"),
        ("wall_thickness_mm", "gt", 25.0, "Mallick 2008 p.395; Carlone 2006 Composites B 37(6)", "exotherm cracking"),
        ("rib_count", "gt", 0, "Starr 2000 p.55", "transverse ribs cannot exit a straight-pull die"),
        # NOTE: a proposed "forbidden for one-off prototypes" clause was
        # REFUTED in adversarial review: lab lines pull one-off experimental
        # profiles and one-offs are cut from stock catalog profiles.
    ],
    "Filament_Winding": [
        ("axisymmetric", "eq", False, "Peters 2011 p.14", "surface-of-revolution kinematic"),
        ("hollow", "eq", False, "Peters 2011 p.14; Campbell 2004 p.388", "mandrel-based shell requires a hollow form"),
        ("undercut_present", "eq", True, "Campbell 2004 p.390", "blocks mandrel extraction"),
        ("rib_count", "gt", 0, "Mallick 2008 p.400", "tow laydown cannot form transverse ribs"),
        ("wall_thickness_mm", "lt", 0.5, "Peters 2011 p.240", "two-layer balanced minimum"),
    ],
    "Overmolding": [
        ("wall_thickness_mm", "gt", 6.0, "Goodship & Love 2002 ch.4; Malloy 1994 p.151", "thick skin over insulating substrate"),
        ("wall_thickness_mm", "lt", 0.3, "Goodship & Love 2002 ch.3; Kutz 2011 ch.12", "unreliable fill around cooler substrate"),
        ("hollow", "eq", True, "Goodship & Love 2002 ch.1; Rosato 2000 ch.22", "substrate + skin topology, not a closed hollow"),
        ("projected_area_cm2", "gt", 20000, "Goodship & Love 2002 ch.2", "two-shot / insert machine envelope"),
        ("max_dimension_mm", "gt", 3000, "[VERIFY: Goodship & Love 2002 ch.2; large 2K press practice]", "multi-component machine envelope (aligned with the injection-molding ceiling class)"),
        # NOTE: two proposed clauses were REFUTED in adversarial review:
        # (a) continuous-fiber forbid was BACKWARDS -- organosheet insert
        # overmolding (Tepex brake pedals, seat shells) is the industrial
        # route TO continuous-fiber parts; (b) one-off forbid fails against
        # prototype insert/overmolding services and RTV/printed molds.
    ],
    "Fused_Filament_Fabrication": [
        ("wall_thickness_mm", "lt", 0.8, "Gibson 2021 ch.6 p.174; Stratasys FDM Guide 2022", "2-perimeter minimum with 0.4 mm nozzle"),
        ("projected_area_cm2", "gt", 10000, "Wohlers 2023 ch.4", "largest commercial FFF/BAAM envelopes"),
        ("max_dimension_mm", "gt", 2200, "[VERIFY: Tractus3D T3500 specifications]", "largest filament-fed commercial envelope (~2.1 m); pellet-extrusion BAAM/LSAM is a distinct machine class"),
        ("production_volume", "eq", 2, "[VERIFY: Gibson 2021 ch.6]", "hours-per-part deposition rate; serially infeasible at mass-production volume"),
    ],
    "Stereolithography": [
        ("wall_thickness_mm", "lt", 0.4, "Formlabs Design Guide 2023; Gibson 2021 ch.4 p.93", "photopolymer minimum wall"),
        ("projected_area_cm2", "gt", 15000, "Wohlers 2023 ch.4", "Mammoth-class SLA upper bound"),
        ("continuous_fiber_required", "eq", True, "[VERIFY: Gibson 2021 ch.4]", "vat photopolymer is neat resin; no continuous fiber reinforcement path"),
        ("max_dimension_mm", "gt", 2100, "[VERIFY: Materialise Mammoth SLA specifications]", "largest commercial vat envelope (~2.1 m)"),
        # NOTE: a proposed "forbidden at mass-production volume" clause was
        # REFUTED in adversarial review: Align Technology prints ~500k+
        # aligner molds/day on SLA farms.
    ],
    "Selective_Laser_Sintering": [
        ("wall_thickness_mm", "lt", 0.7, "EOS PA2200 datasheet 2021; Gibson 2021 ch.5 p.126", "powder-fusion minimum wall"),
        ("projected_area_cm2", "gt", 10000, "Wohlers 2023 ch.4", "commercial SLS bed ceilings"),
        ("continuous_fiber_required", "eq", True, "[VERIFY: Gibson 2021 ch.5]", "powder-bed fusion cannot lay continuous fiber"),
        ("max_dimension_mm", "gt", 1000, "[VERIFY: Farsoon HT1001P specifications]", "largest commercial powder-bed envelope (1000 mm class)"),
        # NOTE: a proposed "forbidden at mass-production volume" clause was
        # REFUTED in adversarial review: nested SLS mass production is real
        # (~1M/month mascara brushes, Erpro Group).
    ],
    "Laminated_Object_Manufacturing": [
        ("hollow", "eq", True, "Gibson 2021 ch.9 pp.257-258; Chua et al. 2010 ch.4", "decube extraction infeasible in a closed cavity"),
        ("wall_thickness_mm", "lt", 1.0, "Gibson 2021 ch.9 p.260", "sheet delamination during decubing"),
        ("projected_area_cm2", "gt", 5000, "Wohlers 2023 ch.4; Helisys LOM-2030 upper bound", "envelope ceiling"),
        ("rib_count", "gt", 2, "Gibson 2021 ch.9 p.260", "stepped z-axis rib profiles brittle across laminae"),
        ("max_dimension_mm", "gt", 820, "[VERIFY: Helisys LOM-2030H specifications (813 mm long axis)]", "laminating envelope ceiling"),
        ("production_volume", "eq", 2, "[VERIFY: Gibson 2021 ch.9]", "sheet-by-sheet lamination rate; mass production infeasible"),
    ],
}


def _rules_dict() -> dict:
    rules = {}
    for proc, clauses in _R.items():
        rules[proc] = {
            "forbidden_if": [
                {"feature": f, "op": op, "value": v, "cite": cite, "reason": reason}
                for (f, op, v, cite, reason) in clauses
            ]
        }
    return rules


# --------------------------------------------------------------------------- #
# Seed component set: 14 hand-labeled polymer-composite automotive parts.
# Each expert process is rule-feasible under perfect perception (asserted by
# tests/test_dataset_feasibility.py). Transcribed from dataset.py; the three
# Phase-1 features were hand-assigned per part and checked against every
# expert process's clause set (e.g. parts with RTM in the expert set are
# low-volume, matching real composite-bodywork practice).
# --------------------------------------------------------------------------- #
COMPONENTS = {
    "steering_wheel": (
        "steering wheel w/ airbag cover, metal insert + overmold",
        {"wall_thickness_mm": 3.0, "undercut_present": False, "draft_angle_deg": 1.5,
         "rib_count": 3, "projected_area_cm2": 1200.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 400.0, "production_volume": 2,
         "continuous_fiber_required": False},
        ["Injection_Molding"]),
    "intake_manifold": (
        "plastic intake manifold (curved runners, mounting flanges)",
        {"wall_thickness_mm": 3.0, "undercut_present": False, "draft_angle_deg": 1.5,
         "rib_count": 2, "projected_area_cm2": 600.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 500.0, "production_volume": 2,
         "continuous_fiber_required": False},
        ["Injection_Molding"]),
    "washer_fluid_reservoir": (
        "small hollow plastic reservoir with cap and hose barb",
        {"wall_thickness_mm": 2.0, "undercut_present": False, "draft_angle_deg": 1.0,
         "rib_count": 0, "projected_area_cm2": 300.0, "hollow": True,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 350.0, "production_volume": 2,
         "continuous_fiber_required": False},
        ["Blow_Molding"]),
    "door_inner_panel": (
        "interior door trim panel w/ storage, textured skin",
        {"wall_thickness_mm": 2.5, "undercut_present": False, "draft_angle_deg": 2.0,
         "rib_count": 5, "projected_area_cm2": 4000.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 1100.0, "production_volume": 2,
         "continuous_fiber_required": False},
        ["Injection_Molding"]),
    "pickup_bed": (
        "pickup truck bed / bed liner, large deep draw",
        {"wall_thickness_mm": 3.0, "undercut_present": False, "draft_angle_deg": 3.0,
         "rib_count": 1, "projected_area_cm2": 20000.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 2400.0, "production_volume": 1,
         "continuous_fiber_required": False},
        ["Thermoforming", "Resin_Transfer_Molding", "Sheet_Molding_Compound"]),
    "rear_spoiler": (
        "rear deck spoiler, conformal to bodywork",
        {"wall_thickness_mm": 3.0, "undercut_present": False, "draft_angle_deg": 2.0,
         "rib_count": 2, "projected_area_cm2": 1500.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 1400.0, "production_volume": 1,
         "continuous_fiber_required": False},
        ["Injection_Molding", "Resin_Transfer_Molding", "Compression_Molding"]),
    "hood": (
        "car hood - aerodynamic, bodywork-conformal outer panel",
        {"wall_thickness_mm": 2.5, "undercut_present": False, "draft_angle_deg": 2.0,
         "rib_count": 2, "projected_area_cm2": 12000.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 1500.0, "production_volume": 1,
         "continuous_fiber_required": False},
        ["Resin_Transfer_Molding", "Compression_Molding", "Sheet_Molding_Compound"]),
    "cf_cylinder": (
        "carbon-fiber axisymmetric hollow cylinder",
        {"wall_thickness_mm": 2.5, "undercut_present": False, "draft_angle_deg": 0.0,
         "rib_count": 0, "projected_area_cm2": 900.0, "hollow": True,
         "constant_cross_section": True, "axisymmetric": True,
         "max_dimension_mm": 1000.0, "production_volume": 1,
         "continuous_fiber_required": True},
        ["Pultrusion", "Filament_Winding"]),
    "leaf_spring": (
        "FRP composite leaf spring, long ~constant cross section",
        {"wall_thickness_mm": 12.0, "undercut_present": False, "draft_angle_deg": 1.0,
         "rib_count": 0, "projected_area_cm2": 1500.0, "hollow": False,
         "constant_cross_section": True, "axisymmetric": False,
         "max_dimension_mm": 1200.0, "production_volume": 1,
         "continuous_fiber_required": True},
        ["Resin_Transfer_Molding", "Pultrusion"]),
    "fuel_tank": (
        "single-piece plastic automotive fuel tank (hollow)",
        {"wall_thickness_mm": 4.0, "undercut_present": False, "draft_angle_deg": 1.5,
         "rib_count": 0, "projected_area_cm2": 3500.0, "hollow": True,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 1000.0, "production_volume": 2,
         "continuous_fiber_required": False},
        ["Blow_Molding"]),
    "seat_frame": (
        "car seat bucket w/ metal insert, plastic frame",
        {"wall_thickness_mm": 3.0, "undercut_present": False, "draft_angle_deg": 1.5,
         "rib_count": 4, "projected_area_cm2": 3000.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 1200.0, "production_volume": 2,
         "continuous_fiber_required": False},
        ["Injection_Molding"]),
    "headliner_trim": (
        "roof headliner + pillar trim, thin large curved",
        {"wall_thickness_mm": 2.0, "undercut_present": False, "draft_angle_deg": 3.0,
         "rib_count": 0, "projected_area_cm2": 15000.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 1800.0, "production_volume": 2,
         "continuous_fiber_required": False},
        ["Thermoforming", "Injection_Molding"]),
    "dashboard": (
        "dashboard / instrument cluster housing",
        {"wall_thickness_mm": 3.0, "undercut_present": False, "draft_angle_deg": 2.0,
         "rib_count": 3, "projected_area_cm2": 6000.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 1600.0, "production_volume": 2,
         "continuous_fiber_required": False},
        ["Injection_Molding"]),
    "headlamp_assembly": (
        "headlamp housing - lens, housing, trim",
        {"wall_thickness_mm": 2.5, "undercut_present": False, "draft_angle_deg": 1.5,
         "rib_count": 2, "projected_area_cm2": 800.0, "hollow": False,
         "constant_cross_section": False, "axisymmetric": False,
         "max_dimension_mm": 500.0, "production_volume": 2,
         "continuous_fiber_required": False},
        ["Injection_Molding"]),
}


def _components_list() -> list:
    return [
        {"id": cid, "description": desc, "image": f"{cid}.png",
         "features": feats, "expert_processes": procs}
        for cid, (desc, feats, procs) in COMPONENTS.items()
    ]


def _rules_markdown(rules: dict) -> str:
    lines = ["# MIP rule table (generated from scripts/build_data.py)\n",
             f"{len(PROCESSES)} processes, {len(FEATURES)} features, "
             f"{sum(len(r['forbidden_if']) for r in rules.values())} cited forbidden_if clauses.\n"]
    for proc in PROCESSES:
        lines.append(f"\n## {proc}\n")
        lines.append("| feature | op | value | source | reason |")
        lines.append("|---|---|---|---|---|")
        for c in rules[proc]["forbidden_if"]:
            lines.append(f"| `{c['feature']}` | {c['op']} | {c['value']} | {c['cite']} | {c['reason']} |")
    return "\n".join(lines) + "\n"


def build() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rules = {"processes": PROCESSES, "features": FEATURES, "rules": _rules_dict()}
    (DATA_DIR / "rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
    (DATA_DIR / "components.json").write_text(
        json.dumps(_components_list(), indent=2), encoding="utf-8")
    (DATA_DIR / "rules.md").write_text(_rules_markdown(rules["rules"]), encoding="utf-8")

    n_clauses = sum(len(r["forbidden_if"]) for r in rules["rules"].values())
    n_verify = sum(1 for r in rules["rules"].values()
                   for c in r["forbidden_if"] if c["cite"].startswith("[VERIFY"))
    print(f"wrote {DATA_DIR/'rules.json'}  "
          f"({len(PROCESSES)} processes, {len(FEATURES)} features, {n_clauses} clauses, "
          f"{n_verify} awaiting citation verification)")
    print(f"wrote {DATA_DIR/'components.json'}  ({len(COMPONENTS)} components)")
    print(f"wrote {DATA_DIR/'rules.md'}")


if __name__ == "__main__":
    build()
