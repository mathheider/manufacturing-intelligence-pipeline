# -*- coding: utf-8 -*-
"""Build data/rules_v3.json: the expanded, soft-evidence rule base.

The frozen v2 table (data/rules.json, 92 forbidden_if clauses, preregistered
bench-v2) is NOT modified. v3 is a superset in a new file:

  G1  v2 hard limits, kept verbatim but as strong soft "forbid" evidence (92)
  G2  process x feature typicality matrix on the 11 existing features
  G3  combination rules (2-3 features, AND)
  G4  visible manufacturing signatures: 28 NEW perception features (marks a
      photo can actually show) and the rules on them. These need one new VLM
      pass; until then they have no samples and contribute nothing.
  G5  fiber-architecture exclusions (continuous vs chopped), both directions

Weights are log-odds-style evidence: 3.0 = physical limit, 1.5-2.5 = near
definitional signature, 0.75-1.25 = strong tendency, 0.25-0.5 = mild tendency.
A single global scale is tuned on the dev split; individual weights are set
from domain knowledge and are NOT fitted to labels.

Citation policy: every v3 rule cites works already in the v2 reference base,
by topic, with status "v3-draft [VERIFY]". No page numbers are claimed for
v3 rules; they are to be added when each rule is checked (same workflow as the
21 [VERIFY] v2 clauses), followed by the adversarial counterexample pass.

    python scripts/build_rules_v3.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
V2 = json.loads((MIP / "data" / "rules.json").read_text(encoding="utf-8"))
PROCS = V2["processes"]

IM, TF, CM, RTM, VARTM, SRIM, SMC, BMC, HLU, BM, PUL, FW, OM, FFF, SLA, SLS, LOM = PROCS
ADDITIVE = [FFF, SLA, SLS, LOM]
MATCHED_DIE = [IM, OM, CM, SMC, BMC, RTM, SRIM]
OPEN_MOLD = [HLU, VARTM]
CONT_FIBER = [RTM, VARTM, HLU, PUL, FW]

OBS = {"hollow": "visible", "constant_cross_section": "visible", "axisymmetric": "visible",
       "undercut_present": "visible", "rib_count": "visible",
       "max_dimension_mm": "estimable", "projected_area_cm2": "estimable",
       "wall_thickness_mm": "not_visible", "draft_angle_deg": "not_visible",
       "production_volume": "not_visible", "continuous_fiber_required": "not_visible"}
OBS_RANK = {"not_visible": 0, "estimable": 1, "visible": 2, "signature": 2}

# ---- citations (works already in the v2 reference base; topic-level, to verify)
CITE = {
    IM: "Malloy 1994; Rosato & Rosato 2000",
    OM: "Goodship & Love 2002",
    BM: "Rosato 2003; Lee 2006",
    TF: "Throne 1996; Throne 2008",
    CM: "Strong 2008; Mallick 2008; Campbell 2004",
    SMC: "Strong 2008; Mallick 2008; Campbell 2004",
    BMC: "Rosato 2004; Mallick 2008",
    RTM: "Potter 1997; Advani & Sozer 2010; Campbell 2004",
    VARTM: "Williams et al. 1996; Campbell 2004; Strong 2008",
    SRIM: "Macosko 1989; Campbell 2004",
    HLU: "Strong 2008; Campbell 2004; Mallick 2007",
    PUL: "Starr 2000; Mallick 2008",
    FW: "Peters 2011; Campbell 2004",
    FFF: "Gibson et al. 2021",
    SLA: "Gibson et al. 2021",
    SLS: "Gibson et al. 2021",
    LOM: "Gibson et al. 2021; Chua et al. 2010",
}

RULES: list = []
_n = Counter()


def C(feature, op, value):
    return {"feature": feature, "op": op, "value": value}


def add(group, proc, kind, when, weight, reason, topic, obs=None):
    feats = [c["feature"] for c in when]
    if obs is None:
        classes = [OBS.get(f, "signature") for f in feats]
        obs = min(classes, key=lambda c: OBS_RANK[c])
    _n[group] += 1
    RULES.append({
        "id": f"{group}-{_n[group]:03d}", "group": group, "process": proc, "kind": kind,
        "when": when, "weight": weight, "observability": obs,
        "reason": reason,
        "cite": f"[VERIFY: {CITE[proc]} ({topic})]",
        "status": "v3-draft [VERIFY]"})


def favor(group, procs, when, w, reason, topic):
    for p in ([procs] if isinstance(procs, str) else procs):
        add(group, p, "favor", when, w, reason, topic)


def disfavor(group, procs, when, w, reason, topic):
    for p in ([procs] if isinstance(procs, str) else procs):
        add(group, p, "disfavor", when, w, reason, topic)


# ============================================================ G1: v2 limits
i = 0
for proc, body in V2["rules"].items():
    for c in body["forbidden_if"]:
        i += 1
        RULES.append({
            "id": f"V2-{i:03d}", "group": "G1", "process": proc, "kind": "forbid",
            "when": [C(c["feature"], c["op"], c["value"])], "weight": 3.0,
            "observability": OBS[c["feature"]], "reason": c.get("reason", ""),
            "cite": c.get("cite", ""),
            "status": "v2-VERIFY" if "VERIFY" in c.get("cite", "") else "v2-verified"})

# ============================================================ G2: typicality matrix
G = "G2"
WT, UC, DA, RIB, AREA = "wall_thickness_mm", "undercut_present", "draft_angle_deg", "rib_count", "projected_area_cm2"
HOL, CCS, AXI, DIM, VOL, CF = "hollow", "constant_cross_section", "axisymmetric", "max_dimension_mm", "production_volume", "continuous_fiber_required"

# --- wall thickness windows (typical design ranges)
favor(G, IM, [C(WT, "gte", 0.8), C(WT, "lte", 4.0)], 0.5, "typical injection-molded wall 0.8-4 mm", "nominal wall design")
disfavor(G, IM, [C(WT, "gt", 6.0)], 1.0, "thick walls sink and void; IM parts are cored out", "nominal wall design")
favor(G, OM, [C(WT, "gte", 0.8), C(WT, "lte", 4.0)], 0.25, "overmold skins and substrates are thin-walled", "skin and substrate thickness")
favor(G, BM, [C(WT, "gte", 0.5), C(WT, "lte", 5.0)], 0.5, "blown walls are thin and parison-limited", "wall distribution")
favor(G, TF, [C(WT, "gte", 0.5), C(WT, "lte", 6.0)], 0.5, "formed from commercial sheet gauges", "sheet gauge")
favor(G, CM, [C(WT, "gte", 2.0), C(WT, "lte", 10.0)], 0.5, "compression charges favor 2-10 mm sections", "charge and section thickness")
favor(G, SMC, [C(WT, "gte", 1.5), C(WT, "lte", 6.0)], 0.5, "SMC panels are typically 1.5-6 mm", "panel thickness")
favor(G, BMC, [C(WT, "gte", 1.0), C(WT, "lte", 6.0)], 0.25, "BMC parts are small, moderately thick", "section thickness")
favor(G, RTM, [C(WT, "gte", 2.0), C(WT, "lte", 10.0)], 0.25, "preform laminates 2-10 mm", "laminate thickness")
favor(G, VARTM, [C(WT, "gte", 2.0), C(WT, "lte", 20.0)], 0.25, "infused laminates and sandwich skins", "laminate thickness")
favor(G, SRIM, [C(WT, "gte", 2.0), C(WT, "lte", 10.0)], 0.25, "structural RIM sections", "section thickness")
favor(G, HLU, [C(WT, "gte", 2.0), C(WT, "lte", 15.0)], 0.25, "hand-laminated thickness range", "laminate thickness")
favor(G, PUL, [C(WT, "gte", 2.0), C(WT, "lte", 15.0)], 0.25, "pultruded profile walls", "profile wall")
disfavor(G, ADDITIVE, [C(WT, "gt", 10.0)], 0.5, "thick solid sections are slow/costly to build additively", "build-time economics")

# --- ribs
favor(G, IM, [C(RIB, "gte", 3)], 0.75, "ribbed, cored-out geometry is the IM norm", "rib design")
favor(G, BMC, [C(RIB, "gte", 2)], 0.25, "BMC fills ribs and bosses under flow", "rib and boss filling")
favor(G, SMC, [C(RIB, "gte", 1)], 0.25, "SMC can mold ribs and bosses on panels", "ribs on panels")
favor(G, CM, [C(RIB, "gte", 1)], 0.25, "matched dies form ribs", "ribs")
disfavor(G, [VARTM, HLU], [C(RIB, "gte", 3)], 0.5, "many ribs need secondary bonding in open molding", "stiffener practice")
favor(G, [TF, HLU, VARTM, FW, PUL, BM], [C(RIB, "eq", 0)], 0.25, "single-sided or kinematic processes give rib-free parts", "rib-free geometry")

# --- undercuts
favor(G, [IM, OM], [C(UC, "eq", True)], 0.5, "slides and lifters make undercuts routine in IM tooling", "side actions")
favor(G, ADDITIVE, [C(UC, "eq", True)], 0.25, "layerwise builds have no draw direction", "geometric freedom")
disfavor(G, [RTM, SRIM], [C(UC, "eq", True)], 0.5, "matched rigid tools release poorly over undercuts", "tool release")

# --- closed hollow bodies
favor(G, BM, [C(HOL, "eq", True)], 2.0, "parison inflation makes closed hollow bodies by definition", "process definition")
disfavor(G, IM, [C(HOL, "eq", True)], 0.75, "single-shot IM cannot close a cavity (two halves need welding)", "core withdrawal")
favor(G, FW, [C(HOL, "eq", True)], 0.75, "wound over a mandrel: shells and vessels", "mandrel winding")
disfavor(G, ADDITIVE[:3], [C(HOL, "eq", True)], 0.25, "sealed cavities trap support or powder", "trapped material")

# --- constant cross-section
favor(G, PUL, [C(CCS, "eq", True)], 2.5, "a constant profile is what a pultrusion die makes", "process definition")
disfavor(G, [IM, CM, SMC, BMC, TF, HLU, BM], [C(CCS, "eq", True)], 0.5, "long constant profiles indicate a continuous process", "profile processes")

# --- axisymmetry
favor(G, FW, [C(AXI, "eq", True)], 1.5, "winding needs a body of revolution", "winding kinematics")
favor(G, BM, [C(AXI, "eq", True)], 0.5, "bottles and tanks are often axisymmetric", "container geometry")
disfavor(G, [SMC, HLU, VARTM, TF], [C(AXI, "eq", True)], 0.25, "panel processes rarely make bodies of revolution", "panel geometry")

# --- size (max dimension, mm)
favor(G, [IM, OM, BMC], [C(DIM, "lt", 300.0)], 0.5, "small precise parts are the high-volume molding domain", "part size")
favor(G, ADDITIVE, [C(DIM, "lt", 400.0)], 0.5, "build envelopes favor small parts", "build envelope")
disfavor(G, ADDITIVE, [C(DIM, "gt", 1000.0)], 1.0, "beyond common build envelopes", "build envelope")
disfavor(G, [IM, OM], [C(DIM, "gt", 2000.0)], 1.0, "beyond common clamp/platen sizes", "machine envelope")
favor(G, [HLU, VARTM], [C(DIM, "gt", 1500.0)], 1.0, "very large parts are open-molded or infused", "large structures")
favor(G, [SMC, TF], [C(DIM, "gte", 500.0), C(DIM, "lte", 3000.0)], 0.5, "panel-scale parts", "panel scale")
favor(G, CM, [C(DIM, "gte", 150.0), C(DIM, "lte", 2000.0)], 0.25, "press-scale parts", "press scale")
favor(G, [RTM, SRIM], [C(DIM, "gte", 300.0), C(DIM, "lte", 2500.0)], 0.25, "structural components", "part scale")
favor(G, PUL, [C(DIM, "gt", 500.0)], 0.5, "profiles are cut from continuous lengths", "profile length")
disfavor(G, BMC, [C(DIM, "gt", 800.0)], 0.75, "BMC is a small-part compound", "part size")

# --- projected area (cm^2)
favor(G, SMC, [C(AREA, "gte", 500.0), C(AREA, "lte", 35000.0)], 0.75, "body-panel scale", "panel area")
favor(G, TF, [C(AREA, "gt", 1000.0)], 0.5, "large shallow shells from sheet", "sheet forming area")
favor(G, [IM, OM, BMC], [C(AREA, "lt", 1500.0)], 0.25, "clamp-tonnage friendly areas", "clamp tonnage")
favor(G, VARTM, [C(AREA, "gt", 10000.0)], 0.75, "infusion scales to very large areas", "infusion scale")

# --- production volume (0 prototype, 1 low, 2 mass)
favor(G, [IM, BM, SMC, BMC], [C(VOL, "eq", 2)], 0.75, "hard tooling pays off only at volume", "tooling economics")
favor(G, [OM, CM, SRIM, PUL], [C(VOL, "eq", 2)], 0.5, "automated high-rate processes", "cycle time")
disfavor(G, [IM, OM, BM, SMC, BMC, CM], [C(VOL, "eq", 0)], 1.0, "one-off parts do not justify steel tooling", "tooling economics")
favor(G, [FFF, SLA, SLS, LOM], [C(VOL, "eq", 0)], 1.5, "additive is the prototype route", "prototyping")
favor(G, SLS, [C(VOL, "eq", 1)], 0.5, "powder-bed fusion serves low-volume production", "low-volume AM")
disfavor(G, [SLA, LOM], [C(VOL, "eq", 2)], 1.0, "build rate rules out mass production", "build rate")
disfavor(G, SLS, [C(VOL, "eq", 2)], 0.75, "build rate limits mass production", "build rate")
favor(G, [HLU, VARTM], [C(VOL, "lte", 1)], 0.75, "manual/infusion processes suit low volume", "labor content")
favor(G, [RTM, TF], [C(VOL, "eq", 1)], 0.5, "medium tooling cost suits low-to-medium volume", "tooling economics")

# --- draft (weak; rarely observable)
favor(G, MATCHED_DIE, [C(DA, "gte", 1.0)], 0.25, "draft on walls signals rigid matched tooling", "draft design")

# ============================================================ G5: fiber architecture
G = "G5"
favor(G, [RTM, VARTM, HLU], [C(CF, "eq", True)], 1.5, "continuous fabric/preform reinforcement", "reinforcement form")
favor(G, [FW, PUL], [C(CF, "eq", True)], 1.25, "continuous tows are the feedstock", "reinforcement form")
favor(G, [CM, SRIM], [C(CF, "eq", True)], 0.5, "prepreg compression and SRIM preforms carry continuous fiber", "reinforcement form")
disfavor(G, [TF, OM, SMC, FFF, LOM], [C(CF, "eq", True)], 0.75, "short or no fiber in the feedstock", "reinforcement form")
disfavor(G, [FW, PUL], [C(CF, "eq", False)], 1.25, "these processes exist to place continuous fiber", "process definition")
disfavor(G, [VARTM, RTM, HLU], [C(CF, "eq", False)], 0.5, "laminating processes usually carry continuous fabric", "reinforcement form")
favor(G, [IM, BM, TF, SMC, BMC, FFF, SLA, SLS], [C(CF, "eq", False)], 0.25, "unreinforced or short-fiber feedstock", "feedstock")

# ============================================================ G3: combinations
G = "G3"
favor(G, [BM, FW], [C(HOL, "eq", True), C(AXI, "eq", True)], 1.0, "closed body of revolution: bottle, tank, vessel", "vessel geometry")
disfavor(G, [IM, CM, SMC, BMC, TF, HLU], [C(HOL, "eq", True), C(AXI, "eq", True)], 0.5, "closed body of revolution is not a panel or molded shell", "vessel geometry")
favor(G, BM, [C(HOL, "eq", True), C(AXI, "eq", False)], 0.5, "irregular closed hollow bodies (tanks, ducts) are blow molded", "technical blow molding")
disfavor(G, FW, [C(HOL, "eq", True), C(AXI, "eq", False)], 1.0, "irregular closed bodies cannot be wound", "winding kinematics")
favor(G, PUL, [C(CCS, "eq", True), C(DIM, "gt", 1000.0)], 1.0, "long constant profile", "profile length")
disfavor(G, [IM, CM, SMC, BMC, TF], [C(CCS, "eq", True), C(DIM, "gt", 1000.0)], 0.75, "long constant profile is not a molded part", "profile processes")
favor(G, [PUL, FW], [C(CCS, "eq", True), C(AXI, "eq", True)], 0.75, "tube or rod", "tubular parts")
favor(G, PUL, [C(CCS, "eq", True), C(CF, "eq", True)], 1.0, "continuous-fiber constant profile", "pultruded profiles")
favor(G, FW, [C(AXI, "eq", True), C(CF, "eq", True)], 1.0, "continuous-fiber body of revolution", "wound structures")
favor(G, FW, [C(HOL, "eq", True), C(AXI, "eq", True), C(CF, "eq", True)], 1.0, "composite pressure vessel", "pressure vessels")
favor(G, [HLU, VARTM], [C(DIM, "gt", 2000.0), C(VOL, "lte", 1)], 1.0, "very large, low-volume part", "large structures")
favor(G, [HLU, VARTM], [C(CF, "eq", True), C(DIM, "gt", 1500.0)], 1.0, "large continuous-fiber structure", "large laminates")
favor(G, [RTM, CM], [C(CF, "eq", True), C(VOL, "eq", 2)], 0.75, "volume continuous-fiber parts are pressed or HP-RTM", "automotive composites")
favor(G, SRIM, [C(CF, "eq", True), C(VOL, "gte", 1)], 0.25, "structural RIM with preforms", "SRIM")
favor(G, [HLU, VARTM], [C(CF, "eq", True), C(VOL, "eq", 0)], 1.0, "one-off continuous-fiber part", "prototype laminates")
favor(G, [FFF, SLA, SLS], [C(CF, "eq", False), C(VOL, "eq", 0)], 0.5, "one-off unreinforced part", "prototyping")
favor(G, [FFF, SLA, SLS], [C(DIM, "lt", 200.0), C(VOL, "eq", 0)], 0.75, "small one-off part", "prototyping")
favor(G, [IM, BMC, OM], [C(DIM, "lt", 200.0), C(VOL, "eq", 2)], 1.0, "small mass-produced part", "high-volume molding")
favor(G, IM, [C(RIB, "gte", 3), C(UC, "eq", True)], 1.0, "ribs plus undercuts: complex molded geometry", "complex moldings")
favor(G, [IM, BMC], [C(RIB, "gte", 3), C(DIM, "lt", 600.0)], 0.75, "small ribbed part", "ribbed moldings")
favor(G, SMC, [C(RIB, "gte", 1), C(DIM, "gt", 800.0)], 0.75, "large ribbed panel", "ribbed panels")
favor(G, TF, [C(WT, "lt", 4.0), C(DIM, "gt", 500.0), C(RIB, "eq", 0)], 1.0, "large thin rib-free shell", "formed shells")
favor(G, TF, [C(AREA, "gt", 2000.0), C(UC, "eq", False), C(RIB, "eq", 0)], 0.75, "large open shell without undercuts or ribs", "formed shells")
favor(G, BM, [C(HOL, "eq", True), C(WT, "lt", 4.0), C(VOL, "eq", 2)], 1.0, "thin closed hollow part at volume", "packaging and tanks")
favor(G, SMC, [C(AREA, "gt", 3000.0), C(VOL, "eq", 2), C(CF, "eq", False)], 0.75, "large short-fiber panel at volume", "SMC panels")
favor(G, ADDITIVE[:3], [C(UC, "eq", True), C(VOL, "eq", 0)], 0.5, "complex one-off geometry", "design freedom")
disfavor(G, [HLU, VARTM, PUL, FW], [C(DIM, "lt", 150.0), C(VOL, "eq", 2)], 0.75, "tiny mass-produced parts are not laminated", "process economics")
disfavor(G, [IM, OM, SMC, BMC], [C(DIM, "gt", 1500.0), C(VOL, "eq", 0)], 1.0, "huge one-off part with steel tooling is implausible", "tooling economics")
favor(G, OM, [C(UC, "eq", True), C(DIM, "lt", 400.0), C(VOL, "eq", 2)], 0.5, "small multi-feature part at volume", "multi-shot parts")
favor(G, LOM, [C(VOL, "eq", 0), C(DIM, "lt", 800.0), C(HOL, "eq", False)], 0.5, "solid one-off within laminating envelope", "sheet lamination")

# ============================================================ G4: visible signatures (NEW features)
G = "G4"
SIG = {
    "sig_ejector_pin_marks": "Look at any back, inner or hidden surface. Do you see small flat circular marks (roughly 3-10 mm across), often several in a row, left by the ejector pins of an injection mold? Answer yes or no.",
    "sig_gate_vestige": "Do you see a small nub, scar or cut-off stub where molten plastic entered the mold (a gate mark), typically at an edge or on a hidden face? Answer yes or no.",
    "sig_parting_line": "Do you see a thin continuous line or slight step running around the part where two mold halves met (a parting line)? Answer yes or no.",
    "sig_pinch_off_seam": "Is there a flattened, pinched seam across the bottom, top or end of a closed hollow body, where a molten tube was pinched shut (typical of blow-molded bottles and tanks)? Answer yes or no.",
    "sig_neck_or_spout": "Does the part have a bottle-like neck, spout or threaded filler opening leading into a hollow interior? Answer yes or no.",
    "sig_woven_fabric_visible": "Can you see a woven fabric pattern (plain or twill weave of carbon, glass or aramid fiber) on any surface or under clear resin? Answer yes or no.",
    "sig_random_chopped_fiber": "Can you see randomly oriented short glass fibers or a chopped-strand mat texture on the surface or at a cut edge? Answer yes or no.",
    "sig_helical_winding_pattern": "Do you see helical or criss-crossing bands of fiber wrapped around the part, forming a diamond pattern (typical of filament winding)? Answer yes or no.",
    "sig_longitudinal_fiber_streaks": "Do you see fine fiber streaks or lines running straight along the full length of the part, as on a pultruded rod or profile? Answer yes or no.",
    "sig_layer_lines": "Do you see regular horizontal layer lines or stair-stepping on curved surfaces, as left by building the part layer by layer? Answer yes or no.",
    "sig_extruded_bead_texture": "Do you see the parallel raster lines or bead paths of extruded filament on a flat top or bottom surface (typical of FDM/FFF 3D printing)? Answer yes or no.",
    "sig_powdery_grainy_surface": "Is the surface uniformly matte and slightly grainy, like fine sandpaper, typical of laser-sintered nylon powder parts? Answer yes or no.",
    "sig_support_nubs": "Do you see small regularly spaced bumps or scars where 3D-printing support structures were broken off? Answer yes or no.",
    "sig_translucent_resin": "Does the part look like a smooth, glossy, translucent or amber/grey photopolymer resin print? Answer yes or no.",
    "sig_paper_wood_laminate": "Does the part look like stacked sheets of paper or thin wood-like laminations bonded together, with layer edges visible on the sides? Answer yes or no.",
    "sig_thinned_corners_sheet": "Does the part look formed from one heated plastic sheet: a uniform shell whose corners and deep areas look stretched thinner, often with trimmed sheet edges? Answer yes or no.",
    "sig_trimmed_sheet_edge": "Are the outer edges straight cut or routed edges of constant thickness, as if trimmed from a larger sheet or laminate? Answer yes or no.",
    "sig_one_sided_finish": "Is one face smooth or glossy (the tool side) while the opposite face is rough, uneven or shows fabric texture? Answer yes or no.",
    "sig_both_sides_tooled": "Are both faces smooth and precisely shaped, with molded features on both sides, as produced between two rigid mold halves? Answer yes or no.",
    "sig_two_materials_bonded": "Is the part made of two clearly different materials molded together, such as a soft rubbery grip or seal on a hard plastic body? Answer yes or no.",
    "sig_sink_marks": "Do you see shallow dents or depressions on a visible face directly opposite thick features such as ribs or bosses (sink marks)? Answer yes or no.",
    "sig_knit_lines": "Do you see faint hairline lines on the surface where two flow fronts of molten plastic met (weld or knit lines), often near holes? Answer yes or no.",
    "sig_molded_bosses_clips": "Does the part have molded-in screw bosses, snap-fit clips, latches or living hinges? Answer yes or no.",
    "sig_flash": "Is there thin excess material (flash) along edges or the parting line, or visible trimming of such flash? Answer yes or no.",
    "sig_class_a_painted_panel": "Is this a large painted exterior body panel with a smooth automotive Class-A finish (like a hood, fender or tailgate skin)? Answer yes or no.",
    "sig_sandwich_core_visible": "At any cut edge or opening, can you see a foam or honeycomb core sandwiched between two thin skins? Answer yes or no.",
    "sig_gelcoat_finish": "Does the part have a thick, glossy, pigmented gelcoat surface typical of fiberglass boats, fairings or body kits? Answer yes or no.",
    "sig_metal_inserts": "Does the part contain molded-in metal inserts such as threaded brass bushings or metal reinforcements? Answer yes or no.",
}
Y = lambda f: [C(f, "eq", True)]
NONMOLD = [HLU, VARTM, FW, PUL, FFF, SLA, SLS, LOM]

favor(G, IM, Y("sig_ejector_pin_marks"), 2.0, "ejector pins are the IM release mechanism", "ejection")
favor(G, OM, Y("sig_ejector_pin_marks"), 1.0, "overmolding is injection-based", "ejection")
favor(G, [BMC, SRIM], Y("sig_ejector_pin_marks"), 0.5, "injection-fed thermoset molding uses ejectors", "ejection")
disfavor(G, NONMOLD + [TF], Y("sig_ejector_pin_marks"), 1.0, "no ejector system in these processes", "ejection")

favor(G, IM, Y("sig_gate_vestige"), 1.5, "gate vestige marks the injection point", "gating")
favor(G, [OM, BMC, SRIM], Y("sig_gate_vestige"), 0.75, "injected or transfer-fed molds leave gates", "gating")
disfavor(G, NONMOLD + [TF, BM], Y("sig_gate_vestige"), 0.75, "no gate in these processes", "gating")

favor(G, [IM, BM], Y("sig_parting_line"), 0.75, "two-half molds leave a parting line", "mold halves")
favor(G, [CM, SMC, BMC, OM, RTM, SRIM], Y("sig_parting_line"), 0.5, "matched dies leave a parting line", "mold halves")
disfavor(G, [FW, PUL, HLU, VARTM] + ADDITIVE, Y("sig_parting_line"), 0.5, "single-sided, kinematic or layerwise: no parting line", "mold halves")

favor(G, BM, Y("sig_pinch_off_seam"), 2.5, "the pinch-off is the blow-molding fingerprint", "pinch-off")
disfavor(G, [IM, CM, SMC, BMC, RTM, VARTM, HLU, TF] + ADDITIVE, Y("sig_pinch_off_seam"), 1.0, "no parison pinch-off in these processes", "pinch-off")

favor(G, BM, Y("sig_neck_or_spout"), 1.5, "necked openings into a hollow body are blown", "container necks")
favor(G, FW, Y("sig_neck_or_spout"), 0.25, "wound vessels have polar bosses", "polar openings")

favor(G, [HLU, VARTM, RTM], Y("sig_woven_fabric_visible"), 1.5, "woven fabric is laid up, infused or injected", "fabric reinforcement")
favor(G, CM, Y("sig_woven_fabric_visible"), 1.0, "prepreg fabric compression molding", "prepreg molding")
favor(G, SRIM, Y("sig_woven_fabric_visible"), 0.5, "SRIM preforms can be fabric", "SRIM preforms")
disfavor(G, [IM, OM, BM, BMC, SMC], Y("sig_woven_fabric_visible"), 1.5, "melt and compound processes cannot carry woven fabric", "reinforcement form")
disfavor(G, [TF], Y("sig_woven_fabric_visible"), 0.75, "only organosheet forming shows fabric", "organosheets")
disfavor(G, ADDITIVE, Y("sig_woven_fabric_visible"), 1.5, "printed feedstock has no woven fabric", "feedstock")

favor(G, [SMC, BMC, CM], Y("sig_random_chopped_fiber"), 1.0, "chopped-fiber compounds", "chopped fiber")
favor(G, [SRIM, HLU], Y("sig_random_chopped_fiber"), 0.5, "chopped-strand mat in lay-up and SRIM", "chopped-strand mat")
disfavor(G, ADDITIVE + [FW, PUL], Y("sig_random_chopped_fiber"), 1.0, "no chopped mat in these processes", "reinforcement form")

favor(G, FW, Y("sig_helical_winding_pattern"), 3.0, "helical criss-cross bands are the winding fingerprint", "winding pattern")
disfavor(G, [p for p in PROCS if p != FW], Y("sig_helical_winding_pattern"), 1.0, "no wound tows in other processes", "winding pattern")

favor(G, PUL, Y("sig_longitudinal_fiber_streaks"), 2.0, "pulled rovings leave longitudinal streaks", "roving alignment")
favor(G, FW, Y("sig_longitudinal_fiber_streaks"), 0.25, "low-angle winding can look longitudinal", "winding angle")
disfavor(G, [IM, OM, BM, TF] + ADDITIVE[:3], Y("sig_longitudinal_fiber_streaks"), 1.0, "no aligned rovings", "reinforcement form")

favor(G, FFF, Y("sig_layer_lines"), 2.0, "extrusion layers are visible", "layerwise build")
favor(G, LOM, Y("sig_layer_lines"), 1.0, "laminated sheets show layers", "sheet lamination")
favor(G, [SLA, SLS], Y("sig_layer_lines"), 0.75, "fine layers, stair-stepping on curves", "layerwise build")
disfavor(G, MATCHED_DIE + [BM, TF], Y("sig_layer_lines"), 1.0, "molded surfaces copy the tool, not layers", "molded surfaces")

favor(G, FFF, Y("sig_extruded_bead_texture"), 2.5, "raster bead paths are the FFF fingerprint", "toolpath texture")
disfavor(G, [SLA, SLS, LOM] + MATCHED_DIE + [BM, TF], Y("sig_extruded_bead_texture"), 1.0, "no extruded beads", "toolpath texture")

favor(G, SLS, Y("sig_powdery_grainy_surface"), 2.0, "sintered powder leaves a grainy matte skin", "powder-bed surface")
disfavor(G, [SLA, IM, OM, BM], Y("sig_powdery_grainy_surface"), 0.75, "smooth tool or resin surfaces", "surface finish")

favor(G, SLA, Y("sig_support_nubs"), 2.0, "vat-photopolymer supports leave nubs", "support removal")
favor(G, FFF, Y("sig_support_nubs"), 0.75, "FFF supports leave scars", "support removal")
disfavor(G, [SLS] + MATCHED_DIE + [BM, TF], Y("sig_support_nubs"), 0.75, "no support structures", "support removal")

favor(G, SLA, Y("sig_translucent_resin"), 1.0, "photopolymer resin look", "resin appearance")
favor(G, IM, Y("sig_translucent_resin"), 0.25, "clear molded thermoplastics exist", "clear moldings")

favor(G, LOM, Y("sig_paper_wood_laminate"), 3.0, "bonded paper sheets are the LOM fingerprint", "sheet lamination")
disfavor(G, [p for p in PROCS if p != LOM], Y("sig_paper_wood_laminate"), 1.0, "no paper laminations", "sheet lamination")

favor(G, TF, Y("sig_thinned_corners_sheet"), 2.0, "stretched sheet thins at corners and draws", "material distribution")
disfavor(G, [IM, CM, SMC, BMC], Y("sig_thinned_corners_sheet"), 0.75, "flow-filled cavities keep nominal wall", "material distribution")

favor(G, TF, Y("sig_trimmed_sheet_edge"), 1.5, "formed sheet is trimmed from the web", "trimming")
favor(G, [HLU, VARTM], Y("sig_trimmed_sheet_edge"), 0.75, "laminates are edge-trimmed", "edge trimming")
favor(G, PUL, Y("sig_trimmed_sheet_edge"), 0.5, "profiles are cut to length", "cut-to-length")

favor(G, [HLU, VARTM], Y("sig_one_sided_finish"), 1.5, "only the tool side is finished in open molding", "single-sided tooling")
favor(G, TF, Y("sig_one_sided_finish"), 1.0, "one face contacts the form", "single-sided tooling")
disfavor(G, [IM, CM, SMC, BMC, RTM], Y("sig_one_sided_finish"), 1.0, "matched dies finish both faces", "matched tooling")

favor(G, [IM, CM, SMC, BMC, RTM], Y("sig_both_sides_tooled"), 0.75, "both faces molded against rigid tools", "matched tooling")
favor(G, [OM, SRIM], Y("sig_both_sides_tooled"), 0.5, "matched tooling", "matched tooling")
disfavor(G, [HLU, VARTM, TF], Y("sig_both_sides_tooled"), 1.0, "single-sided processes cannot finish both faces", "single-sided tooling")

favor(G, OM, Y("sig_two_materials_bonded"), 2.5, "two materials molded together is overmolding", "multi-material molding")
favor(G, IM, Y("sig_two_materials_bonded"), 0.25, "insert/two-shot are IM variants", "multi-shot IM")
disfavor(G, [p for p in PROCS if p not in (OM, IM)], Y("sig_two_materials_bonded"), 0.5, "single-material processes", "multi-material molding")

favor(G, IM, Y("sig_sink_marks"), 1.5, "sink opposite ribs is a classic IM defect", "sink marks")
favor(G, [BMC, OM], Y("sig_sink_marks"), 0.5, "flow-molded thick features sink", "sink marks")

favor(G, IM, Y("sig_knit_lines"), 1.5, "knit lines form where melt fronts meet", "weld lines")
favor(G, [OM, BMC], Y("sig_knit_lines"), 0.5, "flow-front meeting in injected compounds", "weld lines")

favor(G, IM, Y("sig_molded_bosses_clips"), 1.5, "bosses, clips and hinges are molded-in IM features", "integrated features")
favor(G, [OM, BMC], Y("sig_molded_bosses_clips"), 0.5, "molded-in features", "integrated features")
disfavor(G, [HLU, VARTM, FW, PUL, TF], Y("sig_molded_bosses_clips"), 1.0, "these processes cannot mold clips or bosses", "integrated features")

favor(G, CM, Y("sig_flash"), 1.0, "compression flash along the shear edge", "flash")
favor(G, [SMC, BMC], Y("sig_flash"), 0.75, "compound flash and deflashing", "flash")
favor(G, [SRIM, IM], Y("sig_flash"), 0.25, "flash at parting line", "flash")

favor(G, SMC, Y("sig_class_a_painted_panel"), 1.0, "Class-A SMC body panels", "exterior panels")
favor(G, [IM, RTM, CM], Y("sig_class_a_painted_panel"), 0.5, "TPO fascias, RTM and pressed panels", "exterior panels")
disfavor(G, ADDITIVE, Y("sig_class_a_painted_panel"), 1.0, "not a production body-panel route", "exterior panels")

favor(G, [VARTM, HLU], Y("sig_sandwich_core_visible"), 1.0, "cored laminates are infused or laid up", "sandwich construction")
favor(G, [RTM, CM, SRIM], Y("sig_sandwich_core_visible"), 0.5, "cored RTM, pressed sandwich, foam-cored SRIM", "sandwich construction")
disfavor(G, [IM, BM, OM] + ADDITIVE, Y("sig_sandwich_core_visible"), 1.0, "no sandwich core", "sandwich construction")

favor(G, HLU, Y("sig_gelcoat_finish"), 1.5, "gelcoat is sprayed into open molds first", "gelcoat")
favor(G, VARTM, Y("sig_gelcoat_finish"), 1.0, "infused hulls use gelcoat", "gelcoat")
favor(G, RTM, Y("sig_gelcoat_finish"), 0.5, "in-mold gelcoat in RTM", "gelcoat")
disfavor(G, [IM, BM, TF] + ADDITIVE, Y("sig_gelcoat_finish"), 1.0, "no gelcoat", "gelcoat")

favor(G, [IM, OM], Y("sig_metal_inserts"), 0.75, "insert molding", "insert molding")
favor(G, [BMC, CM], Y("sig_metal_inserts"), 0.25, "molded-in inserts in thermoset compounds", "insert molding")

# ============================================================ write
features = dict(V2["features"])
for name, q in SIG.items():
    features[name] = {"type": "boolean", "question": q, "observability": "signature",
                      "status": "needs perception pass (not in the bench-v2 cache)"}
for name in V2["features"]:
    features[name] = dict(features[name], observability=OBS[name])

ids = [r["id"] for r in RULES]
assert len(ids) == len(set(ids))
for r in RULES:
    assert r["process"] in PROCS, r
    for c in r["when"]:
        assert c["feature"] in features, c
by_group = Counter(r["group"] for r in RULES)
by_kind = Counter(r["kind"] for r in RULES)
by_proc = Counter(r["process"] for r in RULES)
evaluable = sum(all(not c["feature"].startswith("sig_") for c in r["when"]) for r in RULES)
out = {
    "version": "v3-draft 2026-10-12",
    "note": ("Soft-evidence rule base. Superset of data/rules.json (frozen, 92 clauses, "
             "bench-v2). G1 = v2 limits as strong soft evidence; G2 typicality matrix; G3 "
             "combinations; G4 visible manufacturing signatures (28 new perception features, "
             "need one VLM pass); G5 fiber architecture. Every v3 rule is a draft to verify."),
    "processes": PROCS,
    "families": json.loads((MIP / "outputs" / "metrics_v3.json").read_text(encoding="utf-8"))["family_map"],
    "features": features,
    "summary": {"n_rules": len(RULES), "by_group": dict(by_group), "by_kind": dict(by_kind),
                "by_process": dict(by_proc), "evaluable_on_bench_v2_cache": evaluable,
                "need_new_perception_pass": len(RULES) - evaluable,
                "n_features": len(features), "new_signature_features": len(SIG)},
    "rules": RULES,
}
(MIP / "data" / "rules_v3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["summary"], indent=1))
