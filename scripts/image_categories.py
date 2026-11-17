"""Component categories for the scaled real-image benchmark.

Each category carries:
  * search_terms     -- queries used to fetch open-licensed photos from
                        Wikimedia Commons / Openverse (single-component images).
  * expert_processes -- the human/category-level ground-truth set of feasible
                        manufacturing processes (used for scoring). Derived
                        from standard polymer-composite practice and the
                        per-process LIT_REVIEW.md, NOT from any model.
  * qc_desc          -- what the photo must actually show, used verbatim by the
                        VLM screening pass (scripts/qc_images.py, prompt v2).
                        It locks the domain so lookalikes from the wrong domain
                        (aircraft panels, architectural gratings...) are
                        rejected.

Process-name spelling matches rules.json / KNOWN_PROCESSES exactly.
The VLM (Gemini) supplies the per-image FEATURES at run time; only the
process labels come from the category. This keeps scoring independent of the
model under test.

Categories added for goal 6 (complete 17-process coverage: every process in
>= 2 categories' expert sets) are marked "[VERIFY]" in the comment above them:
their expert sets are proposed from standard practice and must be checked by
Mathias before the labels count as expert-derived.
"""

CATEGORIES = {
    # --- injection-molded thermoplastic parts -----------------------------
    "intake_manifold": dict(
        search_terms=["car plastic intake manifold", "automobile inlet manifold plastic", "inlet manifold engine", "Ansaugkrümmer"],
        expert_processes={"Injection_Molding"},
        qc_desc="a plastic automotive engine intake manifold (molded runners/plenum, not a metal casting)"),
    "dashboard": dict(
        search_terms=["car dashboard", "automobile instrument panel"],
        expert_processes={"Injection_Molding"},
        qc_desc="an automotive car dashboard / instrument panel from a road-vehicle interior (not an aircraft or boat panel)"),
    "door_trim_panel": dict(
        search_terms=["car door inner trim panel", "automobile door card", "door panel interior vehicle", "Türverkleidung"],
        expert_processes={"Injection_Molding"},
        qc_desc="an automotive interior door trim panel / door card"),
    "headlamp_housing": dict(
        search_terms=["car headlight assembly", "automobile headlamp housing"],
        expert_processes={"Injection_Molding"},
        qc_desc="an automotive headlamp assembly or its plastic housing"),
    "instrument_cluster": dict(
        search_terms=["car instrument cluster", "automobile speedometer housing"],
        expert_processes={"Injection_Molding"},
        qc_desc="an automotive instrument cluster / gauge housing from a road vehicle"),
    "radiator_grille": dict(
        search_terms=["car radiator grille", "automobile front grille plastic"],
        expert_processes={"Injection_Molding"},
        qc_desc="an automotive plastic radiator grille (front grille of a car)"),
    "wheel_cover": dict(
        search_terms=["car wheel cover hubcap", "automobile hubcap plastic"],
        expert_processes={"Injection_Molding"},
        qc_desc="an automotive plastic wheel cover / hubcap"),
    "engine_cover": dict(
        search_terms=["car engine cover plastic", "automobile engine beauty cover", "engine cover", "Motorabdeckung"],
        expert_processes={"Injection_Molding", "Compression_Molding"},
        qc_desc="a plastic automotive engine (beauty) cover"),

    # --- blow-molded hollow parts -----------------------------------------
    "fuel_tank": dict(
        search_terms=["plastic automobile fuel tank", "car fuel tank polyethylene"],
        expert_processes={"Blow_Molding"},
        qc_desc="a plastic automotive fuel tank (hollow polyethylene tank, not a metal tank)"),
    "coolant_reservoir": dict(
        search_terms=["car coolant expansion tank", "automobile washer fluid reservoir", "expansion tank car", "coolant overflow bottle"],
        expert_processes={"Blow_Molding"},
        qc_desc="a plastic automotive coolant expansion tank or washer-fluid reservoir (hollow translucent tank)"),
    "air_intake_duct": dict(
        search_terms=["car air intake duct", "automobile air cleaner duct", "cold air intake", "air intake hose engine"],
        expert_processes={"Blow_Molding"},
        qc_desc="a plastic automotive air-intake duct or air-cleaner duct"),

    # --- compression / SMC / RTM body panels ------------------------------
    "hood": dict(
        search_terms=["car hood bonnet", "composite car hood"],
        expert_processes={"Resin_Transfer_Molding", "Compression_Molding",
                          "Sheet_Molding_Compound"},
        qc_desc="an automotive hood / bonnet panel shown as a part (a detached or clearly featured panel, not just a whole car)"),
    "fender": dict(
        search_terms=["car fender panel", "automobile wing panel"],
        expert_processes={"Sheet_Molding_Compound", "Compression_Molding",
                          "Resin_Transfer_Molding"},
        qc_desc="an automotive fender / wing body panel shown as a part"),
    "tailgate": dict(
        search_terms=["car tailgate liftgate", "automobile trunk lid"],
        expert_processes={"Sheet_Molding_Compound", "Compression_Molding",
                          "Resin_Transfer_Molding"},
        qc_desc="an automotive tailgate, liftgate, or trunk lid shown as a part"),
    "rear_spoiler": dict(
        search_terms=["car rear spoiler", "automobile deck spoiler"],
        expert_processes={"Injection_Molding", "Resin_Transfer_Molding",
                          "Compression_Molding"},
        qc_desc="an automotive rear spoiler / deck wing"),
    "battery_tray": dict(
        search_terms=["car battery tray", "ev battery enclosure composite", "car battery box", "EV battery pack enclosure"],
        expert_processes={"Compression_Molding", "Sheet_Molding_Compound",
                          "Resin_Transfer_Molding"},
        qc_desc="an automotive battery tray or EV battery enclosure (structural tray/box part)"),
    "skid_plate": dict(
        search_terms=["car skid plate underbody", "automobile underbody shield", "engine splash shield", "sump guard vehicle", "Unterfahrschutz"],
        expert_processes={"Compression_Molding", "Sheet_Molding_Compound"},
        qc_desc="an automotive underbody skid plate / shield panel"),

    # --- continuous / axisymmetric composite ------------------------------
    "leaf_spring": dict(
        search_terms=["composite leaf spring", "frp leaf spring vehicle"],
        expert_processes={"Resin_Transfer_Molding", "Pultrusion"},
        qc_desc="a fiber-composite (FRP) vehicle leaf spring (not a steel leaf spring)"),
    "drive_shaft": dict(
        search_terms=["composite drive shaft", "carbon fiber propeller shaft car"],
        expert_processes={"Filament_Winding", "Pultrusion"},
        qc_desc="a fiber-composite / carbon-fiber vehicle drive shaft (tubular shaft, not a steel shaft)"),
    "pressure_vessel": dict(
        search_terms=["cng composite pressure vessel", "carbon fiber pressure tank vehicle", "composite gas cylinder", "SCBA cylinder", "type 4 hydrogen tank"],
        expert_processes={"Filament_Winding"},
        qc_desc="a composite-overwrapped pressure vessel / CNG or hydrogen tank (cylindrical wound tank)"),
    "structural_profile": dict(
        search_terms=["pultruded fiberglass profile", "frp structural profile beam", "fiberglass profile", "GRP profile", "glass fibre pultrusion"],
        expert_processes={"Pultrusion"},
        qc_desc="a pultruded fiberglass (FRP) structural profile: constant-cross-section beam, channel, angle, or grating bar"),

    # --- large open-mold / liquid-molding composite -----------------------
    "monocoque_tub": dict(
        search_terms=["carbon fiber monocoque chassis", "composite car tub", "carbon fibre monocoque", "racing car chassis carbon"],
        expert_processes={"Hand_Lay_up", "Resin_Transfer_Molding",
                          "Vacuum_Assisted_RTM"},
        qc_desc="a carbon-fiber / composite car monocoque chassis tub"),
    "body_kit_fairing": dict(
        search_terms=["car body kit fairing", "fiberglass car fairing"],
        expert_processes={"Hand_Lay_up", "Resin_Transfer_Molding",
                          "Compression_Molding"},
        qc_desc="a fiberglass/composite vehicle body-kit part or fairing (bumper skirt, wide-body panel, motorcycle fairing)"),
    "truck_bed_liner": dict(
        search_terms=["pickup truck bed liner", "truck bed liner plastic", "pickup bed liner", "ute tray liner"],
        expert_processes={"Thermoforming", "Resin_Transfer_Molding",
                          "Sheet_Molding_Compound"},
        qc_desc="a drop-in plastic pickup-truck bed liner (large formed tray)"),

    # --- thermoformed thin parts ------------------------------------------
    "headliner": dict(
        search_terms=["car headliner roof trim", "automobile interior roof liner", "car roof lining", "vehicle headlining interior"],
        expert_processes={"Thermoforming", "Injection_Molding"},
        qc_desc="an automotive roof headliner / interior roof liner panel"),
    "interior_tray": dict(
        search_terms=["car interior storage tray", "automobile cup holder tray", "car trunk tray", "car boot liner tray", "center console tray"],
        expert_processes={"Thermoforming", "Injection_Molding"},
        qc_desc="an automotive interior storage tray, cup-holder insert, or trunk/cargo tray"),

    # --- seating / structural injection -----------------------------------
    "seat_shell": dict(
        search_terms=["car seat shell plastic", "automobile bucket seat frame", "racing bucket seat", "seat shell chair"],
        expert_processes={"Injection_Molding"},
        qc_desc="a plastic automotive seat shell / bucket-seat back or frame"),
    "steering_wheel": dict(
        search_terms=["car steering wheel", "automobile steering wheel"],
        expert_processes={"Injection_Molding"},
        qc_desc="an automotive steering wheel from a road vehicle (not a ship's wheel or aircraft yoke)"),
    "bumper_fascia": dict(
        search_terms=["car bumper cover fascia", "automobile front bumper plastic", "front bumper cover", "rear bumper plastic car"],
        expert_processes={"Injection_Molding"},
        qc_desc="a plastic automotive bumper cover / fascia shown as a part"),

    # --- additive-manufactured prototypes ---------------------------------
    "printed_bracket": dict(
        search_terms=["3d printed bracket", "fdm printed automotive bracket", "3D printed part", "FDM printed part"],
        expert_processes={"Fused_Filament_Fabrication", "Stereolithography",
                          "Selective_Laser_Sintering"},
        qc_desc="a 3D-printed engineering bracket or fixture (visible printed part, layer lines or powder texture)"),
    # [VERIFY] Laminated_Object_Manufacturing added to this prototype
    # category's expert set: LOM is a feasible route for blocky visual
    # prototypes of this kind.
    "printed_prototype": dict(
        search_terms=["3d printed prototype part", "sla resin printed part"],
        expert_processes={"Fused_Filament_Fabrication", "Stereolithography",
                          "Selective_Laser_Sintering",
                          "Laminated_Object_Manufacturing"},
        qc_desc="a 3D-printed prototype part (FDM/SLA/SLS printed object shown as a physical part, not a render or the printer itself)"),

    # ======================================================================
    # Goal-6 additions: complete coverage so every one of the 17 processes
    # appears in >= 2 categories. All expert sets below are [VERIFY].
    # ======================================================================

    # [VERIFY] Overmolding coverage 1/2: shift knobs are classically a rigid
    # injection-molded core overmolded with soft-touch elastomer.
    "gear_shift_knob": dict(
        search_terms=["car gear shift knob", "automobile gear stick knob", "gear stick", "shift lever knob"],
        expert_processes={"Overmolding", "Injection_Molding"},
        qc_desc="an automotive gear shift knob / gear stick handle"),
    # [VERIFY] Overmolding coverage 2/2 (goals list 'overmolded grips').
    "overmolded_tool_grip": dict(
        search_terms=["overmolded tool handle grip", "screwdriver rubber grip handle", "screwdriver handle", "pliers handle grip"],
        expert_processes={"Overmolding", "Injection_Molding"},
        qc_desc="a hand-tool handle or grip with a soft rubber/elastomer layer overmolded onto a rigid plastic core"),
    # [VERIFY] BMC coverage 1/2: headlamp reflectors are the classic BMC part.
    "headlamp_reflector": dict(
        search_terms=["car headlamp reflector", "headlight reflector housing", "headlight reflector", "lamp reflector parabolic"],
        expert_processes={"Bulk_Molding_Compound", "Injection_Molding"},
        qc_desc="an automotive headlamp reflector (the metallized parabolic reflector part of a headlight)"),
    # [VERIFY] BMC coverage 2/2 (goals list 'BMC housings').
    "electrical_junction_box": dict(
        search_terms=["fiberglass electrical junction box", "polyester electrical enclosure box", "junction box", "electrical distribution box", "polyester enclosure"],
        expert_processes={"Bulk_Molding_Compound", "Injection_Molding",
                          "Sheet_Molding_Compound"},
        qc_desc="a molded fiberglass/polyester electrical junction box or enclosure (a housing part, not a metal cabinet)"),
    # [VERIFY] SRIM coverage 1/2: bumper beams / structural carriers are
    # documented SRIM applications (glass preform + PU resin). Adversarial
    # review added Pultrusion (Shape Corp curved-pultruded C8 Corvette beam)
    # and Injection_Molding (LFT-PP beams).
    "bumper_beam": dict(
        search_terms=["car bumper reinforcement beam composite", "automobile bumper beam", "bumper reinforcement bar", "crash beam car"],
        expert_processes={"Structural_Reaction_Injection_Molding",
                          "Compression_Molding", "Sheet_Molding_Compound",
                          "Pultrusion", "Injection_Molding"},
        qc_desc="an automotive bumper reinforcement beam (the structural cross-member behind the bumper cover)"),
    # [VERIFY] SRIM coverage 2/2: load floors / cargo floors and parcel
    # shelves are documented SRIM applications. Adversarial review added
    # Thermoforming (LWRT / Azdel SuperLite load floors are thermoformed).
    "load_floor": dict(
        search_terms=["car trunk load floor panel", "vehicle composite cargo floor", "trunk floor panel car", "parcel shelf car", "boot floor"],
        expert_processes={"Structural_Reaction_Injection_Molding",
                          "Compression_Molding", "Sheet_Molding_Compound",
                          "Thermoforming"},
        qc_desc="an automotive trunk load floor / cargo floor or parcel-shelf panel shown as a part"),
    # [VERIFY] VARTM coverage (goals list 'hand-layup hulls'): boat hulls are
    # the canonical open-mold laminate, infused or laid up by hand.
    # NOTE (adversarial review): Spray_Up is the most common production hull
    # route but is not in this benchmark's 17-process universe, so it cannot
    # appear in the label set.
    "boat_hull": dict(
        search_terms=["fiberglass boat hull", "grp boat hull manufacturing", "boat hull repair", "fiberglass boat construction", "GRP boat"],
        expert_processes={"Hand_Lay_up", "Vacuum_Assisted_RTM",
                          "Resin_Transfer_Molding"},
        qc_desc="a fiberglass (GRP) boat hull shown as a hull (in a mold, under repair, or out of the water so the laminate hull itself is the subject)"),
    # [VERIFY] VARTM coverage: large commercial-vehicle panels (truck cab
    # fairings, bus body panels) are documented infusion/hand-layup parts.
    # Adversarial review added SMC + Compression_Molding (heavy-truck hoods
    # and fairings are classically compression-molded SMC).
    "truck_aero_fairing": dict(
        search_terms=["truck cab fairing composite", "bus fiberglass body panel", "truck roof fairing", "semi truck fairing", "bus body panel"],
        expert_processes={"Vacuum_Assisted_RTM", "Hand_Lay_up",
                          "Resin_Transfer_Molding", "Sheet_Molding_Compound",
                          "Compression_Molding"},
        qc_desc="a large fiberglass/composite truck or bus body panel or cab aero fairing shown as a part"),
    # [VERIFY] LOM coverage: dedicated category; images are scarce, which the
    # coverage report surfaces honestly.
    "lom_prototype": dict(
        search_terms=["laminated object manufacturing part", "LOM 3d printed paper part", "laminated object manufacturing", "paper 3d printing"],
        expert_processes={"Laminated_Object_Manufacturing"},
        qc_desc="a laminated-object-manufacturing (LOM) part: a prototype built from laminated paper/sheet layers, wood-like stepped appearance"),
    # [VERIFY] Filament winding + pultrusion coverage (goals list
    # 'filament-wound ... tubes').
    "composite_tube": dict(
        search_terms=["carbon fiber tube", "filament wound composite tube"],
        expert_processes={"Filament_Winding", "Pultrusion"},
        qc_desc="a hollow fiber-composite tube (carbon/glass fiber tube stock or tubular part with visible fiber weave/winding)"),
}


def all_processes_used():
    s = set()
    for c in CATEGORIES.values():
        s |= c["expert_processes"]
    return sorted(s)
