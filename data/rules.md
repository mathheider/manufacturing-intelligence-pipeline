# MIP rule table (generated from scripts/build_data.py)

17 processes, 11 features, 92 cited forbidden_if clauses.


## Injection_Molding

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | gt | 8.0 | Malloy 1994 p.151; Rosato 2000 ch.3 | sink/void regime above ~8 mm |
| `wall_thickness_mm` | lt | 0.3 | Kutz 2011 ch.11 | below conventional IM fill capability |
| `draft_angle_deg` | lt | 0.5 | Malloy 1994 p.205 | matched-tool ejection minimum draft |
| `projected_area_cm2` | gt | 50000 | Rosato & Rosato 2000 ch.4 | ~4000-ton clamp / platen ceiling |
| `continuous_fiber_required` | eq | True | [VERIFY: Mallick 2007 ch.2; Rosato 2000 ch.1] | melt injection carries only short/long chopped fiber; cannot place continuous reinforcement |
| `max_dimension_mm` | gt | 4000 | [VERIFY: Rosato & Rosato 2000 ch.4; Haitian JU 8800t specs] | beyond the largest commercial platen/shot envelope (~3.5 m tie-bar daylight; 3.4 m hulls are molded) |

## Thermoforming

| feature | op | value | source | reason |
|---|---|---|---|---|
| `hollow` | eq | True | Throne 1996 ch.1 pp.4-6; Throne 2008 ch.1 | single-face draping cannot close a cavity |
| `wall_thickness_mm` | gt | 8.0 | Throne 1996 ch.4 pp.150-160 | through-thickness heating / sag limit |
| `wall_thickness_mm` | lt | 0.2 | Throne 2008 ch.1 | below commercial sheet gauge |
| `rib_count` | gt | 3 | Throne 1996 ch.8 | single-face draping cannot form transverse ribs |
| `undercut_present` | eq | True | Throne 1996 ch.6 | single-face tool cannot release undercuts without side-action |
| `max_dimension_mm` | gt | 6000 | [VERIFY: Throne 2008 ch.9; MAAC/Geiss large-format specs] | beyond the largest commercial sheet/oven envelope (~3.8 x 5.9 m formers; 5.8 m spa shells exist) |

## Compression_Molding

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 0.8 | Mallick 2008 p.376 | charge cannot fill/knit below ~0.8 mm |
| `wall_thickness_mm` | gt | 30.0 | Campbell 2004 p.367 | exotherm voids; schema ceiling |
| `projected_area_cm2` | gt | 30000 | Campbell 2004 p.366 | ~4000-ton commercial press ceiling |
| `hollow` | eq | True | Mallick 2008 p.376; Campbell 2004 p.367 | matched-die closed cavity requires core |
| `undercut_present` | eq | True | Strong 2008 p.331 | matched steel dies cannot release undercuts |
| `draft_angle_deg` | lt | 0.5 | Strong 2008 p.331 | matched-die release minimum |
| `max_dimension_mm` | gt | 4000 | [VERIFY: Campbell 2004 p.366] | press platen envelope ceiling |

## Resin_Transfer_Molding

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 1.0 | Potter 1997 p.47; Advani & Sozer 2010 sec.5.4 | preform fill floor |
| `wall_thickness_mm` | gt | 25.0 | Campbell 2004 p.158; Mallick 2007 p.383 | exotherm in thick sections |
| `draft_angle_deg` | lt | 0.5 | Strong 2008 p.298 | matched steel-tool release requirement |
| `hollow` | eq | True | Potter 1997 ch.6; Campbell 2004 p.160 | no closed cavity without core |
| `projected_area_cm2` | gt | 100000 | Mallick 2007 pp.381-384 | practical RTM tool/clamp ceiling |

## Vacuum_Assisted_RTM

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 1.5 | Williams et al. 1996 Composites A 27(7); Campbell 2004 p.165 | bag-side tolerance floor |
| `undercut_present` | eq | True | Strong 2008 p.303; Campbell 2004 p.165 | single rigid face cannot release undercuts |
| `hollow` | eq | True | Campbell 2004 p.166 | no closed cavity without mandrel |
| `production_volume` | eq | 2 | [VERIFY: Strong 2008 p.303] | manual bagging + hours-scale infusion per part; mass production infeasible |

## Structural_Reaction_Injection_Molding

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 2.0 | Macosko 1989 pp.205-207; Mallick 2007 p.385 | gel-before-fill in thin cavities |
| `wall_thickness_mm` | gt | 15.0 | Macosko 1989 p.210; Campbell 2004 p.169 | runaway exotherm in thick sections |
| `draft_angle_deg` | lt | 0.5 | Strong 2008 p.312 | high-rate matched-tool release |
| `hollow` | eq | True | Campbell 2004 p.169 | no closed cavity in a single shot |
| `projected_area_cm2` | gt | 30000 | Campbell 2004 p.168 | SRIM press / flow-path ceiling |
| `max_dimension_mm` | gt | 3500 | [VERIFY: Campbell 2004 p.168] | press envelope consistent with the projected-area ceiling |

## Sheet_Molding_Compound

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 1.2 | Mallick 2008 p.371 | 25-mm chopped-fiber flow limit |
| `wall_thickness_mm` | gt | 10.0 | Orgeas & Dumont 2012 Composites A 43(4) p.630 | exotherm / segregation |
| `projected_area_cm2` | lt | 100 | Strong 2008 p.334 | SMC uneconomic below panel scale |
| `projected_area_cm2` | gt | 35000 | Campbell 2004 p.369 | ~4500-ton press ceiling |
| `hollow` | eq | True | Strong 2008 p.334; Campbell 2004 p.369 | matched-die closed cavity requires core |
| `undercut_present` | eq | True | Strong 2008 p.334 | matched steel dies cannot release undercuts |
| `max_dimension_mm` | gt | 4000 | [VERIFY: Campbell 2004 p.369] | press platen ceiling |

## Bulk_Molding_Compound

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 0.6 | Rosato 2004 p.420 | viscosity / filler-loading floor |
| `wall_thickness_mm` | gt | 12.0 | Mallick 2008 p.376 | exotherm porosity |
| `projected_area_cm2` | gt | 3000 | Strong 2008 p.335 | SMC preferred above panel scale |
| `hollow` | eq | True | Rosato 2004 p.420; Mallick 2008 p.376 | matched-die closed cavity requires core |
| `continuous_fiber_required` | eq | True | [VERIFY: Rosato 2004 ch.11] | bulk compound is chopped-fiber dough by definition; cannot deliver continuous reinforcement |
| `max_dimension_mm` | gt | 2000 | [VERIFY: Rosato 2004 ch.11] | small-part envelope consistent with the 3000 cm^2 area ceiling |

## Hand_Lay_up

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 1.5 | Strong 2008 p.246; Gurit Guide 2020 p.27 | manual consolidation scatter |
| `undercut_present` | eq | True | Strong 2008 p.247; Campbell 2004 p.119 | open mold cannot release undercuts |
| `hollow` | eq | True | Campbell 2004 p.120; Mallick 2007 p.375 | no closed cavity from a single open mold |
| `rib_count` | gt | 2 | Strong 2008 p.247 | ribs require secondary moldings beyond simple stiffeners |
| `production_volume` | eq | 2 | [VERIFY: Strong 2008 p.247; Mallick 2007 p.375] | hours of serial manual labor per part; mass production infeasible |

## Blow_Molding

| feature | op | value | source | reason |
|---|---|---|---|---|
| `hollow` | eq | False | Rosato 2003 ch.1; Lee 2006 ch.1; Strong 2006 ch.12 | parison inflation is definitional (part must be hollow) |
| `wall_thickness_mm` | gt | 10.0 | Rosato 2003 ch.3 pp.60-72; Lee 2006 ch.2 | parison programming / single-side cooling |
| `wall_thickness_mm` | lt | 0.25 | Lee 2006 ch.2 pp.20-23 | blowout / pinhole regime |
| `rib_count` | gt | 0 | Lee 2006 ch.2; Rosato 2003 ch.2 | parison inflation cannot form transverse ribs |
| `undercut_present` | eq | True | Rosato 2003 ch.2 | split-mold-half release cannot clear undercuts |
| `continuous_fiber_required` | eq | True | [VERIFY: Rosato 2003 ch.1] | an inflated parison cannot carry continuous fiber reinforcement |
| `max_dimension_mm` | gt | 4000 | [VERIFY: Lee 2006 ch.2; accumulator-head kayak practice] | industrial accumulator-head envelope (~4 m canoes are serially blow-molded) |

## Pultrusion

| feature | op | value | source | reason |
|---|---|---|---|---|
| `constant_cross_section` | eq | False | Starr 2000 p.3 | constant cross-section is the defining kinematic |
| `undercut_present` | eq | True | Starr 2000 p.55; Campbell 2004 p.381 | cannot exit die |
| `wall_thickness_mm` | lt | 1.5 | Starr 2000 p.50 | impregnation / die-pressure floor |
| `wall_thickness_mm` | gt | 25.0 | Mallick 2008 p.395; Carlone 2006 Composites B 37(6) | exotherm cracking |
| `rib_count` | gt | 0 | Starr 2000 p.55 | transverse ribs cannot exit a straight-pull die |

## Filament_Winding

| feature | op | value | source | reason |
|---|---|---|---|---|
| `axisymmetric` | eq | False | Peters 2011 p.14 | surface-of-revolution kinematic |
| `hollow` | eq | False | Peters 2011 p.14; Campbell 2004 p.388 | mandrel-based shell requires a hollow form |
| `undercut_present` | eq | True | Campbell 2004 p.390 | blocks mandrel extraction |
| `rib_count` | gt | 0 | Mallick 2008 p.400 | tow laydown cannot form transverse ribs |
| `wall_thickness_mm` | lt | 0.5 | Peters 2011 p.240 | two-layer balanced minimum |

## Overmolding

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | gt | 6.0 | Goodship & Love 2002 ch.4; Malloy 1994 p.151 | thick skin over insulating substrate |
| `wall_thickness_mm` | lt | 0.3 | Goodship & Love 2002 ch.3; Kutz 2011 ch.12 | unreliable fill around cooler substrate |
| `hollow` | eq | True | Goodship & Love 2002 ch.1; Rosato 2000 ch.22 | substrate + skin topology, not a closed hollow |
| `projected_area_cm2` | gt | 20000 | Goodship & Love 2002 ch.2 | two-shot / insert machine envelope |
| `max_dimension_mm` | gt | 3000 | [VERIFY: Goodship & Love 2002 ch.2; large 2K press practice] | multi-component machine envelope (aligned with the injection-molding ceiling class) |

## Fused_Filament_Fabrication

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 0.8 | Gibson 2021 ch.6 p.174; Stratasys FDM Guide 2022 | 2-perimeter minimum with 0.4 mm nozzle |
| `projected_area_cm2` | gt | 10000 | Wohlers 2023 ch.4 | largest commercial FFF/BAAM envelopes |
| `max_dimension_mm` | gt | 2200 | [VERIFY: Tractus3D T3500 specifications] | largest filament-fed commercial envelope (~2.1 m); pellet-extrusion BAAM/LSAM is a distinct machine class |
| `production_volume` | eq | 2 | [VERIFY: Gibson 2021 ch.6] | hours-per-part deposition rate; serially infeasible at mass-production volume |

## Stereolithography

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 0.4 | Formlabs Design Guide 2023; Gibson 2021 ch.4 p.93 | photopolymer minimum wall |
| `projected_area_cm2` | gt | 15000 | Wohlers 2023 ch.4 | Mammoth-class SLA upper bound |
| `continuous_fiber_required` | eq | True | [VERIFY: Gibson 2021 ch.4] | vat photopolymer is neat resin; no continuous fiber reinforcement path |
| `max_dimension_mm` | gt | 2100 | [VERIFY: Materialise Mammoth SLA specifications] | largest commercial vat envelope (~2.1 m) |

## Selective_Laser_Sintering

| feature | op | value | source | reason |
|---|---|---|---|---|
| `wall_thickness_mm` | lt | 0.7 | EOS PA2200 datasheet 2021; Gibson 2021 ch.5 p.126 | powder-fusion minimum wall |
| `projected_area_cm2` | gt | 10000 | Wohlers 2023 ch.4 | commercial SLS bed ceilings |
| `continuous_fiber_required` | eq | True | [VERIFY: Gibson 2021 ch.5] | powder-bed fusion cannot lay continuous fiber |
| `max_dimension_mm` | gt | 1000 | [VERIFY: Farsoon HT1001P specifications] | largest commercial powder-bed envelope (1000 mm class) |

## Laminated_Object_Manufacturing

| feature | op | value | source | reason |
|---|---|---|---|---|
| `hollow` | eq | True | Gibson 2021 ch.9 pp.257-258; Chua et al. 2010 ch.4 | decube extraction infeasible in a closed cavity |
| `wall_thickness_mm` | lt | 1.0 | Gibson 2021 ch.9 p.260 | sheet delamination during decubing |
| `projected_area_cm2` | gt | 5000 | Wohlers 2023 ch.4; Helisys LOM-2030 upper bound | envelope ceiling |
| `rib_count` | gt | 2 | Gibson 2021 ch.9 p.260 | stepped z-axis rib profiles brittle across laminae |
| `max_dimension_mm` | gt | 820 | [VERIFY: Helisys LOM-2030H specifications (813 mm long axis)] | laminating envelope ceiling |
| `production_volume` | eq | 2 | [VERIFY: Gibson 2021 ch.9] | sheet-by-sheet lamination rate; mass production infeasible |
