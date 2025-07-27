"""
BCL Physics Training Battery Generator
Creates diverse physics explanation training pairs for each BCL module
"""

import json
import random

# Physics modules with their fundamental laws and reasoning patterns
PHYSICS_MODULES = {
    "thermal": {
        "laws": [
            ("heat_transfer_conduction", "Fourier's Law: q = k*A*ΔT/thickness", "Heat flows from hot to cold proportional to temperature difference"),
            ("heat_transfer_convection", "Newton's Cooling: q = h*A*ΔT", "Air movement increases heat transfer rate"),
            ("thermal_mass", "Q = m*c*ΔT", "Mass stores thermal energy, dampening temperature swings"),
            ("radiation_transfer", "Stefan-Boltzmann: q = σ*ε*A*(T1⁴-T2⁴)", "All surfaces radiate heat based on temperature"),
            ("thermal_expansion", "ΔL = α*L*ΔT", "Materials expand when heated, creating stress")
        ],
        "safety_principles": [
            "Prevent heat buildup that weakens materials",
            "Maintain comfortable temperatures for occupants", 
            "Prevent condensation and moisture damage",
            "Control fire spread through thermal barriers",
            "Ensure equipment doesn't overheat"
        ]
    },
    
    "structural": {
        "laws": [
            ("bending_stress", "σ = M*y/I", "Beams resist loads through internal stress distribution"),
            ("column_buckling", "Pcr = π²*E*I/(K*L)²", "Slender columns fail by buckling before material yield"),
            ("shear_stress", "τ = V*Q/(I*t)", "Shear forces create sliding tendencies in materials"),
            ("deflection", "δ = P*L³/(3*E*I)", "Stiffness prevents excessive movement"),
            ("fatigue", "N = (σ_endurance/σ_applied)^b", "Repeated loads cause failure below yield strength")
        ],
        "safety_principles": [
            "Prevent sudden catastrophic failure",
            "Limit deflections for occupant comfort",
            "Account for all load combinations",
            "Provide redundant load paths",
            "Consider long-term material degradation"
        ]
    },
    
    "flow": {
        "laws": [
            ("continuity", "Q = A*v", "Mass conservation: what flows in must flow out"),
            ("pressure_drop", "ΔP = f*L*ρ*v²/(2*D)", "Friction causes pressure loss in pipes"),
            ("bernoulli", "P + ½ρv² + ρgh = constant", "Energy conservation in flowing fluids"),
            ("stack_effect", "ΔP = ρ*g*h*ΔT/T", "Temperature differences drive natural ventilation"),
            ("fan_laws", "Q₂/Q₁ = (RPM₂/RPM₁)", "Flow varies directly with fan speed")
        ],
        "safety_principles": [
            "Ensure adequate ventilation for air quality",
            "Prevent dangerous pressure differentials",
            "Control smoke movement in fires",
            "Provide sufficient water pressure for firefighting",
            "Prevent backflow contamination"
        ]
    },
    
    "acoustics": {
        "laws": [
            ("mass_law", "TL = 20*log(f*m) - 48", "Heavier walls block more sound"),
            ("reverb_time", "RT60 = 0.161*V/A", "Room volume and absorption control echo"),
            ("sound_decay", "SPL = SPL_source - 20*log(distance)", "Sound intensity decreases with distance"),
            ("resonance", "f = v/(2*L)", "Cavities amplify specific frequencies"),
            ("absorption", "α = 1 - (reflected/incident)", "Soft materials absorb sound energy")
        ],
        "safety_principles": [
            "Protect hearing from excessive noise",
            "Ensure emergency announcements are intelligible",
            "Prevent noise-induced stress and fatigue",
            "Maintain speech privacy where required",
            "Control vibration transmission"
        ]
    },
    
    "electrical": {
        "laws": [
            ("ohms_law", "V = I*R", "Voltage drives current through resistance"),
            ("power_dissipation", "P = I²*R", "Current creates heat in conductors"),
            ("ampacity", "I = k*A^0.5", "Larger conductors carry more current safely"),
            ("voltage_drop", "Vd = I*R*L", "Long runs reduce available voltage"),
            ("fault_current", "Isc = V/(Z_source + Z_line)", "Short circuits create dangerous current spikes")
        ],
        "safety_principles": [
            "Prevent electrical fires from overheating",
            "Protect against electric shock",
            "Ensure reliable power for safety systems",
            "Provide proper grounding paths",
            "Enable quick fault isolation"
        ]
    },
    
    "human_factors": {
        "laws": [
            ("evacuation_flow", "Flow = k*width*density", "Doorway capacity limits evacuation speed"),
            ("visibility", "S = K/σ", "Smoke density reduces sight distance"),
            ("reaction_time", "t_total = t_detect + t_decide + t_act", "Human response has multiple components"),
            ("crowd_density", "v = v_max*(1 - ρ/ρ_max)", "Speed decreases with crowding"),
            ("panic_threshold", "P_panic = f(familiarity, lighting, noise)", "Environmental factors affect behavior")
        ],
        "safety_principles": [
            "Provide intuitive wayfinding under stress",
            "Size exits for worst-case occupancy",
            "Account for reduced mobility occupants",
            "Minimize decision points during evacuation",
            "Maintain visibility in smoke conditions"
        ]
    }
}

def generate_physics_explanations():
    """Generate diverse physics explanation training pairs"""
    training_data = []
    
    # Pattern 1: Explain specific BCL rule with physics
    for module, content in PHYSICS_MODULES.items():
        for law_name, formula, explanation in content["laws"]:
            for principle in random.sample(content["safety_principles"], 2):
                # Create a BCL rule that uses this physics
                rule_examples = generate_rule_for_physics(module, law_name, formula)
                
                for rule in rule_examples:
                    training_pair = {
                        "instruction": "Explain the physics and safety reasoning behind this BCL rule",
                        "input": rule,
                        "output": f"This rule applies {law_name} from {module} physics. {explanation}. The formula {formula} shows that {generate_physics_insight(law_name, module)}. This ensures safety by: {principle}. The specific values in the rule are chosen based on {generate_engineering_rationale(module)}."
                    }
                    training_data.append(training_pair)
    
    # Pattern 2: Identify which physics law applies
    for module, content in PHYSICS_MODULES.items():
        for law_name, formula, explanation in content["laws"]:
            constraint = generate_constraint_from_law(law_name, module)
            training_pair = {
                "instruction": "Which physics law governs this BCL constraint and why",
                "input": f"must: {constraint}",
                "output": f"This constraint is based on {law_name} ({formula}) from {module} physics. {explanation}. This physical law is critical here because {random.choice(content['safety_principles'])}."
            }
            training_data.append(training_pair)
    
    # Pattern 3: Derive safe values from physics
    for module, content in PHYSICS_MODULES.items():
        scenario = generate_safety_scenario(module)
        law_name, formula, explanation = random.choice(content["laws"])
        training_pair = {
            "instruction": "Using physics principles, determine safe values for this building scenario",
            "input": scenario,
            "output": f"Applying {law_name} ({formula}): {explanation}. For this scenario, {generate_calculation_narrative(module, law_name)}. Therefore, safe values would be: {generate_safe_values(module, law_name)}. This provides safety margin because {random.choice(content['safety_principles'])}."
        }
        training_data.append(training_pair)
    
    # Pattern 4: Multi-physics interactions
    for i in range(20):
        modules = random.sample(list(PHYSICS_MODULES.keys()), 2)
        interaction = generate_physics_interaction(modules[0], modules[1])
        training_pair = {
            "instruction": "Explain how multiple physics domains interact in this BCL rule",
            "input": interaction["rule"],
            "output": f"This rule involves both {modules[0]} and {modules[1]} physics. {interaction['explanation']} The interaction is critical because {generate_interaction_safety(modules[0], modules[1])}."
        }
        training_data.append(training_pair)
    
    return training_data

def generate_rule_for_physics(module, law_name, formula):
    """Generate BCL rules that demonstrate specific physics laws"""
    rules = []
    
    if module == "thermal" and law_name == "heat_transfer_conduction":
        rules.append("""rule exterior_wall_insulation:
    where: wall.type = 'exterior'
    must: thermal_resistance(wall) >= 20.hr_ft2_F_per_BTU
    # Based on conduction: lower k and higher thickness reduce heat flow""")
    
    elif module == "structural" and law_name == "column_buckling":
        rules.append("""rule slender_column_check:
    where: column.slenderness_ratio > 120
    must: applied_load(column) <= euler_buckling_load(column) * 0.6
    # Euler buckling with safety factor for long columns""")
    
    elif module == "flow" and law_name == "stack_effect":
        rules.append("""rule stairwell_pressurization:
    where: building.height > 75.feet
    must: pressure_differential(stairwell, floor) >= 0.05.inches_water
    # Stack effect creates negative pressure that must be overcome""")
    
    elif module == "acoustics" and law_name == "mass_law":
        rules.append("""rule party_wall_stc:
    where: wall.between_units = true
    must: sound_transmission_class(wall) >= 50
    # Mass law: doubling mass adds ~6 dB reduction""")
    
    elif module == "electrical" and law_name == "power_dissipation":
        rules.append("""rule conductor_derating:
    where: conductor.bundled_count > 3
    must: ampacity(conductor) *= 0.8
    # I²R heating increases with bundling, reducing heat dissipation""")
    
    elif module == "human_factors" and law_name == "evacuation_flow":
        rules.append("""rule exit_door_capacity:
    where: space.occupancy > 50
    must: total_exit_width >= occupancy * 0.2.inches_per_person
    # Flow rate through doors limits evacuation speed""")
    
    # Add more examples...
    return rules if rules else [f"rule {module}_{law_name}_example:\n    must: {law_name}_constraint >= safe_value"]

def generate_constraint_from_law(law_name, module):
    """Generate realistic constraints based on physics laws"""
    constraints = {
        "heat_transfer_conduction": "wall.u_value <= 0.05.BTU_per_hr_ft2_F",
        "column_buckling": "column.buckling_capacity >= 1.5 * design_load",
        "pressure_drop": "sprinkler_system.pressure_at_end >= 15.psi",
        "mass_law": "floor_ceiling_assembly.stc >= 45 + (floor_number > 3 ? 5 : 0)",
        "ohms_law": "voltage_drop(circuit) <= 0.03 * nominal_voltage",
        "evacuation_flow": "exit_capacity >= 0.005 * occupant_load / seconds"
    }
    return constraints.get(law_name, f"{module}.{law_name}_value >= required_minimum")

def generate_physics_insight(law_name, module):
    """Generate insights about what the physics formula tells us"""
    insights = {
        "heat_transfer_conduction": "heat flow is inversely proportional to thickness, so doubling insulation thickness halves heat loss",
        "column_buckling": "buckling load decreases with the square of length, making tall columns exponentially weaker",
        "pressure_drop": "pressure loss increases with the square of velocity, so oversized pipes save pump energy",
        "mass_law": "each doubling of wall mass increases sound reduction by 6 dB",
        "power_dissipation": "heat generation increases with the square of current, making overloading extremely dangerous",
        "evacuation_flow": "flow rate is linear with width but limited by human walking speed and spacing"
    }
    return insights.get(law_name, f"{law_name} creates a physical constraint on safe operation")

def generate_engineering_rationale(module):
    """Generate engineering rationale for specific values"""
    rationales = {
        "thermal": "industry standards, climate zones, and comfort studies showing 68-78°F optimal range",
        "structural": "material safety factors, statistical load variations, and historical failure analysis", 
        "flow": "ASHRAE ventilation standards, smoke control research, and pressure measurement capabilities",
        "acoustics": "psychoacoustic research on speech intelligibility and sleep disturbance thresholds",
        "electrical": "NEC ampacity tables, temperature rise tests, and fire incident statistics",
        "human_factors": "evacuation drills, crowd flow studies, and visibility research in smoke"
    }
    return rationales.get(module, "engineering standards and safety margin analysis")

def generate_safety_scenario(module):
    """Generate realistic building scenarios requiring physics analysis"""
    scenarios = {
        "thermal": "A server room generates 50kW of heat. The room is 20x30x10 feet with one exterior wall.",
        "structural": "A conference room has a 30-foot clear span with a movable partition system adding 15 psf load.",
        "flow": "A 20-story building needs stairwell pressurization to prevent smoke infiltration during fires.",
        "acoustics": "A residential unit is adjacent to a mechanical room with chillers operating at 85 dB.",
        "electrical": "An emergency lighting circuit runs 200 feet from the panel to the farthest fixture.",
        "human_factors": "A assembly space holds 500 people with three exits of varying widths."
    }
    return scenarios.get(module, f"A critical {module} system in a high-rise building")

def generate_calculation_narrative(module, law_name):
    """Generate narrative explanation of physics calculations"""
    narratives = {
        ("thermal", "heat_transfer_conduction"): "calculating heat flow: with outdoor temp -10°F and indoor 70°F, through a wall with R-20 insulation: q = ΔT/R = 80/20 = 4 BTU/hr·ft²",
        ("structural", "column_buckling"): "for a 20-foot steel column with I=100 in⁴, E=29,000 ksi: Pcr = π²EI/(KL)² = 9.87×29,000×100/(1×240)² = 496 kips",
        ("flow", "pressure_drop"): "in 2-inch pipe at 5 ft/s over 100 feet: ΔP = 0.02×100×0.5×25/2 = 25 psi pressure loss",
        ("acoustics", "mass_law"): "for a 4-inch concrete wall (50 lb/ft²) at 500 Hz: TL = 20×log(500×50) - 48 = 40 dB reduction",
        ("electrical", "voltage_drop"): "for 100A load on #2 AWG copper 150 feet: Vd = 100×0.2×150/1000 = 3V drop on 120V circuit (2.5%)",
        ("human_factors", "evacuation_flow"): "through a 44-inch door: flow = 24 persons/minute/unit width × 2 units = 48 persons/minute"
    }
    return narratives.get((module, law_name), f"applying {law_name} formula with typical values")

def generate_safe_values(module, law_name):
    """Generate specific safe values based on physics calculations"""
    values = {
        ("thermal", "heat_transfer_conduction"): "minimum R-30 insulation for roofs, R-20 for walls in cold climates",
        ("structural", "column_buckling"): "maximum slenderness ratio of 200 for steel, 75 for concrete",
        ("flow", "pressure_drop"): "maximum velocity 8 ft/s in pipes, 2000 fpm in ducts to limit noise",
        ("acoustics", "mass_law"): "minimum STC 50 between units, STC 60 for mechanical rooms",
        ("electrical", "voltage_drop"): "maximum 3% drop for branch circuits, 5% total to last outlet",
        ("human_factors", "evacuation_flow"): "minimum 0.2 inches exit width per person, maximum 75 feet to exit"
    }
    return values.get((module, law_name), f"values meeting {module} safety requirements with appropriate margin")

def generate_physics_interaction(module1, module2):
    """Generate rules showing interaction between physics domains"""
    interactions = {
        ("thermal", "structural"): {
            "rule": """rule thermal_expansion_joint:
    where: structure.length > 200.feet
    must: expansion_joint.spacing <= 150.feet
    must: joint.movement_capacity >= thermal_expansion(ΔT=100°F)""",
            "explanation": "Thermal expansion (ΔL = α×L×ΔT) creates structural stresses. Steel expands 0.0000065 per °F, so 200 feet × 100°F × 0.0000065 = 1.3 inches movement."
        },
        ("flow", "thermal"): {
            "rule": """rule kitchen_hood_makeup_air:
    where: kitchen.hood_exhaust > 400.cfm
    must: makeup_air.temperature >= 50°F
    must: makeup_air.volume >= 0.8 * exhaust_volume""",
            "explanation": "Exhaust flow creates negative pressure (continuity equation). Cold makeup air requires heating load Q = 1.08×CFM×ΔT."
        },
        ("acoustics", "structural"): {
            "rule": """rule floating_floor_impact:
    where: floor.use = 'multi_family'
    must: impact_insulation_class >= 50
    must: structural_deflection <= span/480""",
            "explanation": "Impact creates both structure-borne vibration and airborne sound. Stiff structures (low deflection) transmit more impact energy."
        }
    }
    
    key = (module1, module2) if (module1, module2) in interactions else (module2, module1)
    return interactions.get(key, {
        "rule": f"rule {module1}_{module2}_interaction:\n    must: consider_both({module1}, {module2})",
        "explanation": f"{module1} physics affects {module2} performance through coupled physical phenomena"
    })

def generate_interaction_safety(module1, module2):
    """Explain why physics interactions matter for safety"""
    safety_interactions = {
        ("thermal", "structural"): "thermal stress can cause cracking, buckling, or connection failure if not accommodated",
        ("flow", "thermal"): "inadequate ventilation causes overheating, while excess ventilation wastes heating/cooling energy",
        ("acoustics", "structural"): "structure-borne sound travels through the building frame, bypassing acoustic barriers",
        ("electrical", "thermal"): "electrical resistance creates heat that must be dissipated to prevent insulation damage",
        ("human_factors", "flow"): "smoke movement patterns must align with evacuation routes to maintain visibility",
        ("structural", "flow"): "structural vibrations can cause pipe fatigue and leaks at connections"
    }
    
    key = (module1, module2) if (module1, module2) in safety_interactions else (module2, module1)
    return safety_interactions.get(key, f"coupled {module1}-{module2} effects can create unexpected failure modes")

# Generate the training data
print("Generating BCL Physics Training Battery...")
physics_training_data = generate_physics_explanations()

# Add some high-level physics philosophy examples
philosophy_examples = [
    {
        "instruction": "Explain why BCL rules must be grounded in physics",
        "input": "Why can't building codes just use arbitrary safety factors?",
        "output": "Building codes must be grounded in physics because physical laws govern how buildings behave. Arbitrary safety factors without physics basis lead to either: 1) Overdesign - wasting materials and money, or 2) Underdesign - creating dangerous conditions. Physics provides the causal relationships: heat flows from hot to cold (thermodynamics), forces must balance (statics), fluids seek equilibrium (fluid mechanics). These laws are universal and predictable, allowing us to calculate exact requirements for safety. For example, fire resistance ratings come from heat transfer calculations, not guesswork. This physics foundation ensures codes protect lives efficiently."
    },
    {
        "instruction": "How do physics safety margins account for uncertainty",
        "input": "How do we set safety factors when real-world conditions vary?",
        "output": "Safety factors bridge the gap between idealized physics and messy reality. Physics gives us the theoretical minimum - column buckling at Pcr = π²EI/L². But real columns have imperfections, loads vary, and materials degrade. We handle this through: 1) Statistical analysis of variations (load factors like 1.2×Dead + 1.6×Live), 2) Material reliability factors (φ=0.9 for steel, 0.75 for concrete), 3) Consequence-based margins (higher factors for brittle failures), 4) Redundancy requirements (multiple load paths). The physics tells us HOW things fail; statistics tell us WHEN. Combined, they create robust safety margins that account for known unknowns without excessive waste."
    }
]

physics_training_data.extend(philosophy_examples)

# Save the training data
with open('bcl_physics_training_battery.jsonl', 'w') as f:
    for item in physics_training_data:
        f.write(json.dumps(item) + '\n')

print(f"Generated {len(physics_training_data)} physics training examples")
print("\nExample distribution:")
for module in PHYSICS_MODULES:
    count = sum(1 for item in physics_training_data if module in str(item))
    print(f"  {module}: ~{count} examples")

print("\nFirst 3 examples:")
for i, example in enumerate(physics_training_data[:3]):
    print(f"\n--- Example {i+1} ---")
    print(f"Instruction: {example['instruction']}")
    print(f"Input: {example['input'][:100]}...")
    print(f"Output: {example['output'][:150]}...")
