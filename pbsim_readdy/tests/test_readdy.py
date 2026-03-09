import math
import random

import numpy as np
from process_bigraph import ProcessTypes, Composite
from readdy import ReactionDiffusionSystem, Simulation
from simularium_readdy_models import ReaddyUtil
from simularium_readdy_models.actin import ActinSimulation

from pb_multiscale_actin.create_readdy_pbif import generate_readdy_pbg, register_items_into_core, get_default_config
from pb_multiscale_actin.library.utils import simulate_readdy, id_difference, get_monomers

expected_monomers = {'particles': {0: {'neighbor_ids': [1, 2], 'position': (-25.0, 1.617886445553052, 0.0), 'type_name': 'actin#pointed_fixed_ATP_1'}, 1: {'neighbor_ids': [0, 2, 3], 'position': (-22.196901237, -1.576092210430552, -0.365362689562598), 'type_name': 'actin#fixed_ATP_2'}, 2: {'neighbor_ids': [0, 1, 3, 4], 'position': (-19.393802473999997, 1.4528688137021482, 0.7118488328823619), 'type_name': 'actin#mid_fixed_ATP_3'}, 3: {'neighbor_ids': [1, 2, 4, 5], 'position': (-16.565576538463482, -1.3372094936829628, -1.114031626689281), 'type_name': 'actin#mid_ATP_4'}, 4: {'neighbor_ids': [2, 3, 5, 6], 'position': (-13.738821325968699, 1.0182086874814005, 1.3966644784722928), 'type_name': 'actin#mid_ATP_5'}, 5: {'neighbor_ids': [3, 4, 6, 7], 'position': (-10.91200067688146, -0.6440630350887194, -1.496799140798851), 'type_name': 'actin#mid_ATP_1'}, 6: {'neighbor_ids': [4, 5, 7, 8], 'position': (-8.154446931973464, 0.3773712693689378, 1.692653980436068), 'type_name': 'actin#mid_ATP_2'}, 7: {'neighbor_ids': [5, 6, 8, 9], 'position': (-5.419161813769452, 0.05908096890544945, -1.578867199328362), 'type_name': 'actin#mid_ATP_3'}, 8: {'neighbor_ids': [6, 7, 9, 10], 'position': (-2.5657159614349014, -0.5219405019031342, 1.655121915924541), 'type_name': 'actin#mid_ATP_4'}, 9: {'neighbor_ids': [7, 8, 10, 11], 'position': (0.23306114966888436, 0.7060911084293366, -1.3727211709455847), 'type_name': 'actin#mid_ATP_5'}, 10: {'neighbor_ids': [8, 9, 11, 12], 'position': (3.0116870970390055, -1.145825162673377, 1.3964754937790183), 'type_name': 'actin#mid_ATP_1'}, 11: {'neighbor_ids': [9, 10, 12, 13], 'position': (5.8065669911853215, 1.2227346522280191, -1.0864550028109574), 'type_name': 'actin#mid_ATP_2'}, 12: {'neighbor_ids': [10, 11, 13, 14], 'position': (8.550383442834187, -1.570044266422198, 0.6647797173226726), 'type_name': 'actin#mid_ATP_3'}, 13: {'neighbor_ids': [11, 12, 14, 15], 'position': (11.542297622578925, 1.5450478788934063, -0.2320128507932875), 'type_name': 'actin#mid_ATP_4'}, 14: {'neighbor_ids': [12, 13, 15, 16], 'position': (14.329602861300495, -1.77333752286588, 0.04925619947858807), 'type_name': 'actin#mid_ATP_5'}, 15: {'neighbor_ids': [13, 14, 16], 'position': (17.082513477236127, 1.5671076201558154, 0.6622087907921218), 'type_name': 'actin#barbed_ATP_1'}, 16: {'neighbor_ids': [14, 15], 'position': (19.844940145413545, -1.5181098121608179, -0.7625430560274378), 'type_name': 'binding_site#2'}, 17: {'neighbor_ids': [25, 18, 19], 'position': (27.718951839599, -49.882789094504226, -49.75510973152522), 'type_name': 'membrane#outer_edge_4_1'}, 18: {'neighbor_ids': [17, 19, 26, 20], 'position': (27.492890105285817, 0.17537174492761753, -50.13558282128007), 'type_name': 'membrane#outer_edge_1'}, 19: {'neighbor_ids': [17, 18, 27, 20, 22, 21], 'position': (28.063169362976208, -25.431179966399434, -6.058168455811198), 'type_name': 'membrane#outer'}, 20: {'neighbor_ids': [18, 19, 28, 22], 'position': (27.44801201296938, 38.74129013596217, -6.77574726275579), 'type_name': 'membrane#outer_edge_2'}, 21: {'neighbor_ids': [19, 29, 22, 23], 'position': (27.542427896106684, -49.92983418697352, 36.488346728918884), 'type_name': 'membrane#outer_edge_4'}, 22: {'neighbor_ids': [19, 20, 21, 23, 30, 24], 'position': (27.45349544717826, -0.11460435123082707, 36.71621803352907), 'type_name': 'membrane#outer'}, 23: {'neighbor_ids': [21, 22, 31, 24], 'position': (26.79560014787575, -24.475981177124236, 3.9458332512301784), 'type_name': 'membrane#outer_edge_3'}, 24: {'neighbor_ids': [22, 23, 32], 'position': (27.571927277012037, 38.685593038640086, 53.36513607404555), 'type_name': 'membrane#outer_edge_2_3'}, 25: {'neighbor_ids': [17, 26, 27], 'position': (22.54172476229952, -49.93150078563962, -49.79608630605943), 'type_name': 'membrane#inner_edge_4_1'}, 26: {'neighbor_ids': [18, 25, 27, 28], 'position': (22.45664209412317, 0.19684079428604412, -50.166749020673095), 'type_name': 'membrane#inner_edge_1'}, 27: {'neighbor_ids': [19, 25, 26, 28, 30, 29], 'position': (22.98913347096495, -24.343092459969608, -6.387067382355522), 'type_name': 'membrane#inner'}, 28: {'neighbor_ids': [20, 26, 27, 30], 'position': (22.28348995721144, 38.599444962356436, -6.880771637735726), 'type_name': 'membrane#inner_edge_2'}, 29: {'neighbor_ids': [21, 27, 30, 31], 'position': (22.38478399914333, -50.1053806274756, 36.64554952137954), 'type_name': 'membrane#inner_edge_4'}, 30: {'neighbor_ids': [22, 27, 28, 29, 31, 32], 'position': (22.33904774673846, -0.04872912630944603, 36.658772606269075), 'type_name': 'membrane#inner'}, 31: {'neighbor_ids': [23, 29, 30, 32], 'position': (21.887313587683636, -25.884280685404622, 4.5234437207781975), 'type_name': 'membrane#inner_edge_3'}, 32: {'neighbor_ids': [24, 30, 31], 'position': (22.44315551260011, 38.77548461110144, 53.26770591683964), 'type_name': 'membrane#inner_edge_2_3'}, 33: {'neighbor_ids': [], 'position': (12.719143674716248, 10.998089733455654, 4.988440999350409), 'type_name': 'actin#free_ATP'}, 34: {'neighbor_ids': [], 'position': (13.199570117383574, -3.579438100108558, 7.265204991286533), 'type_name': 'actin#free_ATP'}, 35: {'neighbor_ids': [], 'position': (10.770345426371064, 19.611627651207964, 23.212405503851453), 'type_name': 'actin#free_ATP'}, 36: {'neighbor_ids': [], 'position': (9.450125289550936, 14.53066921385757, 1.4432114433700802), 'type_name': 'actin#free_ATP'}, 37: {'neighbor_ids': [], 'position': (13.254500380633868, 21.02084679614323, -21.522773298626856), 'type_name': 'actin#free_ATP'}, 38: {'neighbor_ids': [], 'position': (3.6601468517895186, -23.70940606340732, 16.6215679400315), 'type_name': 'actin#free_ATP'}, 39: {'neighbor_ids': [], 'position': (17.317869647619958, 18.67223379431631, 23.84072174659442), 'type_name': 'actin#free_ATP'}, 40: {'neighbor_ids': [], 'position': (18.03001399435682, -1.593752271590134, 14.209981848800536), 'type_name': 'actin#free_ATP'}, 41: {'neighbor_ids': [], 'position': (4.3985816426348485, 6.982142903282081, -17.575990467068575), 'type_name': 'actin#free_ATP'}, 42: {'neighbor_ids': [], 'position': (21.16037796725733, 1.1948995128230733, -4.505275130173861), 'type_name': 'actin#free_ATP'}, 43: {'neighbor_ids': [], 'position': (7.526896911818613, 13.567745996099996, -2.039219615405501), 'type_name': 'actin#free_ATP'}, 44: {'neighbor_ids': [], 'position': (13.57786978193259, -23.949086753913015, 5.72955542257794), 'type_name': 'actin#free_ATP'}, 45: {'neighbor_ids': [], 'position': (14.524125465432725, 6.004266933779155, 21.93307767163814), 'type_name': 'actin#free_ATP'}, 46: {'neighbor_ids': [], 'position': (15.966857212713533, -6.861263278102597, -3.2626323287007537), 'type_name': 'actin#free_ATP'}, 47: {'neighbor_ids': [], 'position': (15.984895555642568, -22.155108870412338, 8.255951254681282), 'type_name': 'actin#free_ATP'}}, 'topologies': {0: {'particle_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'type_name': 'Actin-Polymer'}, 1: {'particle_ids': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], 'type_name': 'Membrane'}, 2: {'particle_ids': [33], 'type_name': 'Actin-Monomer-ATP'}, 3: {'particle_ids': [34], 'type_name': 'Actin-Monomer-ATP'}, 4: {'particle_ids': [35], 'type_name': 'Actin-Monomer-ATP'}, 5: {'particle_ids': [36], 'type_name': 'Actin-Monomer-ATP'}, 6: {'particle_ids': [37], 'type_name': 'Actin-Monomer-ATP'}, 7: {'particle_ids': [38], 'type_name': 'Actin-Monomer-ATP'}, 8: {'particle_ids': [39], 'type_name': 'Actin-Monomer-ATP'}, 9: {'particle_ids': [40], 'type_name': 'Actin-Monomer-ATP'}, 10: {'particle_ids': [41], 'type_name': 'Actin-Monomer-ATP'}, 11: {'particle_ids': [42], 'type_name': 'Actin-Monomer-ATP'}, 12: {'particle_ids': [43], 'type_name': 'Actin-Monomer-ATP'}, 13: {'particle_ids': [44], 'type_name': 'Actin-Monomer-ATP'}, 14: {'particle_ids': [45], 'type_name': 'Actin-Monomer-ATP'}, 15: {'particle_ids': [46], 'type_name': 'Actin-Monomer-ATP'}, 16: {'particle_ids': [47], 'type_name': 'Actin-Monomer-ATP'}}}

def test_readdy_actin_model() -> None:
    random.seed(0)
    np.random.seed(0)
    config = get_default_config()
    actin_sim = ActinSimulation(config, False, False)
    readdy_system: ReactionDiffusionSystem = actin_sim.system
    readdy_simulation: Simulation = actin_sim.simulation

    ReaddyUtil.add_monomers_from_data(readdy_simulation, get_monomers())
    simulate_readdy(0.1, readdy_system=readdy_system, readdy_simulation=readdy_simulation, timestep=1)

    id_diff = id_difference(readdy_simulation.current_topologies)
    result_monomers = ReaddyUtil.get_current_monomers(
        readdy_simulation.current_topologies,
        id_diff
    )

    for k in result_monomers["particles"].keys():
        pos = result_monomers["particles"][k]["position"]
        result_monomers["particles"][k]["position"] = tuple(pos.tolist())

    assert result_monomers is not None
    for k in range(len(result_monomers["particles"].keys())):
        res_particles = result_monomers["particles"][k]
        expected_particles = expected_monomers["particles"][k]
        assert res_particles["neighbor_ids"] == expected_particles["neighbor_ids"]
        for i in range(len(res_particles["position"])):
            assert math.isclose(res_particles["position"][i], expected_particles["position"][i], abs_tol=5)
        assert res_particles["type_name"] == expected_particles["type_name"]


def test_readdy_actin_pb() -> None:
    state = generate_readdy_pbg(output_dir="")

    core = ProcessTypes()
    register_items_into_core(core)

    sim = Composite(
        {
            "state": state,
        },
        core=core,
    )

    # simulate
    sim.run(1)  # time in ns

    assert 'particles' in sim.state
    assert 'topologies' in sim.state

