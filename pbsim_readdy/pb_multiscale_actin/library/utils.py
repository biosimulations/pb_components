import math

import numpy as np
from readdy import ReactionDiffusionSystem, Simulation
from readdy.api.experimental.action_factory import ActionFactory
from simularium_readdy_models.actin import ActinGenerator, FiberData
from simularium_readdy_models.common import get_membrane_monomers


def get_monomers(membrane_particle_radius: float=25):
    actin_monomers = ActinGenerator.get_monomers(
        fibers_data=[
            FiberData(
                28,
                [
                    np.array([-25, 0, 0]),
                    np.array([25, 0, 0]),
                ],
                "Actin-Polymer",
            )
        ],
        use_uuids=False,
        start_normal=np.array([0.0, 1.0, 0.0]),
        longitudinal_bonds=True,
        barbed_binding_site=True,
    )
    actin_monomers = ActinGenerator.setup_fixed_monomers(
        actin_monomers,
        orthogonal_seed=True,
        n_fixed_monomers_pointed=3,
        n_fixed_monomers_barbed=0,
    )
    membrane_monomers = get_membrane_monomers(
        center=np.array([25.0, 0.0, 0.0]),
        size=np.array([0.0, 100.0, 100.0]),
        particle_radius=membrane_particle_radius,
        start_particle_id=len(actin_monomers["particles"].keys()),
        top_id=1,
    )
    free_actin_monomers = ActinGenerator.get_free_actin_monomers(
        concentration=500.0,
        box_center=np.array([12.0, 0.0, 0.0]),
        box_size=np.array([20.0, 50.0, 50.0]),
        start_particle_id=len(actin_monomers["particles"].keys())
        + len(membrane_monomers["particles"].keys()),
        start_top_id=2,
    )
    monomers = {
        "particles": {
            **actin_monomers["particles"],
            **membrane_monomers["particles"],
            **free_actin_monomers["particles"]
        },
        "topologies": {
            **actin_monomers["topologies"],
            **membrane_monomers["topologies"],
            **free_actin_monomers["topologies"],
        },
    }

    return monomers


def id_difference(current_topologies):
    """
    Get the first ID coming out of ReaDDy, it should be zero
    unless Readdy ran multiple times and cached the IDs, which
    would cause Vivarium to create new particles instead of
    updating existing particles.

    (This is a HACK needed as long as ReaDDy has this behavior.)
    """
    return current_topologies[0].particles[0].id


def simulate_readdy(internal_timestep: float, readdy_system: ReactionDiffusionSystem, readdy_simulation: Simulation, timestep: float):
    """
    Simulate in ReaDDy for the given timestep
    """

    def loop():
        readdy_actions: ActionFactory = readdy_simulation._actions
        init = readdy_actions.initialize_kernel() # Type: readdybinding.api.actions.Action
        diffuse = readdy_actions.integrator_euler_brownian_dynamics(internal_timestep)
        calculate_forces = readdy_actions.calculate_forces()
        create_nl = readdy_actions.create_neighbor_list(
            readdy_system.calculate_max_cutoff().magnitude
        )
        update_nl = readdy_actions.update_neighbor_list()
        react = readdy_actions.reaction_handler_uncontrolled_approximation(
            internal_timestep
        )
        init()
        create_nl()
        calculate_forces()
        update_nl()
        n_steps = int(timestep / internal_timestep)
        print(f"running readdy for {n_steps} steps")
        for t in range(1, n_steps + 1):
            diffuse()
            update_nl()
            react()
            update_nl()
            calculate_forces()

    readdy_simulation._run_custom_loop(loop, show_summary=False)


##################
# Test Functions #
##################

pre_sim_particles = {0: {'neighbor_ids': [1, 2], 'position': (-25.0, 1.617886445553052, 0.0), 'type_name': 'actin#pointed_fixed_ATP_1'}, 1: {'neighbor_ids': [0, 2, 3], 'position': (-22.196901237, -1.576092210430552, -0.365362689562598), 'type_name': 'actin#fixed_ATP_2'}, 2: {'neighbor_ids': [1, 3, 0, 4], 'position': (-19.393802473999997, 1.4528688137021482, 0.7118488328823619), 'type_name': 'actin#mid_fixed_ATP_3'}, 3: {'neighbor_ids': [2, 4, 1, 5], 'position': (-16.590703710999996, -1.254582620114409, -1.0215571447604492), 'type_name': 'actin#mid_ATP_4'}, 4: {'neighbor_ids': [3, 5, 2, 6], 'position': (-13.787604947999997, 0.9914781308424695, 1.2784864750029112), 'type_name': 'actin#mid_ATP_5'}, 5: {'neighbor_ids': [4, 6, 3, 7], 'position': (-10.984506184999997, -0.6771486990127372, -1.4693625114755011), 'type_name': 'actin#mid_ATP_1'}, 6: {'neighbor_ids': [5, 7, 4, 8], 'position': (-8.181407421999998, 0.32783422602981593, 1.584323600451516), 'type_name': 'actin#mid_ATP_2'}, 7: {'neighbor_ids': [6, 8, 5, 9], 'position': (-5.378308658999998, 0.038417876613766805, -1.617430251188811), 'type_name': 'actin#mid_ATP_3'}, 8: {'neighbor_ids': [7, 9, 6, 10], 'position': (-2.5752098959999983, -0.4026851109465761, 1.5669720010664332), 'type_name': 'actin#mid_ATP_4'}, 9: {'neighbor_ids': [8, 10, 7, 11], 'position': (0.22788886700000166, 0.7461475276687426, -1.43555578702397), 'type_name': 'actin#mid_ATP_5'}, 10: {'neighbor_ids': [9, 11, 8, 12], 'position': (3.0309876300000016, -1.0510600631020548, 1.2299712575731516), 'type_name': 'actin#mid_ATP_1'}, 11: {'neighbor_ids': [10, 12, 9, 13], 'position': (5.8340863930000015, 1.3016693419305578, -0.9608399840671874), 'type_name': 'actin#mid_ATP_2'}, 12: {'neighbor_ids': [11, 13, 10, 14], 'position': (8.637185156000001, -1.4850275789100815, 0.6420666948071793), 'type_name': 'actin#mid_ATP_3'}, 13: {'neighbor_ids': [12, 14, 11, 15], 'position': (11.440283919, 1.591661529130228, -0.2901208841000041), 'type_name': 'actin#mid_ATP_4'}, 14: {'neighbor_ids': [13, 15, 12, 16], 'position': (14.243382682, -1.6160619253617787, -0.07681408790133856), 'type_name': 'actin#mid_ATP_5'}, 15: {'neighbor_ids': [14, 16, 13], 'position': (17.046481445, 1.5569681155979302, 0.4397804426251286), 'type_name': 'actin#barbed_ATP_1'}, 16: {'neighbor_ids': [15, 14], 'position': (19.849580208000003, -1.4174331949262187, -0.7800254410116159), 'type_name': 'binding_site#2'}, 17: {'neighbor_ids': [25, 18, 19], 'position': (50.0, -50.0, -50.0), 'type_name': 'membrane#outer_edge_4_1'}, 18: {'neighbor_ids': [17, 19, 26, 20], 'position': (50.0, 0.0, -50.0), 'type_name': 'membrane#outer_edge_1'}, 19: {'neighbor_ids': [17, 18, 27, 20, 22, 21], 'position': (50.0, -25.0, -6.698729810778069), 'type_name': 'membrane#outer'}, 20: {'neighbor_ids': [18, 19, 28, 22], 'position': (50.0, 25.0, -6.698729810778069), 'type_name': 'membrane#outer_edge_2'}, 21: {'neighbor_ids': [19, 29, 22, 23], 'position': (50.0, -50.0, 36.60254037844386), 'type_name': 'membrane#outer_edge_4'}, 22: {'neighbor_ids': [19, 20, 21, 23, 30, 24], 'position': (50.0, 0.0, 36.60254037844386), 'type_name': 'membrane#outer'}, 23: {'neighbor_ids': [21, 22, 31, 24], 'position': (50.0, -25.0, 79.9038105676658), 'type_name': 'membrane#outer_edge_3'}, 24: {'neighbor_ids': [22, 23, 32], 'position': (50.0, 25.0, 79.9038105676658), 'type_name': 'membrane#outer_edge_2_3'}, 25: {'neighbor_ids': [17, 26, 27], 'position': (0.0, -50.0, -50.0), 'type_name': 'membrane#inner_edge_4_1'}, 26: {'neighbor_ids': [25, 27, 18, 28], 'position': (0.0, 0.0, -50.0), 'type_name': 'membrane#inner_edge_1'}, 27: {'neighbor_ids': [25, 26, 19, 28, 30, 29], 'position': (0.0, -25.0, -6.698729810778069), 'type_name': 'membrane#inner'}, 28: {'neighbor_ids': [26, 27, 20, 30], 'position': (0.0, 25.0, -6.698729810778069), 'type_name': 'membrane#inner_edge_2'}, 29: {'neighbor_ids': [27, 21, 30, 31], 'position': (0.0, -50.0, 36.60254037844386), 'type_name': 'membrane#inner_edge_4'}, 30: {'neighbor_ids': [27, 28, 29, 31, 22, 32], 'position': (0.0, 0.0, 36.60254037844386), 'type_name': 'membrane#inner'}, 31: {'neighbor_ids': [29, 30, 23, 32], 'position': (0.0, -25.0, 79.9038105676658), 'type_name': 'membrane#inner_edge_3'}, 32: {'neighbor_ids': [30, 31, 24], 'position': (0.0, 25.0, 79.9038105676658), 'type_name': 'membrane#inner_edge_2_3'}, 33: {'neighbor_ids': [], 'position': (12.976270078546495, 10.759468318620973, 5.138168803582194), 'type_name': 'actin#free_ATP', 'unique_id': 33}, 34: {'neighbor_ids': [], 'position': (12.897663659937937, -3.8172600330547644, 7.294705653332806), 'type_name': 'actin#free_ATP', 'unique_id': 34}, 35: {'neighbor_ids': [], 'position': (10.75174422525385, 19.58865003910399, 23.183138025051463), 'type_name': 'actin#free_ATP', 'unique_id': 35}, 36: {'neighbor_ids': [], 'position': (9.668830376515555, 14.58625190413323, 1.444745987645224), 'type_name': 'actin#free_ATP', 'unique_id': 36}, 37: {'neighbor_ids': [], 'position': (13.360891221878646, 21.27983191463305, -21.44819709010565), 'type_name': 'actin#free_ATP', 'unique_id': 37}, 38: {'neighbor_ids': [], 'position': (3.742585994030815, -23.989080127983716, 16.6309922773969), 'type_name': 'actin#free_ATP', 'unique_id': 38}, 39: {'neighbor_ids': [], 'position': (17.56313501899701, 18.50060741234096, 23.930917111638202), 'type_name': 'actin#free_ATP', 'unique_id': 39}, 40: {'neighbor_ids': [], 'position': (17.98317128433447, -1.9260318873534077, 14.026458814322773), 'type_name': 'actin#free_ATP', 'unique_id': 40}, 41: {'neighbor_ids': [], 'position': (4.365488517378664, 6.996051066376191, -17.83233562954768), 'type_name': 'actin#free_ATP', 'unique_id': 41}, 42: {'neighbor_ids': [], 'position': (20.893378340991678, 1.0924160875035838, -4.266903000473821), 'type_name': 'actin#free_ATP', 'unique_id': 42}, 43: {'neighbor_ids': [], 'position': (7.29111224209254, 13.711684471710834, -2.1924833891725726), 'type_name': 'actin#free_ATP', 'unique_id': 43}, 44: {'neighbor_ids': [], 'position': (13.36867897737297, -24.060509978182242, 5.8817748537938535), 'type_name': 'actin#free_ATP', 'unique_id': 44}, 45: {'neighbor_ids': [], 'position': (14.241914454448429, 5.846699843737846, 22.187403925731207), 'type_name': 'actin#free_ATP', 'unique_id': 45}, 46: {'neighbor_ids': [], 'position': (15.636405982069668, -7.0246049713106995, -3.1484023100329273), 'type_name': 'actin#free_ATP', 'unique_id': 46}, 47: {'neighbor_ids': [], 'position': (15.952623918545298, -21.98872641853651, 8.338335772283385), 'type_name': 'actin#free_ATP', 'unique_id': 47}}
pre_sim_topologies = {0: {'particle_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'type_name': 'Actin-Polymer'}, 1: {'particle_ids': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], 'type_name': 'Membrane'}, 2: {'particle_ids': [33], 'type_name': 'Actin-Monomer-ATP'}, 3: {'particle_ids': [34], 'type_name': 'Actin-Monomer-ATP'}, 4: {'particle_ids': [35], 'type_name': 'Actin-Monomer-ATP'}, 5: {'particle_ids': [36], 'type_name': 'Actin-Monomer-ATP'}, 6: {'particle_ids': [37], 'type_name': 'Actin-Monomer-ATP'}, 7: {'particle_ids': [38], 'type_name': 'Actin-Monomer-ATP'}, 8: {'particle_ids': [39], 'type_name': 'Actin-Monomer-ATP'}, 9: {'particle_ids': [40], 'type_name': 'Actin-Monomer-ATP'}, 10: {'particle_ids': [41], 'type_name': 'Actin-Monomer-ATP'}, 11: {'particle_ids': [42], 'type_name': 'Actin-Monomer-ATP'}, 12: {'particle_ids': [43], 'type_name': 'Actin-Monomer-ATP'}, 13: {'particle_ids': [44], 'type_name': 'Actin-Monomer-ATP'}, 14: {'particle_ids': [45], 'type_name': 'Actin-Monomer-ATP'}, 15: {'particle_ids': [46], 'type_name': 'Actin-Monomer-ATP'}, 16: {'particle_ids': [47], 'type_name': 'Actin-Monomer-ATP'}}

def compare_particles(particles: dict, expected_particles: dict, is_ndarray: bool = True) -> None:
    if is_ndarray:
        for k in particles.keys():
            pos = particles[k]["position"]
            particles[k]["position"] = tuple(pos.tolist())

    for k in range(len(particles.keys())):
        res_particles = particles[k]
        exp_particle = expected_particles[k]
        res_particles["neighbor_ids"].sort()
        exp_particle["neighbor_ids"].sort()
        for i in range(len(res_particles["position"])):
            assert math.isclose(res_particles["position"][i], exp_particle["position"][i], abs_tol=5)
        assert res_particles["type_name"] == exp_particle["type_name"]

def compare_topologies(topologies: dict, expected_topologies: dict) -> None:
    for k in topologies.keys():
        topologies[k]["particle_ids"].sort()
        expected_topologies[k]["particle_ids"].sort()

        assert topologies[k]["particle_ids"] == expected_topologies[k]["particle_ids"]
        assert topologies[k]["type_name"] == expected_topologies[k]["type_name"]

