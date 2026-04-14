import math

import numpy as np
from readdy import ReactionDiffusionSystem, Simulation
from readdy.api.experimental.action_factory import ActionFactory
from simularium_readdy_models.actin import ActinGenerator, FiberData
from simularium_readdy_models.common import get_membrane_monomers


def get_monomers(config: dict):
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
    counter_id = 0
    monomers = {}
    if config['add_filament']:
        actin_monomers = ActinGenerator.setup_fixed_monomers(
            actin_monomers,
            orthogonal_seed=True,
            n_fixed_monomers_pointed=config['n_fixed_monomers_pointed'],
            n_fixed_monomers_barbed=config['n_fixed_monomers_barbed'],
        )
        counter_id += len(actin_monomers["particles"])
        free_actin_monomers = ActinGenerator.get_free_actin_monomers(
            concentration=config['actin_concentration'],
            box_center=np.array(
                [config['actin_box_center_x'], config['actin_box_center_y'], config['actin_box_center_z']]),
            box_size=np.array([config['actin_box_size_x'], config['actin_box_size_y'], config['actin_box_size_z']]),
            start_particle_id=counter_id,
            start_top_id=2,
        )
        counter_id += len(free_actin_monomers["particles"])
        monomers.update({
            "particles": {
                **actin_monomers["particles"],
                **free_actin_monomers["particles"]
            },
            "topologies": {
                **actin_monomers["topologies"],
                **free_actin_monomers["topologies"]
            }
        })
    if config["add_membrane"]:
        membrane_monomers = get_membrane_monomers(
            center=np.array([config['membrane_center_x'], config['membrane_center_y'], config['membrane_center_z']]),
            size=np.array([config['membrane_size_x'], config['membrane_size_y'], config['membrane_size_z']]),
            particle_radius=config['membrane_particle_radius'],
            start_particle_id=counter_id,
            top_id=1,
        )
        for k, v in membrane_monomers['particles'].items():
            monomers['particles'].update({k: v})
        for k, v in membrane_monomers['topologies'].items():
            monomers['topologies'].update({k: v})

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
            print(f"step {t}")
            diffuse()
            update_nl()
            react()
            update_nl()
            calculate_forces()

    readdy_simulation._run_custom_loop(loop, show_summary=False)
