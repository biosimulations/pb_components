import random
from typing import Any

import numpy as np
from process_bigraph import ProcessTypes, Composite
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results
from simularium_readdy_models.actin import ActinGenerator, FiberData
from simularium_readdy_models.common import get_membrane_monomers

from pb_multiscale_actin.processes import ReaddyActinMembrane
from pb_multiscale_actin.processes import SimulariumEmitter

def get_default_config() -> dict[str, Any]:
    return {
        "name": "actin_membrane",
        "internal_timestep": 0.1,  # ns
        "box_size": np.array([float(150.0)] * 3),  # nm
        "periodic_boundary": True,
        "reaction_distance": 1.0,  # nm
        "n_cpu": 4,
        "only_linear_actin_constraints": True,
        "reactions": True,
        "dimerize_rate": 1e-30,  # 1/ns
        "dimerize_reverse_rate": 1.4e-9,  # 1/ns
        "trimerize_rate": 2.1e-2,  # 1/ns
        "trimerize_reverse_rate": 1.4e-9,  # 1/ns
        "pointed_growth_ATP_rate": 2.4e-5,  # 1/ns
        "pointed_growth_ADP_rate": 2.95e-6,  # 1/ns
        "pointed_shrink_ATP_rate": 8.0e-10,  # 1/ns
        "pointed_shrink_ADP_rate": 3.0e-10,  # 1/ns
        "barbed_growth_ATP_rate": 1e30,  # 1/ns
        "barbed_growth_ADP_rate": 7.0e-5,  # 1/ns
        "nucleate_ATP_rate": 2.1e-2,  # 1/ns
        "nucleate_ADP_rate": 7.0e-5,  # 1/ns
        "barbed_shrink_ATP_rate": 1.4e-9,  # 1/ns
        "barbed_shrink_ADP_rate": 8.0e-9,  # 1/ns
        "arp_bind_ATP_rate": 2.1e-2,  # 1/ns
        "arp_bind_ADP_rate": 7.0e-5,  # 1/ns
        "arp_unbind_ATP_rate": 1.4e-9,  # 1/ns
        "arp_unbind_ADP_rate": 8.0e-9,  # 1/ns
        "barbed_growth_branch_ATP_rate": 2.1e-2,  # 1/ns
        "barbed_growth_branch_ADP_rate": 7.0e-5,  # 1/ns
        "debranching_ATP_rate": 1.4e-9,  # 1/ns
        "debranching_ADP_rate": 7.0e-5,  # 1/ns
        "cap_bind_rate": 2.1e-2,  # 1/ns
        "cap_unbind_rate": 1.4e-9,  # 1/ns
        "hydrolysis_actin_rate": 1e-30,  # 1/ns
        "hydrolysis_arp_rate": 3.5e-5,  # 1/ns
        "nucleotide_exchange_actin_rate": 1e-5,  # 1/ns
        "nucleotide_exchange_arp_rate": 1e-5,  # 1/ns
        "verbose": False,
        "use_box_actin": True,
        "use_box_arp": False,
        "use_box_cap": False,
        "obstacle_radius": 0.0,
        "obstacle_diff_coeff": 0.0,
        "use_box_obstacle": False,
        "position_obstacle_stride": 0,
        "displace_pointed_end_tangent": False,
        "displace_pointed_end_radial": False,
        "tangent_displacement_nm": 0.0,
        "radial_displacement_radius_nm": 0.0,
        "radial_displacement_angle_deg": 0.0,
        "longitudinal_bonds": True,
        "displace_stride": 1,
        "bonds_force_multiplier": 0.2,
        "angles_force_constant": 1000.0,
        "dihedrals_force_constant": 1000.0,
        "actin_constraints": True,
        "use_box_actin": True,
        "actin_box_center_x": 12.0,
        "actin_box_center_y": 0.0,
        "actin_box_center_z": 0.0,
        "actin_box_size_x": 20.0,
        "actin_box_size_y": 50.0,
        "actin_box_size_z": 50.0,
        "add_extra_box": False,
        "barbed_binding_site": True,
        "binding_site_reaction_distance": 3.0,
        "add_membrane": True,
        "membrane_center_x": 25.0,
        "membrane_center_y": 0.0,
        "membrane_center_z": 0.0,
        "membrane_size_x": 0.0,
        "membrane_size_y": 100.0,
        "membrane_size_z": 100.0,
        'membrane_particle_radius': 2.5,
        'obstacle_controlled_position_x': 0.0,
        'obstacle_controlled_position_y': 0.0,
        'obstacle_controlled_position_z': 0.0,
        'random_seed': 0,
    }


def register_items_into_core(core: ProcessTypes):
    particle = {
        'type_name': 'string',
        'position': 'tuple[float,float,float]',
        'neighbor_ids': 'list[integer]',
        '_apply': 'set',
    }
    topology = {
        'type_name': 'string',
        'particle_ids': 'list[integer]',
        '_apply': 'set',
    }
    core.register('topology', topology)
    core.register('particle', particle)

    core.register_process('pb_multiscale_actin.processes.readdy_actin_membrane.ReaddyActinMembrane', ReaddyActinMembrane)
    core.register_process('pb_multiscale_actin.processes.simularium_emitter.SimulariumEmitter', SimulariumEmitter)


def generate_readdy_pbg(output_dir):
    emitters_from_wires = emitter_from_wires({
        'particles': ['particles'],
        'topologies': ['topologies'],
        'global_time': ['global_time']
    }, address='local:pb_multiscale_actin.processes.simularium_emitter.SimulariumEmitter')
    emitters_from_wires["config"]["output_dir"] = output_dir

    state = {
        "emitter": emitters_from_wires,
        'readdy': {
            '_type': 'process',
            'config': get_default_config(),
            'address': 'local:pb_multiscale_actin.processes.readdy_actin_membrane.ReaddyActinMembrane',
            'inputs': {
                'particles': ['particles'],
                'topologies': ['topologies']
            },
            'outputs': {
                'particles': ['particles'],
                'topologies': ['topologies']
            }
        },
    }
    return state

def run_readdy_actin_membrane(total_time=3):
    state = generate_readdy_pbg(output_dir="")

    core = ProcessTypes()
    register_items_into_core(core)

    sim = Composite({
        "state": state,
    }, core=core)

    # simulate
    sim.run(total_time)  # time in ns

    results = gather_emitter_results(sim)


if __name__ == "__main__":
    run_readdy_actin_membrane()


