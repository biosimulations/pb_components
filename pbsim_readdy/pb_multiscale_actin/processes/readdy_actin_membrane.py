import random

import numpy as np
from process_bigraph import Process
from readdy import ReactionDiffusionSystem, Simulation
from simularium_readdy_models.actin import (
    ActinSimulation,
)
from simularium_readdy_models.common import ReaddyUtil

from pb_multiscale_actin.library.utils import get_monomers, simulate_readdy, id_difference


class ReaddyActinMembrane(Process):
    """
    This process runs ReaDDy models with coarse-grained particle
    actin filaments and membrane patches.
    """

    config_schema = {
        "name": "string",
        "internal_timestep": "float{0.1}", # ns
        "box_size": "tuple[float,float,float]{150, 150, 150}",
        "periodic_boundary": "boolean{true}",
        "reaction_distance": "float{1.0}", # nm
        "n_cpu": "integer{4}",
        "only_linear_actin_constraints": "boolean{true}",
        "reactions": "boolean{true}",
        "dimerize_rate": "float{1e-30}", # 1/ns
        "dimerize_reverse_rate": "float{1.4e-9}", # 1/ns
        "trimerize_rate": "float{2.1e-2}", # 1/ns
        "trimerize_reverse_rate": "float{1.4e-9}", # 1/ns
        "pointed_growth_ATP_rate": "float{2.4e-5}", # 1/ns
        "pointed_growth_ADP_rate": "float{2.95e-6}", # 1/ns
        "pointed_shrink_ATP_rate": "float{8.0e-10}", # 1/ns
        "pointed_shrink_ADP_rate": "float{3.0e-10}", # 1/ns
        "barbed_growth_ATP_rate": "float{1e30}", # 1/ns
        "barbed_growth_ADP_rate": "float{7.0e-5}", # 1/ns
        "nucleate_ATP_rate": "float{2.1e-2}", # 1/ns
        "nucleate_ADP_rate": "float{7.0e-5}", # 1/ns
        "barbed_shrink_ATP_rate": "float{1.4e-9}", # 1/ns
        "barbed_shrink_ADP_rate": "float{8.0e-9}", # 1/ns
        "arp_bind_ATP_rate": "float{2.1e-2}", # 1/ns
        "arp_bind_ADP_rate": "float{7.0e-5}", # 1/ns
        "arp_unbind_ATP_rate": "float{1.4e-9}", # 1/ns
        "arp_unbind_ADP_rate": "float{8.0e-9}", # 1/ns
        "barbed_growth_branch_ATP_rate": "float{2.1e-2}", # 1/ns
        "barbed_growth_branch_ADP_rate": "float{7.0e-5}", # 1/ns
        "debranching_ATP_rate": "float{1.4e-9}", # 1/ns
        "debranching_ADP_rate": "float{7.0e-5}", # 1/ns
        "cap_bind_rate": "float{2.1e-2}", # 1/ns
        "cap_unbind_rate": "float{1.4e-9}", # 1/ns
        "hydrolysis_actin_rate": "float{1e-30}", # 1/ns
        "hydrolysis_arp_rate": "float{3.5e-5}", # 1/ns
        "nucleotide_exchange_actin_rate": "float{1e-5}", # 1/ns
        "nucleotide_exchange_arp_rate": "float{1e-5}", # 1/ns
        "verbose": "boolean{false}",
        "use_box_actin": "boolean{true}",
        "use_box_arp": "boolean{false}",
        "use_box_cap": "boolean{false}",
        "obstacle_radius": "float{0.0}",
        "obstacle_diff_coeff": "float{0.0}",
        "use_box_obstacle": "boolean{false}",
        "position_obstacle_stride": "integer{0}",
        "displace_pointed_end_tangent": "boolean{false}",
        "displace_pointed_end_radial": "boolean{false}",
        "tangent_displacement_nm": "float{0.0}",
        "radial_displacement_radius_nm": "float{0.0}",
        "radial_displacement_angle_deg": "float{0.0}",
        "longitudinal_bonds": "boolean{true}",
        "displace_stride": "integer{1}",
        "bonds_force_multiplier": "float{0.2}",
        "angles_force_constant": "float{1000.0}",
        "dihedrals_force_constant": "float{1000.0}",
        "actin_constraints": "boolean{true}",
        "use_box_actin": "boolean{true}",
        "actin_box_center_x": "float{12.0}",
        "actin_box_center_y": "float{0.0}",
        "actin_box_center_z": "float{0.0}",
        "actin_box_size_x": "float{20.0}",
        "actin_box_size_y": "float{50.0}",
        "actin_box_size_z": "float{50.0}",
        "actin_concentration": "float{500.0}",
        "add_extra_box": "boolean{false}",
        "barbed_binding_site": "boolean{true}",
        "binding_site_reaction_distance": "float{3.0}",
        "add_membrane": "boolean{true}",
        "membrane_center_x": "float{25.0}",
        "membrane_center_y": "float{0.0}",
        "membrane_center_z": "float{0.0}",
        "membrane_size_x": "float{0.0}",
        "membrane_size_y": "float{100.0}",
        "membrane_size_z": "float{100.0}",
        "membrane_particle_radius": "float{2.5}",
        "obstacle_controlled_position_x": "float{0.0}",
        "obstacle_controlled_position_y": "float{0.0}",
        "obstacle_controlled_position_z": "float{0.0}",
        "random_seed": "integer",
        "total_steps": "float",
        "arp23_concentration": "integer",  # 0
        "cap_concentration": "integer",  # 0
        "n_fixed_monomers_barbed": "integer{0}",
        "n_fixed_monomers_pointed": "integer{3}",
        "tangent_displace_speed_um_s": "float",
        "plot_actin_compression": "boolean",  # true
        "visualize_edges": "boolean",  # true
        "visualize_normals": "boolean",  # true
        "visualize_control_pts": "boolean",  # true
    }

    def initialize(self, config, readdy_system=None):
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        actin_simulation = ActinSimulation(self.config, False, False, readdy_system)
        self.readdy_system: ReactionDiffusionSystem = actin_simulation.system
        self.readdy_simulation: Simulation = actin_simulation.simulation

    def initial_state(self):
        return get_monomers(self.config)

    def inputs(self):
        return {
            "topologies": "map[overwrite[topology]]",
            "particles": "map[overwrite[particle]]",
        }

    def outputs(self):
        return {
            "topologies": "map[overwrite[topology]]",
            "particles": "map[overwrite[particle]]",
        }

    def update(self, inputs, interval: float):
        self.initialize(self.config, self.readdy_system)
        monomers = {
            'particles': inputs['particles'],
            'topologies': inputs['topologies'],
        }

        ReaddyUtil.add_monomers_from_data(self.readdy_simulation, monomers)

        simulate_readdy(
            self.config["internal_timestep"],
            self.readdy_system,
            self.readdy_simulation,
            interval,
        )

        id_diff = id_difference(self.readdy_simulation.current_topologies)

        readdy_monomers = ReaddyUtil.get_current_monomers(
            self.readdy_simulation.current_topologies, id_diff
        )

        return readdy_monomers
