# MIT License

# Copyright (c) 2024 anonymity

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import json
import logging
import sys
import time

import numpy as np
import torch

from lib import config, Ising, utils, problem


def load_default_config(energy):
    """
    Load default parameter configuration from file.

    Args:
        tasks: String with the energy name

    Returns:
        Dictionary of default parameters for the given energy
    """
    if energy == "Lyapunov function":
        default_config = "etc/Lyapunov_energy.json"
    elif energy == "Hamiltonian":
        default_config = "etc/default_para.json"
    else:
        raise ValueError("Ising based Energy model \"{}\" not defined.".format(energy))

    with open(default_config) as config_json_file:
        cfg = json.load(config_json_file)

    return cfg


def parse_shell_args(args):
    """
    Parse shell arguments for this script.

    Args:
        args: List of shell arguments

    Returns:
        Dictionary of shell arguments
    """
    parser = argparse.ArgumentParser(                  
        description="Evolve Ising spin swarms on random graphs using IOF and BiSON."
    )

    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS,
                        help="Population of spin swarms during optimization.")
    parser.add_argument("--graph", type=str, default=argparse.SUPPRESS, 
                        help="Ising coupling topology to be optimized")
    parser.add_argument("--energy", choices=["Hamiltonian", "Lyapunov function"],               #***
                        default="Hamiltonian", help="Type of energy representations.")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
                        help="Number of epochs to evolve.")
    parser.add_argument("--expanding_rate", type=float, default=argparse.SUPPRESS,
                        help="Expanding rate of the energy space.")
    parser.add_argument("--log_dir", type=str, default="",                                     #***
                        help="Subdirectory within ./log/ to store logs.")
    parser.add_argument("--nonlinearity", choices=["SHG", "periodic", "polynomial", "tanh"],
                        default=argparse.SUPPRESS, help="Nonlinearity for Ising activations.")
    parser.add_argument("--animation", type=bool, default=argparse.SUPPRESS,
                        help="Generate animation of the spin swarm evolution")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS,
                        help="Random seed for pytorch")

    return vars(parser.parse_args(args))


def run_Ising_optimization(cfg):
    """
    Main script.

    Args:
        cfg: Dictionary defining parameters of the run
    """

    # log record of the current optimization
    logging.info("Start Ising Optimization with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))
    
    # Initialize seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ****************************************** deifne the problem to be optimized *******************************************
    J, J_intact, eig_max, global_minima= problem.graph(cfg['graph'], device, cfg["mu"], SD=cfg['SD'],)
    spins = len(J)


    spin = (torch.randn(cfg['batch_size'], spins) * 0.01).to(device)
    his_best = 0


    T0 = time.perf_counter()
    for epoch in range(cfg['epochs']):
            
            #************************************************** spin state mutation - basin jump ****************************************************************           
            P = (torch.exp(torch.tensor(-epoch / cfg['epochs'], dtype=torch.float32)) - torch.exp(torch.tensor(-1.0))) / (1 - torch.exp(torch.tensor(-1.0)))
            # Calculate number of spins to flip
            num_flips = max(int(spins * 0.5 * P.item()), 1)  # P.item() to get scalar value
            
            # Generate unique random indices for each batch using PyTorch         
            prob_matrix = torch.ones(cfg['batch_size'], spins, device=device)
            index = torch.multinomial(prob_matrix, num_flips, replacement=False)
            
            # Flip the selected spins for each batch
            batch_indices = torch.arange(cfg['batch_size'], device=device).unsqueeze(1).expand(cfg['batch_size'], num_flips)
            spin[batch_indices, index] = -spin[batch_indices, index]

            # ********************************************************* Ising Optimizer ************************************************************
            spin, local_minima = Ising.swarm_evolution(cfg['alpha'], cfg['beta'], spin, J, J_intact, iterations=cfg['evo_steps'], device=device)
            
            
             # ********************************************************* Greedy ************************************************************
            greedy, batch_negative_indices, batch4greedy= utils.greedy(J_intact, cfg['batch_size'], spin)

            g_times = 0
            while greedy:

                flipped_bit_indices = [tensor[torch.randint(0, tensor.size(0), (1,)).item()] for tensor, select in zip(batch_negative_indices, batch4greedy) if select]

                # Update the spin configuration
                index4greedy = torch.nonzero(torch.tensor(batch4greedy), as_tuple=False)
                spin = utils.Move(spin, flipped_bit_indices, index4greedy.squeeze())

                greedy, batch_negative_indices, batch4greedy= utils.greedy(J_intact, cfg['batch_size'], spin)

                g_times += 1

            #Update best-found
            local_minima = utils.Ham(spin, J_intact)
            # Update best-found values
            current_best = local_minima.min().item()
            
            if current_best < his_best:
                his_best = current_best
                spin_best = spin[local_minima.argmin()].clone()

            if his_best<=global_minima:
                print(f"Succeed at the {epoch}-th epoch!")
                break

    TN = time.perf_counter()
    print(f"spin number: {spins}      total optimizatino time：{1000*(TN-T0)} ms")
    # print(f"historical best solution: {his_best}\nbest spin state:{np.sign(spin_best.cpu())}")

    # Logging
    logging.info(
    "epoch: {} \t Discovered Lowest Hamiltonian: {:.4f} \t Best Cut: {:.4f} \t Optimization Time: {:.4f} ms".format(
        epoch, his_best, utils.Cut( spin_best.unsqueeze(0) , J_intact ).item(), 1000*(TN-T0) )
    )



if __name__ == '__main__':
    # Parse shell arguments as input configuration
    user_config = parse_shell_args(sys.argv[1:])   
    print(f"The user_config is: {(user_config)}")

    # Load default parameter configuration from file for the specified Ising energy-based model  <class 'dict'>
    cfg = load_default_config(user_config["energy"])
    print(f"The default cfg is: {(cfg)}")

    # Overwrite default parameters with user configuration where applicable
    cfg.update(user_config)
    print(f"The user-defined cfg is: {(cfg)}")

    # Setup global logger and logging directory
    config.setup_logging(cfg["energy"],
                         dir=cfg['log_dir'])                                              

    # Run the script using the created paramter configuration
    run_Ising_optimization(cfg)
