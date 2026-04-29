# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import copy
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import torch
import argparse

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs:
            runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def parse_default_arguments(description="Isaac Gym Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
    parser.add_argument('--pipeline', type=str, default="gpu", help='Tensor API pipeline (cpu/gpu)')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    # Parse the default arguments (ignore system input)
    args, _ = parser.parse_known_args([])

    args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)
    pipeline = args.pipeline.lower()

    assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
    args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

    if args.sim_device_type != 'cuda' and args.flex:
        print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
        args.sim_device = 'cuda:0'
        args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(args.sim_device)

    if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
        print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
        args.pipeline = 'CPU'
        args.use_gpu_pipeline = False

    # Default to PhysX
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = (args.sim_device_type == 'cuda')

    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args


def get_default_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = parse_default_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,
            "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,
            "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--export", "action": "store_true", "default": False,
            "help": "Export policy as JIT after loading checkpoint"},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'proximal_memory_a'):
        if actor_critic._proximal_indices.numel() == 0:
            raise RuntimeError(
                "Sampling-plan indices are empty. Run at least one act_inference() "
                "call before exporting to trigger _build_sampling_plan."
            )
        exporter = PolicyExporterPDRiskNet(actor_critic)
        exporter.export(path)
    elif hasattr(actor_critic, 'memory_a'):
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterPDRiskNet(torch.nn.Module):
    """TorchScript-compatible wrapper that exports the complete PDRiskNet inference pipeline.

    Copies the trained sub-module weights, bakes the static sampling-plan indices,
    and reimplements the full forward path with operations that ``torch.jit.script``
    can compile.
    """

    def __init__(self, actor_critic):
        super().__init__()
        ac = actor_critic

        # -- 1. copy trained sub-module weights ---------------------------------
        self.prox_point_encoder = copy.deepcopy(ac.proximal_point_encoder)
        self.dist_point_encoder = copy.deepcopy(ac.distal_point_encoder)
        self.prox_spatial_gru = copy.deepcopy(ac.proximal_gru)
        self.dist_spatial_gru = copy.deepcopy(ac.distal_spatial_gru)
        self.prox_memory_gru = copy.deepcopy(ac.proximal_memory_a.rnn)
        self.dist_memory_gru = copy.deepcopy(ac.distal_memory_a.rnn)
        self.actor = copy.deepcopy(ac.actor)

        # -- 2. bake static sampling-plan indices -------------------------------
        self.register_buffer("prox_indices", ac._proximal_indices.clone())
        self.register_buffer("dist_sorted_indices", ac._distal_sorted_indices.clone())
        self.register_buffer("dist_bin_ids", ac._distal_bin_ids.clone())
        self.register_buffer("dist_bin_counts", ac._distal_bin_counts.clone())

        # -- 3. temporal-memory GRU hidden states -------------------------------
        self.register_buffer("prox_hidden", torch.zeros(1, 1, ac.proximal_feature_dim))
        self.register_buffer("dist_hidden", torch.zeros(1, 1, ac.distal_feature_dim))

        # -- 4. hyper-parameters carried as plain attributes --------------------
        self.proprio_dim = int(ac.proprio_obs_dim)
        self.num_points = int(ac.num_lidar_points)
        self.prox_points = int(ac.proximal_points)
        self.dist_points = int(ac.distal_points)
        self.prox_feat_dim = int(ac.proximal_feature_dim)
        self.dist_feat_dim = int(ac.distal_feature_dim)

    # ------------------------------------------------------------------
    #  TorchScript-safe helpers
    # ------------------------------------------------------------------

    def _sort_by_spherical(self, points: torch.Tensor) -> torch.Tensor:
        """Sort points by (theta, phi) ascending (same key as _sort_by_spherical)."""
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        theta = torch.atan2(z, torch.sqrt(x * x + y * y + 1e-8))
        phi = torch.atan2(y, x)
        order = torch.argsort(theta * (2.0 * math.pi) + phi, dim=-1)
        return torch.gather(points, 1, order.unsqueeze(-1).expand_as(points))

    # ------------------------------------------------------------------
    #  Forward  (the exact inference pipeline)
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (batch, proprio_dim + num_lidar_points*3).  Returns target joint positions."""
        # --- split observation ------------------------------------------------
        proprio = obs[:, : self.proprio_dim]  # (B, 48)
        lidar = obs[:, self.proprio_dim :].reshape(-1, self.num_points, 3)  # (B, 432, 3)

        # --- proximal path (FPS sampling) -------------------------------------
        prox_pts = torch.index_select(lidar, dim=1, index=self.prox_indices)  # (B, 192, 3)
        prox_pts = self._sort_by_spherical(prox_pts)
        prox_enc = self.prox_point_encoder(prox_pts.reshape(-1, 3)).reshape(
            -1, self.prox_points, 64
        )
        _, prox_h = self.prox_spatial_gru(prox_enc)  # h: (1, B, 187)
        prox_feat = prox_h.squeeze(0)  # (B, 187)

        # --- distal path (average down-sampling) ------------------------------
        dist_pts = torch.index_select(lidar, dim=1, index=self.dist_sorted_indices)  # (B, M, 3)
        M = int(dist_pts.shape[1])
        K = self.dist_points  # 56
        out = torch.zeros(dist_pts.shape[0], K, 3, device=obs.device, dtype=obs.dtype)
        scatter_idx = self.dist_bin_ids[:M].view(1, M, 1).expand(dist_pts.shape[0], M, 3)
        out.scatter_add_(1, scatter_idx, dist_pts)
        dist_pts = out / self.dist_bin_counts[:K].clamp(min=1.0).view(1, K, 1)
        dist_pts = self._sort_by_spherical(dist_pts)
        dist_enc = self.dist_point_encoder(dist_pts.reshape(-1, 3)).reshape(
            -1, K, 64
        )
        _, dist_h = self.dist_spatial_gru(dist_enc)  # h: (1, B, 64)
        dist_feat = dist_h.squeeze(0)  # (B, 64)

        # --- temporal memory GRUs (seq_len=1, batch_first=False) --------------
        prox_seq = prox_feat.unsqueeze(0)  # (1, B, 187)
        _, self.prox_hidden = self.prox_memory_gru(prox_seq, self.prox_hidden)
        dist_seq = dist_feat.unsqueeze(0)  # (1, B, 64)
        _, self.dist_hidden = self.dist_memory_gru(dist_seq, self.dist_hidden)

        # --- actor MLP --------------------------------------------------------
        latent = torch.cat(
            [proprio, self.prox_hidden.squeeze(0), self.dist_hidden.squeeze(0)], dim=-1
        )  # (B, 299)
        return self.actor(latent)  # (B, 12)

    @torch.jit.export
    def reset(self):
        self.prox_hidden.zero_()
        self.dist_hidden.zero_()

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "policy_pd_risknet_1.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
