

from typing import Any, Dict, List, Optional, Tuple, SupportsFloat

import numpy as np
import casadi as ca
from gymnasium import spaces

from gl_gym.environments.base_env import GreenLightEnv
from gl_gym.environments.observations import *
from gl_gym.environments.rewards import BaseReward, GreenhouseReward
# from gl_gym.environments.models.greenlight_model import GreenLight

from gl_gym.environments.models.utils import define_model

from gl_gym.environments.utils import load_weather_data, init_state
from gl_gym.environments.parameters import init_default_params
from gl_gym.environments.noise import parametric_crop_uncertainty

REWARDS = {"GreenhouseReward": GreenhouseReward}

OBSERVATION_MODULES = {
    "StateObservations": StateObservations,
    "IndoorClimateObservations": IndoorClimateObservations,
    "BasicCropObservations": BasicCropObservations,
    "ControlObservations": ControlObservations,
    "WeatherObservations": WeatherObservations,
    "WeatherForecastObservations": WeatherForecastObservations,
    "TimeObservations": TimeObservations
}

class LettuceEnv(GreenLightEnv):
    """
    Hydroponic Lettuce Environment for GreenLight Gym
    
    This environment is specifically configured for hydroponic lettuce cultivation
    with soil parameters disabled and hydroponic-specific optimizations.
    """
    
    def __init__(
        self,
        reward_function: str,                   # reward function
        observation_modules: List[str],         # observation function
        constraints: Dict[str, Any],            # constraints for the environment
        eval_options: Dict[str, Any],           # days for evaluation
        reward_params: Dict[str, Any] = {},     # reward function arguments
        base_env_params: Dict[str, Any] = {},   # base environment parameters
        uncertainty_scale: float = 0.0,
        hydroponic_params: Dict[str, Any] = None  # hydroponic-specific parameters
        ) -> None:
        
        # Call parent constructor first
        super().__init__(
            reward_function=reward_function,
            observation_modules=observation_modules,
            constraints=constraints,
            eval_options=eval_options,
            reward_params=reward_params,
            base_env_params=base_env_params,
            uncertainty_scale=uncertainty_scale
        )
        
        # Initialize hydroponic parameters
        self.hydroponic_params = hydroponic_params or self._get_default_hydroponic_params()
        
        # =======================================================
        # 🔹 تعديل مهم: إلغاء معاملات التربة Soil → Hydroponic
        # =======================================================
        self._disable_soil_parameters()
        
        # Initialize hydroponic-specific parameters
        self._init_hydroponic_parameters()
        
        print("✅ LettuceEnv running in Hydroponic mode (soil disabled).")
        print(f"🌱 Hydroponic system: {self.hydroponic_params['system_type']}")
        print(f"💧 Target pH: {self.hydroponic_params['target_ph']:.1f}")
        print(f"⚡ Target EC: {self.hydroponic_params['target_ec']:.1f} mS/cm")

        # Initialize observation and action spaces
        self.observation_modules = self._init_observations(observation_modules)
        self.observation_space = self._generate_observation_space()
        self.action_space = self._generate_action_space()

        # Initialize the model
        self.F = define_model(
            nx=self.nx,
            nu=self.nu,
            nd=self.nd,
            n_params=self.num_params,
            dt=self.dt,
        )

        # Set constraints
        self.constraints_low = np.array([
            constraints["co2_min"],
            constraints["temp_min"],
            constraints["rh_min"],
        ])

        self.constraints_high = np.array([
            constraints["co2_max"],
            constraints["temp_max"],
            constraints["rh_max"],
        ])

        # Initialize the reward function
        self.reward = self._init_rewards(reward_function, reward_params)

    def _terminalState(self) -> bool:
        """
        Function that checks whether the simulation has reached a terminal state.
        Terminal states are reached when the simulation has reached the end of the growing season.
        """
        if self.timestep >= self.N:
            return True
        return False

    def _init_observations(
        self,
        observation_modules: List[str],
    ) -> List[BaseObservations]:
        return [OBSERVATION_MODULES[module](self) for module in observation_modules]

    def _generate_observation_space(self) -> spaces.Box:
        spaces_low_list = []
        spaces_high_list = []

        for module in self.observation_modules:
            module_obs_space = module.observation_space()
            spaces_low_list.append(module_obs_space.low)
            spaces_high_list.append(module_obs_space.high)

        low = np.concatenate(spaces_low_list, axis=0)  # Concatenate low bounds
        high = np.concatenate(spaces_high_list, axis=0)  # Concatenate high bounds

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _generate_action_space(self) -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=(self.nu,), dtype=np.float32)

    def _init_rewards(self, reward_function: str, reward_params: Dict[str, Any]) -> BaseReward:
        return REWARDS[reward_function](self, **reward_params)

    def _get_reward(self) -> SupportsFloat:
        return self.reward.compute_reward()

    def _scale(self, action, action_min, action_max):
        return (action + 1) * (action_max - action_min) / 2 + action_min

    def action_to_control(self, action: np.ndarray) -> np.ndarray:
        """
        Function that converts the action to control inputs.
        """
        return np.clip(self.u + action*self.delta_u_max, self.u_min, self.u_max) 

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        # scale the action from controller (between -1, 1) to (u_min, u_max)
        self.u = self.action_to_control(action)
        params = parametric_crop_uncertainty(self.p, self.uncertainty_scale, self._np_random)
        try:
            p_dyn = ca.vertcat(ca.DM(self.weather_data[self.timestep]), params)
            res = self.F(x0=ca.DM(self.x), u=ca.DM(self.u), p=p_dyn)
            self.x = res["xf"].full().flatten()

            # self.x = self.gl_model.evalF(self.x, self.u, self.weather_data[self.timestep], params)
        except:
            print("Error in ODE approximation")
            self.terminated = True

        # update time
        self.day_of_year += (self.dt/self.c) % 365
        self.hour_of_day +=  (self.dt/3600)
        self.hour_of_day = self.hour_of_day % 24

        self.obs = self._get_obs()
        if self._terminalState():
            self.terminated = True
        # compute reward
        reward = self._get_reward()
        # additional information to return
        info = self._get_info()
        self.timestep += 1
        self.x_prev = np.copy(self.x)

        return (
                self.obs,
                reward,
                self.terminated,
                False,
                info
                )

    def step_raw_control(self, control: np.ndarray):
        self.u = control
        params = parametric_crop_uncertainty(self.p, self.uncertainty_scale, self._np_random)
        p_dyn = ca.vertcat(ca.DM(self.weather_data[self.timestep]), params)
        res = self.F(x0=ca.DM(self.x), u=ca.DM(self.u), p=p_dyn)
        self.x = res["xf"].full().flatten()

        # update time
        self.day_of_year += (self.dt/self.c) % 365
        self.hour_of_day +=  (self.dt/3600)
        self.hour_of_day = self.hour_of_day % 24

        self.obs = self._get_obs()

        if self._terminalState():
            self.terminated = True
        # compute reward
        reward = self._get_reward()
        self.timestep += 1
        self.x_prev = np.copy(self.x)
        # additional information to return
        return (
                self.obs,
                reward,
                self.terminated,
                False,
                self._get_info()
                )

    def step_raw_control_pipeinput(self, control: np.ndarray):
        self.u = control

        self.x = self.F(self.x, self.u, self.weather_data[self.timestep], self.p)
        # self.x = self.gl_model.evalF(self.x, self.u, self.weather_data[self.timestep], self.p)

        if self._terminalState():
            self.terminated = True
        # compute reward
        reward = self._get_reward()

        # additional information to return
        # info = self._get_info()
        self.timestep += 1
        return (
                self.x,
                self.terminated, 
                )

    def _get_obs(self):
        obs = []
        for module in self.observation_modules:
            obs.append(module.compute_obs())
        obs = np.concatenate(obs, axis=0)
        return obs

    def get_obs_names(self):
        """
        """
        obs_names = []
        for module in self.observation_modules:
            obs_names.extend(module.obs_names)
        return obs_names

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state"""
        info = {
            "EPI": self.reward.profit,
            "revenue": self.reward.gains,
            "variable_costs": self.reward.variable_costs,
            "fixed_costs": self.reward.fixed_costs,
            "co2_cost": self.reward.co2_costs,
            "heat_cost": self.reward.heat_costs,
            "elec_cost": self.reward.elec_costs,
            "temp_violation": self.reward.temp_violation,
            "co2_violation": self.reward.co2_violation,
            "rh_violation": self.reward.rh_violation,
            "lamp_violation": self.reward.lamp_violation,
            "controls": self.u,
        }
        
        # Add hydroponic-specific information
        if hasattr(self, 'hydroponic_params'):
            info.update({
                "hydroponic_system": self.hydroponic_params['system_type'],
                "target_ph": self.hydroponic_params['target_ph'],
                "target_ec": self.hydroponic_params['target_ec'],
                "solution_temp_min": self.hydroponic_params['solution_temp_min'],
                "solution_temp_max": self.hydroponic_params['solution_temp_max'],
            })
        
        return info

    def set_crop_state(
        self,
        cBuf: float,
        cLeaf: float,
        cStem: float,
        cFruit: float,
        tCanSum: float
    ) -> None:
        self.x[22] = cBuf
        self.x[23] = cLeaf
        self.x[24] = cStem
        self.x[25] = cFruit
        self.x[26] = tCanSum

    def _get_default_hydroponic_params(self) -> Dict[str, Any]:
        """Get default hydroponic parameters for lettuce cultivation"""
        return {
            "system_type": "NFT",  # Nutrient Film Technique
            "target_ph": 6.0,      # Optimal pH for lettuce
            "target_ec": 1.4,      # Optimal EC in mS/cm
            "ph_tolerance": 0.5,   # pH tolerance range
            "ec_tolerance": 0.2,   # EC tolerance range
            "solution_temp_min": 18.0,  # Minimum solution temperature
            "solution_temp_max": 24.0,  # Maximum solution temperature
            "nutrient_flow_rate": 2.0,  # L/min per plant
            "aeration_rate": 1.0,       # L/min per plant
            "recirculation": True,      # Enable nutrient recirculation
            "ph_control": True,         # Enable pH control
            "ec_control": True,         # Enable EC control
        }

    def _disable_soil_parameters(self) -> None:
        """Disable soil-related parameters for hydroponic operation"""
        # Soil storage nodes
        self.p[22:27] = 0.0   # hSo1..hSo5 (soil storage nodes)
        self.p[27] = 0.0      # hSoOut (soil outflow)
        
        # Soil thermal properties
        self.p[102] = 0.0     # rhoCpSo (soil volumetric heat capacity)
        self.p[103] = 0.0     # lambdaSo (soil thermal conductivity)
        
        # Additional soil parameters that might exist
        soil_param_indices = [28, 29, 30, 31, 32, 33, 34, 35]  # Extended soil params
        for idx in soil_param_indices:
            if idx < len(self.p):
                self.p[idx] = 0.0

    def _init_hydroponic_parameters(self) -> None:
        """Initialize hydroponic-specific parameters"""
        # Add hydroponic parameters to the parameter vector if space allows
        if hasattr(self, 'p') and len(self.p) > 150:
            # Add hydroponic parameters at the end of the parameter vector
            self.hydroponic_param_indices = {
                'ph_target': 150,
                'ec_target': 151,
                'solution_temp_target': 152,
                'nutrient_concentration': 153,
                'oxygen_level': 154,
            }
            
            # Set default values
            self.p[self.hydroponic_param_indices['ph_target']] = self.hydroponic_params['target_ph']
            self.p[self.hydroponic_param_indices['ec_target']] = self.hydroponic_params['target_ec']
            self.p[self.hydroponic_param_indices['solution_temp_target']]  = (self.hydroponic_params['solution_temp_min'] + self.hydroponic_params['solution_temp_max']) / 2
            self.p[self.hydroponic_param_indices['nutrient_concentration']] = 1.0
            self.p[self.hydroponic_param_indices['oxygen_level']] = 8.0  # mg/L

    def get_hydroponic_status(self) -> Dict[str, Any]:
        """Get current hydroponic system status"""
        if not hasattr(self, 'hydroponic_params'):
            return {"error": "Hydroponic parameters not initialized"}
            
        return {
            "system_type": self.hydroponic_params['system_type'],
            "target_ph": self.hydroponic_params['target_ph'],
            "target_ec": self.hydroponic_params['target_ec'],
            "ph_tolerance": self.hydroponic_params['ph_tolerance'],
            "ec_tolerance": self.hydroponic_params['ec_tolerance'],
            "solution_temp_range": [
                self.hydroponic_params['solution_temp_min'],
                self.hydroponic_params['solution_temp_max']
            ],
            "nutrient_flow_rate": self.hydroponic_params['nutrient_flow_rate'],
            "aeration_rate": self.hydroponic_params['aeration_rate'],
            "recirculation_enabled": self.hydroponic_params['recirculation'],
            "ph_control_enabled": self.hydroponic_params['ph_control'],
            "ec_control_enabled": self.hydroponic_params['ec_control'],
        }

    def update_hydroponic_params(self, new_params: Dict[str, Any]) -> None:
        """Update hydroponic parameters during runtime"""
        if hasattr(self, 'hydroponic_params'):
            self.hydroponic_params.update(new_params)
            print(f"✅ Updated hydroponic parameters: {new_params}")
        else:
            print("❌ Hydroponic parameters not initialized")




######################################################################################
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # pick a random growth year and start day if we are training
        if self.training:
            self.growth_year = self._np_random.choice(self.train_years)
            self.start_day = self._np_random.choice(self.train_days)
        else:
            self.growth_year = self._np_random.choice(self.eval_options["eval_years"])
            self.start_day = self._np_random.choice(self.eval_options["eval_days"])
            self.location = self.eval_options["location"]
            self.increase_eval_idx()

        self.day_of_year = self.start_day
        self.hour_of_day = 0

        # load in weather data for specific simulation
        self.weather_data = load_weather_data(
            self.weather_data_dir,
            self.location,
            self.growth_year,
            self.start_day,
            self.season_length,
            self.Np+1,
            self.dt,
            self.nd
        )

        # ===== Add data checking and processing =====
        # Convert NaN/Inf to appropriate values
        self.weather_data = np.nan_to_num(self.weather_data, nan=0.0, posinf=1e6, neginf=-1e6)

        # clip values to logical ranges for each variable (modify by columns in order)
        # Assume columns: [Temperature, Humidity, CO2, Light, ...]
        # Adjust the numbers according to your real data
        self.weather_data[:, 0] = np.clip(self.weather_data[:, 0], 15, 35)   # Temperature
        self.weather_data[:, 1] = np.clip(self.weather_data[:, 1], 0, 100)   # Humidity
        self.weather_data[:, 2] = np.clip(self.weather_data[:, 2], 200, 2000) # CO2
        self.weather_data[:, 3] = np.clip(self.weather_data[:, 3], 0, 1000)   # Light
        # ===== End of processing =====

        self.u = np.zeros(self.nu)
        self.x = init_state(self.weather_data[0])
        self.x_prev = np.copy(self.x)
        self.timestep = 0
        self.obs = self._get_obs()

        self.terminated = False
        return self.obs, {}


