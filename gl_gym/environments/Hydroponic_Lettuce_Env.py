from typing import Any, Dict, List, Optional, Tuple, SupportsFloat

import numpy as np
import casadi as ca
from gymnasium import spaces

from gl_gym.environments.base_env import GreenLightEnv

class HydroponicLettuceEnv(GreenLightEnv):
    """
    Advanced Hydroponic Lettuce Environment for GreenLight Gym
    
    This environment extends the basic LettuceEnv with additional hydroponic-specific
    features including pH and EC control, nutrient management, and solution monitoring.
    """
    
    def __init__(
        self,
        reward_function: str = "GreenhouseReward",  # نوع دالة المكافأة (افتراضي GreenhouseReward)
        observation_modules: List[str] = None,      # وحدات الملاحظة (بتحدد إيه اللي الـ agent بيشوفه من الحالة)
        constraints: Dict[str, Any] = None,         # القيم الدنيا والعليا للقيود (temp, rh, pH, ec …)
        eval_options: Dict[str, Any] = None,        # خيارات التقييم (metrics أو مؤشرات الأداء)
        reward_params: Dict[str, Any] = None,      # بارامترات إضافية للتحكم في شكل المكافأة
        base_env_params: Dict[str, Any] = None,    # إعدادات البيئة الأساسية (مناخ، تحكم …)
        uncertainty_scale: float = 0.0,            # مقياس عدم اليقين (noise في المحاكاة)
        base: Dict[str, Any] = None,               # *EDITED*

        hydroponic_params: Dict[str, Any] = None, # إعدادات خاصة بالهيدروبونيك (نظام NFT/DWC, حجم المحلول …)
        ) -> None:
        
        # Set default values if not provided
        if observation_modules is None:
            observation_modules = [
                "IndoorClimateObservations",    # ملاحظات المناخ الداخلي
                "BasicCropObservations",        # ملاحظات المحصول الأساسية
                "ControlObservations",          # ملاحظات التحكم
                "WeatherObservations",          # ملاحظات الطقس
                "TimeObservations",             # ملاحظات الوقت
                "WeatherForecastObservations"   # ملاحظات توقعات الطقس
            ]
        
        if constraints is None:
            constraints = {
                # ✅ قيود المناخ الداخلي
                "co2_min": 300.0,   # الحد الأدنى لثاني أكسيد الكربون [ppm]
                "co2_max": 1300.0,  # الحد الأقصى لثاني أكسيد الكربون [ppm]
                "temp_min": 15.0,   # الحد الأدنى لدرجة الحرارة [°C]
                "temp_max": 28.0,   # الحد الأقصى لدرجة الحرارة [°C]
                "rh_min": 50.0,     # الحد الأدنى للرطوبة النسبية [%]
                "rh_max": 85.0,     # الحد الأقصى للرطوبة النسبية [%]
                
                # ✅ قيود الهيدروبونيك المتقدمة
                "ph_min": 5.5,             # الحد الأدنى للـ pH
                "ph_max": 6.5,             # الحد الأقصى للـ pH
                "ec_min": 1.2,             # الحد الأدنى للتوصيلية الكهربائية [mS/cm]
                "ec_max": 2.2,             # الحد الأقصى للتوصيلية الكهربائية [mS/cm]
                "solution_temp_min": 18.0, # الحد الأدنى لدرجة حرارة المحلول [°C]
                "solution_temp_max": 24.0  # الحد الأقصى لدرجة حرارة المحلول [°C]
            }
        
        if eval_options is None:
            eval_options = {
                "eval_days": [59],      # الأيام المستخدمة في التقييم
                "eval_years": [2024],   # السنوات المستخدمة في التقييم
                "location": "Egypt"     # موقع الصوبه
            }
        
        if reward_params is None:
            reward_params = {
                
                # ✅ التكاليف الأساسية
                "fixed_greenhouse_cost": 847.89,      # تكلفة ثابتة للصوبه [EGP]
                "fixed_co2_cost": 0.85 ,              # تكلفة CO2 ثابتة [EGP]
                "fixed_lamp_cost":3.96,               # تكلفة الإضاءة [EGP]
                "fixed_screen_cost":113.05,           # تكلفة الستائر [EGP]
                
                # ✅ أسعار الطاقة والمواد
                "elec_price": 16.96,                   # سعر الكهرباء [EGP/kWh]
                "heating_price": 5.09,                 # سعر التدفئة [EGP/kWh]
                "co2_price": 16.96,                    # سعر CO2 [EGP/kg]
                "fruit_price":67.83,                 # سعر المحصول [EGP/kg]
                
                # ✅ معاملات المحصول
                "dmfm": 0.05,    # نسبة المادة الجافة إلى الطازجة
               
               # ✅ أوزان العقوبات 
               "pen_weights": [4.e-4, 5.e-3, 7.e-4], # أوزان العقوبات للمناخ
                "pen_lamp": 0.1,                     # عقوبة استهلاك اللمبات
               
                # ✅ عقوبات الهيدروبونيك
                "ph_violation_penalty": 0.1,            # عقوبة انتهاك pH
                "ec_violation_penalty": 0.1,            # عقوبة انتهاك EC
                "solution_temp_violation_penalty": 0.1  # عقوبة انتهاك درجة حرارة المحلول
            }
        
        if base_env_params is None:
            base_env_params = {
                "weather_data_dir": "gl_gym/environments/weather",
                "location": "Egypt",
                "num_params": 208,      # عدد معاملات النموذج 
                "nx": 30,  # 28 + 2 (pH, EC) 
                "nu": 8,   # 6 + 2 (pH & EC control)
                "nd": 10,  # عدد الاضطرابات المناخية
                "dt": 900, # [s] خطوة زمنية
                "u_min": [0, 0, 0, 0, 0, 0, 0, 0],   # 8 عناصر تحكم
                "u_max": [1, 1, 1, 1, 1, 1, 1, 1],   # 8 عناصر تحكم
                "delta_u_max": 0.1,                  # معدل التغيير الأقصى
                "pred_horizon": 0.5,                 # [أيام] أفق التنبؤ
                "season_length": 60,                 # طول الموسم (بالأيام)
                "start_train_year": 2024,            # بداية فترة التدريب
                "end_train_year": 2024,              # نهاية فترة التدريب
                "start_train_day": 59,               # بداية فترة التدريب (يوم في السنة)
                "end_train_day": 59,                 # نهاية فترة التدريب (يوم في السنة)
                "training": True                     # وضع التدريب
            }
        
        if hydroponic_params is None:
            hydroponic_params = {
               
                # ✅ نوع النظام
                "system_type": "NFT", # "NFT" or "DWC"  or "Aeroponics"  # Nutrient Film Technique
               
                # ✅ الأهداف المثالية
                "target_ph": 6.0,      # Optimal pH for lettuce    # درجة الحموضة المثالية
                "target_ec": 1.4,      # Optimal EC in mS/cm       # التوصيلية الكهربائية المثالية [mS/cm]
               
                # ✅ نطاقات التسامح
                "ph_tolerance": 0.5,   # pH tolerance range
                "ec_tolerance": 0.2,   # EC tolerance range
                
                # ✅ درجة حرارة المحلول
                "solution_temp_min": 18.0,  # Minimum solution temperature
                "solution_temp_max": 24.0,  # Maximum solution temperature
               
                # ✅ معدلات التدفق والتهوية
                "nutrient_flow_rate": 2.0,  # L/min per plant
                "aeration_rate": 1.0,       # L/min per plant
               
               # ✅ إعدادات النظام
                "recirculation": True,      # Enable nutrient recirculation
                "ph_control": True,         # Enable pH control
                "ec_control": True,         # Enable EC control
                
                # ✅ مزيج المغذيات الأساسية [ppm]
                "nutrient_mix": {
                    "N": 150,   # Nitrogen (ppm)
                    "P": 50,    # Phosphorus (ppm)
                    "K": 200,   # Potassium (ppm)
                    "Ca": 150,  # Calcium (ppm)
                    "Mg": 50,   # Magnesium (ppm)
                    "S": 100    # Sulfur (ppm)
                },
                
                # ✅ المغذيات الدقيقة [ppm]
                "micronutrients": {
                    "Fe": 2.0,  # Iron (ppm)
                    "Mn": 0.5,  # Manganese (ppm)
                    "Zn": 0.1,  # Zinc (ppm)
                    "Cu": 0.05, # Copper (ppm)
                    "B": 0.3,   # Boron (ppm)
                    "Mo": 0.05  # Molybdenum (ppm)
                }
            }
        
        # Remove hydroponic-specific penalties from reward_params before passing to parent
        filtered_reward_params = reward_params.copy()
        for k in ["ph_violation_penalty", "ec_violation_penalty", "solution_temp_violation_penalty"]:
            filtered_reward_params.pop(k, None)

        # Call parent constructor with required arguments
        super().__init__(
            weather_data_dir=base_env_params["weather_data_dir"],
            location=base_env_params["location"],
            num_params=base_env_params["num_params"],    # عدد معاملات النموذج
            nx=base_env_params["nx"],                    # 28 + 2 (pH, EC)
            nu=base_env_params["nu"],                    # 6 + 2 (تحكم في pH & EC)
            nd=base_env_params["nd"],                    # عدد الاضطرابات المناخية
            dt=base_env_params["dt"],                    # [s] خطوة زمنية
            u_min=base_env_params["u_min"],              # 8 عناصر تحكم
            u_max=base_env_params["u_max"],              # 8 عناصر تحكم
            delta_u_max=base_env_params["delta_u_max"],  # معدل التغيير الأقصى
            pred_horizon=base_env_params["pred_horizon"], # [أيام] أفق التنبؤ
            )
        # 🟢 controllers
        self.controllers = {
            "lamp": 0.0,
            "ventilation": 0.0,
            "co2_dosing": 0.0,
            "screen": 0.0,
            "heating": 0.0,
            "irrigation": 0.0,
            "ph_control": 0.0,
            "ec_control": 0.0,
        }

        # 🟢 init hydroponic + economics state
        self.nutrient_levels = hydroponic_params["nutrient_mix"].copy()
        self.micronutrient_levels = hydroponic_params["micronutrients"].copy()
        self.solution_temperature = np.mean([
            hydroponic_params["solution_temp_min"], 
            hydroponic_params["solution_temp_max"]
        ])

        self.flow_rate = hydroponic_params["nutrient_flow_rate"]
        self.aeration_rate = hydroponic_params["aeration_rate"]

        # القيم الاقتصادية
        self.revenue = 0.0
        self.heat_costs = 0.0
        self.co2_costs = 0.0
        self.elec_costs = 0.0
        self.irrigation_volume = 0.0
        self.nutrient_costs = 0.0
        self.water_costs = 0.0
        self.pump_energy_costs = 0.0

        # 🟢 env params
        self.base_env_params = base_env_params
        self.max_episode_days = base_env_params["season_length"]
        self.days_passed = 0
        self.episode = 0

        self.hydroponic_params = hydroponic_params
        self._init_hydroponic_state()
        self.reward_params = reward_params
        self.constraints = constraints

        # 🟢 spaces
        # Define observation space using actual observation shape
        obs = self._get_observation()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array(base_env_params["u_min"], dtype=np.float32),
            high=np.array(base_env_params["u_max"], dtype=np.float32),
            dtype=np.float32
        )

        # 🟢 extra params
        self.start_day = base_env_params.get("start_train_day", 1)
        self.growth_year = base_env_params.get("start_train_year", 2024)

        # 🟢 track last action
        self.u = np.zeros(self.action_space.shape, dtype=np.float32)

        # Initialize constraint bounds
        self.constraints_low = [
            constraints["co2_min"],
            constraints["temp_min"],
            constraints["rh_min"],
            constraints["ph_min"],
            constraints["ec_min"],
            constraints["solution_temp_min"]
        ]
        self.constraints_high = [
            constraints["co2_max"],
            constraints["temp_max"],
            constraints["rh_max"],
            constraints["ph_max"],
            constraints["ec_max"],
            constraints["solution_temp_max"]
        ]

        print("🚀 HydroponicLettuceEnv initialized successfully!")
        print(f"🌱 System: {self.hydroponic_params['system_type']}")
        print(f"💧 pH Range: {constraints['ph_min']:.1f} - {constraints['ph_max']:.1f}")
        print(f"⚡ EC Range: {constraints['ec_min']:.1f} - {constraints['ec_max']:.1f} mS/cm")
        print(f"🌡️ Solution Temp: {constraints['solution_temp_min']:.1f} - {constraints['solution_temp_max']:.1f}°C")


    def _init_hydroponic_state(self) -> None:
        """Initialize hydroponic-specific state variables"""
        # Ensure self.x exists
        if not hasattr(self, "x"):
            self.x = np.zeros(30)  # fallback initialization

        # Add pH and EC to the state vector if space allows
        if len(self.x) >= 30:
            # pH and EC are already in the state vector
            self.current_ph = 6.0
            self.current_ec = 1.4
        else:
            # Extend state vector to include pH and EC
            self.x = np.append(self.x, [6.0, 1.4])  # Add pH and EC
            self.current_ph = 6.0
            self.current_ec = 1.4
        
        # Initialize hydroponic monitoring variables
        self.nutrient_levels = self.hydroponic_params.get("nutrient_mix", {}).copy()
        self.micronutrient_levels = self.hydroponic_params.get("micronutrients", {}).copy()
        self.solution_temperature = 21.0  # Default solution temperature
        self.oxygen_level = 8.0  # mg/L
        self.flow_rate = self.hydroponic_params.get("nutrient_flow_rate", 2.0)
        self.aeration_rate = self.hydroponic_params.get("aeration_rate", 1.0)

    
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one step in the hydroponic environment"""

        # -------- تحديث الحالة الداخلية --------
        self._update_hydroponic_state(action)
        self.days_passed += 1  

        # تحديث الكنترولرز من الـ action
        self.controllers.update({
            "lamp": float(action[0]),
            "ventilation": float(action[1]),
            "co2_dosing": float(action[2]),
            "screen": float(action[3]),
            "heating": float(action[4]),
            "irrigation": float(action[5]),
            "ph_control": float(action[6]),
            "ec_control": float(action[7]),
        })

        # -------- حسابات رئيسية --------
        biomass = self._calculate_biomass()
        reward = float(self._calculate_reward())
        terminated = self._check_termination()
        truncated = self.days_passed >= self.max_episode_days

        # -------- تجميع الـ info dict --------
        info = {
           
            # 🌱 Hydroponics
            "current_ph": float(self.current_ph),
            "current_ec": float(self.current_ec),
           
            # 🧪 المحلول المغذي          
            "solution_ph": self.current_ph,
            "solution_ec": self.current_ec,
            "solution_temperature": self.solution_temperature,
            "oxygen_level": self.oxygen_level,
            "flow_rate": self.flow_rate,
            "aeration_rate": self.aeration_rate,
            "nutrient_levels": self.nutrient_levels.copy(),
            "micronutrient_levels": self.micronutrient_levels.copy(),

           
            # 🌍 المناخ الداخلي والخارجي
            "co2_air": getattr(self, "co2_air", None),
            "temp_air": getattr(self, "temp_air", None),
            "rh_air": getattr(self, "rh_air", None),
            "pipe_temp": getattr(self, "pipe_temp", None),
            "CanTemp24": getattr(self, "CanTemp24", None),
            "cFruit": getattr(self, "cFruit", None),
            "tSum": getattr(self, "tSum", None),
            "glob_rad": getattr(self, "glob_rad", None),
            "temp_out": getattr(self, "temp_out", None),
            "rh_out": getattr(self, "rh_out", None),
            "co2_out": getattr(self, "co2_out", None),
            "wind_speed": getattr(self, "wind_speed", None),

            # ⏱️ الزمن
            "timestep": self.days_passed,
            "day_of_year_sin": getattr(self, "day_of_year_sin", None),
            "day_of_year_cos": getattr(self, "day_of_year_cos", None),
            "hour_of_day_sin": getattr(self, "hour_of_day_sin", None),
            "hour_of_day_cos": getattr(self, "hour_of_day_cos", None),

            

            # 🧑‍🔬 المغذيات الكبرى
            "nutrient_N": self.nutrient_levels.get("N", None),
            "nutrient_P": self.nutrient_levels.get("P", None),
            "nutrient_K": self.nutrient_levels.get("K", None),
            "nutrient_Ca": self.nutrient_levels.get("Ca", None),
            "nutrient_Mg": self.nutrient_levels.get("Mg", None),
            "nutrient_S": self.nutrient_levels.get("S", None),

            # 🧑‍🔬 العناصر الصغرى
            "Fe": self.micronutrient_levels.get("Fe", None),
            "Mn": self.micronutrient_levels.get("Mn", None),
            "Zn": self.micronutrient_levels.get("Zn", None),
            "Cu": self.micronutrient_levels.get("Cu", None),
            "B": self.micronutrient_levels.get("B", None),
            "Mo": self.micronutrient_levels.get("Mo", None),

            # 🚰 استهلاك وإدارة
            "irrigation_volume": getattr(self, "irrigation_volume", None),
            "nutrient_costs": getattr(self, "nutrient_costs", None),
            "water_costs": getattr(self, "water_costs", None),
            "pump_energy_costs": getattr(self, "pump_energy_costs", None),

            # 📊 اقتصاديات عامة
            "Rewards": reward,
            "EPI": float(getattr(self, "EPI", 0.0)),
            "revenue": getattr(self, "revenue", None),
            "heat_costs": getattr(self, "heat_costs", None),
            "co2_costs": getattr(self, "co2_costs", None),
            "elec_costs": getattr(self, "elec_costs", None),
            "variable_costs": float(getattr(self, "variable_costs", 0.0)),  # 👈 جديد
            "fixed_costs": float(getattr(self, "fixed_costs", 0.0)),
            "co2_cost": float(getattr(self, "co2_cost", 0.0)),
            "heat_cost": float(getattr(self, "heat_cost", 0.0)),
            "elec_cost": float(getattr(self, "elec_cost", 0.0)),


            


            # 🚨 مؤشرات الانحراف
            "ph_violation": self._check_ph_violation(),
            "ec_violation": self._check_ec_violation(),
            "co2_violation": self._check_co2_violation(),
            "rh_violation": self._check_rh_violation(),
            "lamp_violation": self._check_lamp_violation() if hasattr(self, "_check_lamp_violation") else 0.0,
            "temp_violation": self._check_temp_violation() if hasattr(self, "_check_temp_violation") else 0.0,
            # 🌱 إنتاجية
            "biomass": biomass,

            # 🎛️ Controllers
            "controllers": self.controllers.copy(),
            "uLamp": self.controllers["lamp"],
            "uVent": self.controllers["ventilation"],
            "uCO2": self.controllers["co2_dosing"],
            "uScr": self.controllers["screen"],
            "uHeat": self.controllers["heating"],
            "uIrr": self.controllers["irrigation"],
            "uPH": self.controllers["ph_control"],
            "uEC": self.controllers["ec_control"],

            # رقم الحلقة
            "episode": {
            "r": float(reward),              # cumulative reward (ممكن تخليه episode_reward لو بتحسبه)
            "l": int(self.days_passed)       # طول الحلقة (عدد الـ timesteps)
                }
                
   
        }

        # -------- الحصول على observation --------
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info


    def _check_temp_violation(self):
        return 0.0



    

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # إعادة تهيئة الحالة الهيدروبونيك
        self._init_hydroponic_state()
        self.days_passed = 0  

        # Reset controllers
        self.controllers = {k: 0.0 for k in self.controllers}

        # نجيب الـ observation
        obs = self._get_observation()

        # ممكن نحط info أساسي زي بداية الـ episode
        info = {
            "episode": getattr(self, "episode", 0),
            "timestep": self.days_passed,
        }

        return obs, info


    def _update_hydroponic_state(self, action: np.ndarray) -> None:
        """Update hydroponic-specific state variables"""
        # Update pH and EC based on control actions
        if len(action) >= 8:
            # pH control (action[6])
            ph_change = action[6] * 0.1  # pH change per step
            self.current_ph = np.clip(self.current_ph + ph_change, 5.0, 7.0)
            
            # EC control (action[7])
            ec_change = action[7] * 0.1  # EC change per step
            self.current_ec = np.clip(self.current_ec + ec_change, 0.5, 3.0)
        
        # Update solution temperature based on environmental conditions
        if hasattr(self, 'x') and len(self.x) >= 2:
            air_temp = self.x[2]  # Air temperature
            # Solution temperature follows air temperature with some lag
            temp_diff = air_temp - self.solution_temperature
            self.solution_temperature += temp_diff * 0.01
        
        # Update nutrient levels (simplified model)
        self._update_nutrient_levels()
        
        # Update oxygen level based on aeration
        self._update_oxygen_level()

    def _update_nutrient_levels(self) -> None:
        """Update nutrient levels based on plant uptake and replenishment"""
        # Simplified nutrient uptake model
        uptake_rate = 0.001  # Nutrient uptake rate per step
        
        for nutrient in self.nutrient_levels:
            if self.nutrient_levels[nutrient] > 0:
                # Plant uptake
                self.nutrient_levels[nutrient] -= uptake_rate
                # Ensure non-negative
                self.nutrient_levels[nutrient] = max(0, self.nutrient_levels[nutrient])

    def _update_oxygen_level(self) -> None:
        """Update oxygen level based on aeration and plant consumption"""
        # Oxygen consumption by plants
        consumption_rate = 0.01  # mg/L per step
        
        # Oxygen addition from aeration
        aeration_efficiency = 0.1
        oxygen_addition = self.aeration_rate * aeration_efficiency
        
        # Update oxygen level
        self.oxygen_level = np.clip(
            self.oxygen_level - consumption_rate + oxygen_addition,
            4.0,  # Minimum oxygen level
            12.0  # Maximum oxygen level
        )

    def _get_hydroponic_info(self) -> Dict[str, Any]:
        """معلومات الهيدروبونيك والاقتصاد والمناخ للخطوة الحالية"""
        info = {
            "current_ph": self.current_ph,
            "current_ec": self.current_ec,
            "solution_temperature": self.solution_temperature,
            "oxygen_level": self.oxygen_level,
            "nutrient_levels": self.nutrient_levels,
            "micronutrient_levels": self.micronutrient_levels,
            "flow_rate": self.flow_rate,
            "aeration_rate": self.aeration_rate,
            "ph_violation": self._check_ph_violation(),
            "ec_violation": self._check_ec_violation(),
            "solution_temp_violation": self._check_solution_temp_violation(),
            "nutrient_deficiency": self._check_nutrient_deficiency(),
            "EPI": self._calculate_epi(),
            "revenue": self._calculate_revenue(),

            # انتهاكات المناخ والهيدروبونيك
            "lamp_violation": self._check_lamp_violation(),
            "temp_violation": self.solution_temperature > self.constraints.get("solution_temp_max", 24.0),
            "co2_violation": self._check_co2_violation(),
            "rh_violation": self._check_rh_violation(),

            # التكاليف والدخل
            "variable_costs": self._calculate_variable_costs(),
            "fixed_costs": self._calculate_fixed_costs(),
            "total_costs": self._calculate_total_costs(),
            "income": self._calculate_income(),
            "co2_cost": self._calculate_co2_cost(),
            "lamp_cost": self._calculate_lamp_cost(),
            "screen_cost": self._calculate_screen_cost(),
            "heating_cost": self._calculate_heating_cost(),
            "electricity_cost": self._calculate_electricity_cost(),
            "fruit_income": self._calculate_fruit_income(),
            "biomass": self._calculate_biomass(),
            "heat_cost": self._calculate_heating_cost(),
            "elec_cost": self._calculate_electricity_cost(),
        }
        return info

    def _get_observation(self) -> np.ndarray:
        """
        Returns the current observation vector for the agent.
        عدل ترتيب وقيم العناصر حسب ما يناسب حالتك!
        """
        obs = [
            self.current_ph,
            self.current_ec,
            self.solution_temperature,
            self.oxygen_level,
            self.flow_rate,
            self.aeration_rate,
        ]
        # أضف العناصر الغذائية الأساسية
        obs += [self.nutrient_levels.get(n, 0.0) for n in ["N", "P", "K", "Ca", "Mg", "S"]]
        # أضف المغذيات الدقيقة
        obs += [self.micronutrient_levels.get(m, 0.0) for m in ["Fe", "Mn", "Zn", "Cu", "B", "Mo"]]
        # يمكنك إضافة متغيرات أخرى حسب الحاجة
        return np.array(obs, dtype=np.float32)

    def get_obs_names(self):
        """
        """
        obs_names = [
        "current_ph",
        "current_ec",
        "solution_temperature",
        "oxygen_level",
        "flow_rate",
        "aeration_rate",
        "N", "P", "K", "Ca", "Mg", "S",
        "Fe", "Mn", "Zn", "Cu", "B", "Mo"
    ]
        #for module in self.observation_modules:
         #   obs_names.extend(module.obs_names)
        return obs_names

    # ------------------------------
    #  Calculation functions (primary logic, customizable)
    #  ------------------------------


   

    def _calculate_epi(self):
        # مؤشر الأداء البيئي: عدد الانتهاكات في الخطوة الحالية
        return int(self._check_ph_violation() or self._check_ec_violation() or self._check_solution_temp_violation())

    def _calculate_revenue(self):
        # الإيرادات = الكتلة الحيوية المنتجة * سعر البيع
        biomass = self._calculate_biomass()
        fruit_price = self.reward_params.get("fruit_price", 1.2)
        return biomass * fruit_price

    def _check_lamp_violation(self):
        # مثال: إذا تجاوز استهلاك اللمبات حد معين (غير مفعل هنا)
        return False  # عدلها إذا كان لديك منطق استهلاك اللمبات

    def _check_co2_violation(self):
        # تحقق من تجاوز تركيز CO2 للحدود
        co2 = getattr(self, "co2", 400.0)  # يجب تحديث co2 في كل خطوة
        co2_min = self.constraints.get("co2_min", 300.0)
        co2_max = self.constraints.get("co2_max", 1300.0)
        return co2 < co2_min or co2 > co2_max

    def _check_rh_violation(self):
        # تحقق من تجاوز الرطوبة النسبية للحدود
        rh = getattr(self, "rh", 60.0)  # يجب تحديث rh في كل خطوة
        rh_min = self.constraints.get("rh_min", 50.0)
        rh_max = self.constraints.get("rh_max", 85.0)
        return rh < rh_min or rh > rh_max

    def _check_ph_violation(self) -> bool:
        """تحقق هل قيمة pH خارج الحدود المسموحة"""
        ph_min = self.constraints_low[3] if hasattr(self, "constraints_low") else 5.5
        ph_max = self.constraints_high[3] if hasattr(self, "constraints_high") else 6.5
        return self.current_ph < ph_min or self.current_ph > ph_max

    def _check_ec_violation(self) -> bool:
        """تحقق هل قيمة EC خارج الحدود المسموحة"""
        ec_min = self.constraints_low[4] if hasattr(self, "constraints_low") else 1.2
        ec_max = self.constraints_high[4] if hasattr(self, "constraints_high") else 2.2
        return self.current_ec < ec_min or self.current_ec > ec_max

    def _check_solution_temp_violation(self) -> bool:
        """تحقق هل درجة حرارة المحلول خارج الحدود المسموحة"""
        temp_min = self.constraints_low[5] if hasattr(self, "constraints_low") else 18.0
        temp_max = self.constraints_high[5] if hasattr(self, "constraints_high") else 24.0
        return self.solution_temperature < temp_min or self.solution_temperature > temp_max

    def _check_nutrient_deficiency(self) -> bool:
        """
        تحقق هل يوجد نقص في أي عنصر غذائي أساسي أو دقيق.
        يمكنك تعديل الحدود حسب حالتك.
        """
        # حدود افتراضية للنقص
        min_levels = {
            "N": 20, "P": 10, "K": 30, "Ca": 20, "Mg": 10, "S": 15,
            "Fe": 0.2, "Mn": 0.1, "Zn": 0.05, "Cu": 0.02, "B": 0.05, "Mo": 0.01
        }
        # تحقق من العناصر الأساسية
        for n, min_val in min_levels.items():
            if n in self.nutrient_levels and self.nutrient_levels[n] < min_val:
                return True
            if n in self.micronutrient_levels and self.micronutrient_levels[n] < min_val:
                return True
        return False

    def _calculate_variable_costs(self):
        # التكاليف المتغيرة: كهرباء + تدفئة + CO2 + أسمدة (مثال)
        return (
            self._calculate_electricity_cost() +
            self._calculate_heating_cost() +
            self._calculate_co2_cost()
            # أضف تكلفة الأسمدة إذا أردت
        )

    def _calculate_fixed_costs(self):
        # التكاليف الثابتة من ملف الإعدادات
        return (
            self.reward_params.get("fixed_greenhouse_cost", 15.0) +
            self.reward_params.get("fixed_co2_cost", 0.015) +
            self.reward_params.get("fixed_lamp_cost", 0.07) +
            self.reward_params.get("fixed_screen_cost", 2.0)
        )

    def _calculate_total_costs(self):
        # مجموع التكاليف
        return self._calculate_variable_costs() + self._calculate_fixed_costs()

    def _calculate_income(self):
        # الدخل من بيع المحصول
        return self._calculate_fruit_income()

    def _calculate_co2_cost(self):
        # تكلفة CO2 = كمية CO2 المستخدمة * سعر CO2
        co2_used = getattr(self, "co2_used", 0.0)  # يجب تحديثه في كل خطوة
        co2_price = self.reward_params.get("co2_price", 0.3)
        return co2_used * co2_price

    def _calculate_lamp_cost(self):
        # تكلفة الإضاءة = ساعات تشغيل اللمبات * تكلفة الساعة
        lamp_hours = getattr(self, "lamp_hours", 0.0)  # يجب تحديثه في كل خطوة
        lamp_cost = self.reward_params.get("fixed_lamp_cost", 0.07)
        return lamp_hours * lamp_cost

    def _calculate_screen_cost(self):
        # تكلفة الستائر (ثابتة أو حسب الاستخدام)
        return self.reward_params.get("fixed_screen_cost", 2.0)

    def _calculate_heating_cost(self):
        # تكلفة التدفئة = كمية الطاقة الحرارية * سعر التدفئة
        heating_used = getattr(self, "heating_used", 0.0)  # يجب تحديثه في كل خطوة
        heating_price = self.reward_params.get("heating_price", 0.09)
        return heating_used * heating_price

    def _calculate_electricity_cost(self):
        # تكلفة الكهرباء = كمية الكهرباء المستخدمة * سعر الكهرباء
        electricity_used = getattr(self, "electricity_used", 0.0)  # يجب تحديثه في كل خطوة
        elec_price = self.reward_params.get("elec_price", 0.3)
        return electricity_used * elec_price

    def _calculate_fruit_income(self):
        # دخل بيع الثمار = الكتلة الحيوية المنتجة * سعر البيع
        biomass = self._calculate_biomass()
        fruit_price = self.reward_params.get("fruit_price", 1.2)
        return biomass * fruit_price

    def _calculate_biomass(self):
        # الكتلة الحيوية = دالة بسيطة على الوقت أو الحالة
        # يمكنك تعديلها لتكون أكثر واقعية حسب نمو النبات في نظامك
        days = getattr(self, "days_passed", 0)
        growth_rate = 0.05  # kg/day لكل نبات (افتراضي)
        return days * growth_rate

    def _calculate_reward(self) -> float:
        """
        حساب المكافأة للخطوة الحالية.
        - الإيرادات من بيع المحصول
        - ناقص التكاليف (ثابتة + متغيرة)
        - ناقص العقوبات لو في violations
        """

        # ✅ الإيرادات
        revenue = self._calculate_revenue()

        # ✅ التكاليف
        variable_costs = self._calculate_variable_costs()
        fixed_costs = self._calculate_fixed_costs()
        total_costs = variable_costs + fixed_costs

        # ✅ العقوبات (violations penalties)
        penalties = 0.0

        if self._check_ph_violation():
            penalties += self.reward_params.get("ph_violation_penalty", 0.1)
        if self._check_ec_violation():
            penalties += self.reward_params.get("ec_violation_penalty", 0.1)
        if self._check_solution_temp_violation():
            penalties += self.reward_params.get("solution_temp_violation_penalty", 0.1)
        if self._check_co2_violation():
            penalties += self.reward_params.get("pen_weights", [0.001])[0]
        if self._check_rh_violation():
            penalties += self.reward_params.get("pen_weights", [0.0, 0.001])[1]
        if self._check_temp_violation():
            penalties += self.reward_params.get("pen_weights", [0.0, 0.0, 0.001])[2]

        # ✅ المكافأة = الإيرادات - التكاليف - العقوبات
        reward = revenue - total_costs - penalties

        return reward

    def _check_termination(self) -> bool:
        """
        تحقق هل يجب إنهاء الحلقة (مثلاً عند انتهاء الموسم أو عدد الخطوات).
        يمكنك تعديل المنطق حسب حالتك.
        """
        # مثال: انتهاء عدد الأيام أو الخطوات
        
        # max_days =   #60
        # days_passed = getattr(self, "days_passed", 1)
        # return days_passed >= max_days

        max_days = self.base_env_params["season_length"]
        return self.days_passed >= max_days

