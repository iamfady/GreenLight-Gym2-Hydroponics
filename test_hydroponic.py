#!/usr/bin/env python3
"""
Test script for Hydroponic Lettuce Environment

This script demonstrates the functionality of the hydroponic environment
and shows how to use its features.
"""

import numpy as np
from gl_gym.environments.Hydroponic_Lettuce_Env import HydroponicLettuceEnv

def test_hydroponic_environment():
    """Test the hydroponic environment with various scenarios"""
    
    print("🌱 Testing Hydroponic Lettuce Environment")
    print("=" * 50)
    
    # Create the environment with custom parameters
    env = HydroponicLettuceEnv(
        hydroponic_params={
            "system_type": "DWC",  # Deep Water Culture
            "target_ph": 6.2,
            "target_ec": 1.6,
            "solution_temp_min": 19.0,
            "solution_temp_max": 23.0
        }
    )
    
    print(f"\n✅ Environment created successfully!")
    print(f"🌱 System type: {env.hydroponic_params['system_type']}")
    print(f"💧 Target pH: {env.hydroponic_params['target_ph']}")
    print(f"⚡ Target EC: {env.hydroponic_params['target_ec']}")
    
    # Test initial state
    print("\n📊 Initial State:")
    print(f"Current pH: {env.current_ph:.2f}")
    print(f"Current EC: {env.current_ec:.2f}")
    print(f"Solution Temperature: {env.solution_temperature:.2f}°C")
    print(f"Oxygen Level: {env.oxygen_level:.2f} mg/L")
    
    # Test nutrient levels
    print("\n🧪 Nutrient Levels:")
    for nutrient, level in env.nutrient_levels.items():
        print(f"{nutrient}: {level:.1f} ppm")
    
    # Test system health score
    health_score = env.get_hydroponic_health_score()
    print(f"\n🏥 System Health Score: {health_score:.1f}/100")
    
    # Test recommendations
    print("\n💡 System Recommendations:")
    recommendations = env.get_system_recommendations()
    for rec in recommendations:
        print(f"  {rec}")
    
    # Test pH adjustment
    print("\n🔧 Testing pH Adjustment:")
    env.adjust_ph(6.5)
    print(f"New pH: {env.current_ph:.2f}")
    
    # Test EC adjustment
    print("\n🔧 Testing EC Adjustment:")
    env.adjust_ec(1.8)
    print(f"New EC: {env.current_ec:.2f}")
    
    # Test nutrient addition
    print("\n🧪 Testing Nutrient Addition:")
    env.add_nutrients("N", 50)
    env.add_nutrients("K", 30)
    
    # Test flow rate adjustment
    print("\n💧 Testing Flow Rate Adjustment:")
    env.set_flow_rate(3.0)
    print(f"New flow rate: {env.flow_rate} L/min")
    
    # Test aeration rate adjustment
    print("\n💨 Testing Aeration Rate Adjustment:")
    env.set_aeration_rate(1.5)
    print(f"New aeration rate: {env.aeration_rate} L/min")
    
    # Test a few simulation steps
    print("\n🔄 Testing Simulation Steps:")
    obs = env.reset()
    print(f"Initial observation shape: {obs[0].shape}")
    
    for step in range(5):
        # Random action
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  pH: {env.current_ph:.2f}")
        print(f"  EC: {env.current_ec:.2f}")
        print(f"  Solution Temp: {env.solution_temperature:.2f}°C")
        print(f"  Oxygen: {env.oxygen_level:.2f} mg/L")
        
        if terminated:
            break
    
    # Final health score
    final_health = env.get_hydroponic_health_score()
    print(f"\n🏥 Final System Health Score: {final_health:.1f}/100")
    
    # Final recommendations
    print("\n💡 Final Recommendations:")
    final_recs = env.get_system_recommendations()
    for rec in final_recs:
        print(f"  {rec}")
    
    print("\n✅ Hydroponic environment test completed successfully!")

def test_advanced_features():
    """Test advanced hydroponic features"""
    
    print("\n🚀 Testing Advanced Hydroponic Features")
    print("=" * 50)
    
    # Create environment with advanced parameters
    env = HydroponicLettuceEnv(
        hydroponic_params={
            "system_type": "NFT",  # Nutrient Film Technique
            "target_ph": 6.0,
            "target_ec": 1.4,
            "nutrient_mix": {
                "N": 200, "P": 60, "K": 250,
                "Ca": 180, "Mg": 60, "S": 120
            },
            "micronutrients": {
                "Fe": 3.0, "Mn": 0.8, "Zn": 0.2,
                "Cu": 0.1, "B": 0.5, "Mo": 0.1
            }
        }
    )
    
    print("✅ Advanced environment created!")
    
    # Test nutrient deficiency detection
    print("\n🔍 Testing Nutrient Deficiency Detection:")
    
    # Simulate nutrient depletion
    env.nutrient_levels["N"] = 30  # Below critical level
    env.nutrient_levels["P"] = 15  # Below critical level
    
    deficiencies = env._check_nutrient_deficiency()
    print("Detected deficiencies:")
    for nutrient, is_deficient in deficiencies.items():
        if is_deficient:
            print(f"  🔴 {nutrient}: {env.nutrient_levels[nutrient]:.1f} ppm (Critical!)")
        else:
            print(f"  ✅ {nutrient}: {env.nutrient_levels[nutrient]:.1f} ppm")
    
    # Test constraint violations
    print("\n⚠️ Testing Constraint Violations:")
    
    # Test pH violation
    env.current_ph = 4.5  # Too low
    ph_violation = env._check_ph_violation()
    print(f"pH violation: {ph_violation}")
    
    # Test EC violation
    env.current_ec = 3.5  # Too high
    ec_violation = env._check_ec_violation()
    print(f"EC violation: {ec_violation}")
    
    # Test solution temperature violation
    env.solution_temperature = 15.0  # Too low
    temp_violation = env._check_solution_temp_violation()
    print(f"Temperature violation: {temp_violation}")
    
    # Test oxygen level
    env.oxygen_level = 3.0  # Too low
    print(f"Oxygen level: {env.oxygen_level} mg/L")
    
    # Calculate health score with violations
    health_score = env.get_hydroponic_health_score()
    print(f"\n🏥 Health Score with Violations: {health_score:.1f}/100")
    
    print("\n✅ Advanced features test completed!")

if __name__ == "__main__":
    try:
        test_hydroponic_environment()
        test_advanced_features()
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
