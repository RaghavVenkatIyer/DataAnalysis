import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =====================================================================
# 1. AEROSPACE CORE CALCULATIONS ENGINE (18 PARAMETERS)
# =====================================================================
def compute_aerospace_metrics(df):
    """
    Derives 18 advanced flight metrics using standard physics equations, 
    aerodynamic models, and orbital mechanics.
    """
    # Ensure data is sorted by time
    df = df.sort_values('time').reset_index(drop=True)
    dt = df['time'].diff().fillna(method='bfill')
    
    # --- A. KINEMATICS & MOTION ---
    # 1. Total Acceleration (Derived dynamically in Section 4 if missing)
    # 2. Axial G-Force
    df['g_force'] = df['acceleration'] / 9.80665
    
    # 3. Vertical Velocity
    df['vertical_velocity'] = df['altitude'].diff() / dt
    df['vertical_velocity'] = df['vertical_velocity'].fillna(0)
    
    # 4. Horizontal / Downrange Velocity (Pythagorean theorem)
    df['horizontal_velocity'] = np.sqrt(np.maximum(0, df['velocity']**2 - df['vertical_velocity']**2))
    
    # 5. Flight Path Angle (Pitch) in degrees and radians
    df['flight_path_angle_rad'] = np.arctan2(df['vertical_velocity'], df['horizontal_velocity'])
    df['flight_path_angle'] = np.degrees(df['flight_path_angle_rad'])
    
    # 6. Downrange Distance (Integral of horizontal velocity)
    df['downrange_distance'] = (df['horizontal_velocity'] * dt).cumsum()
    
    # 7. Total Arc Length (Integral of total velocity)
    df['total_arc_length'] = (df['velocity'] * dt).cumsum()

    # --- B. AERODYNAMICS ---
    # 8. Atmospheric Density (ISA model approximation up to 85km)
    df['air_density'] = 1.225 * np.exp(-df['altitude'] / 8500.0)
    
    # 9. Dynamic Pressure (Max-Q)
    df['dynamic_pressure'] = 0.5 * df['air_density'] * df['velocity']**2
    
    # 10. Mach Number
    df['temperature_k'] = np.where(df['altitude'] < 11000, 288.15 - 0.0065 * df['altitude'], 216.65)
    df['speed_of_sound'] = np.sqrt(1.4 * 287.05 * df['temperature_k'])
    df['mach_number'] = df['velocity'] / df['speed_of_sound']
    
    # 11. Aerodynamic Heating Proxy (Stagnation Temperature)
    df['aero_heating_temp_k'] = df['temperature_k'] * (1 + 0.2 * df['mach_number']**2)

    # --- C. ORBITAL MECHANICS ---
    R_E = 6371000.0  # Earth Radius in meters
    mu = 3.986004418e14  # Earth's Standard Gravitational Parameter (G * M_E)
    r_current = R_E + df['altitude']
    
    # 12. Local Gravity Field
    df['local_gravity'] = mu / (r_current**2)
    
    # 13. Specific Kinetic Energy
    df['specific_kinetic_energy'] = 0.5 * df['velocity']**2
    
    # 14. Specific Potential Energy
    df['specific_potential_energy'] = -mu / r_current
    
    # 15. Specific Orbital Energy
    df['specific_orbital_energy'] = df['specific_kinetic_energy'] + df['specific_potential_energy']
    
    # 16. Instantaneous Projected Apogee
    # Uses specific angular momentum (h) and eccentricity (e)
    h_angular = r_current * df['horizontal_velocity']
    eccentricity = np.sqrt(np.maximum(0, 1 + (2 * df['specific_orbital_energy'] * h_angular**2) / (mu**2)))
    
    # Semi-major axis (a). If orbital energy >= 0, it's an escape trajectory (apogee = inf)
    semi_major_axis = np.where(df['specific_orbital_energy'] < 0, -mu / (2 * df['specific_orbital_energy']), np.inf)
    projected_apogee_radius = semi_major_axis * (1 + eccentricity)
    df['projected_apogee'] = projected_apogee_radius - R_E

    # --- D. VEHICLE & PROPULSION DYNAMICS ---
    # 17. Estimated Thrust-to-Weight Ratio (TWR)
    # TWR Proxy = (Acceleration along path + Gravity loss) / Standard Gravity
    gravity_loss = df['local_gravity'] * np.sin(df['flight_path_angle_rad'])
    df['estimated_twr'] = (df['acceleration'] + gravity_loss) / 9.80665
    
    # 18. Fuel Mass Fraction (Using Tsiolkovsky's Rocket Equation)
    # Proxy: delta-v integrated over time. Assuming Average Isp of ~300s (Exhaust velocity Ve = 2940 m/s)
    ideal_dv = (df['acceleration'] * dt).cumsum()
    df['fuel_mass_fraction'] = np.exp(-ideal_dv / 2940.0) * 100 # Converted to percentage

    return df

# =====================================================================
# 2. DELTA-ANALYSIS & COMPARISON ENGINE
# =====================================================================
def perform_delta_analysis(df_base, df_test):
    common_time = df_base['time'].values
    comparison_data = {'time': common_time}
    
    # Compare key performance metrics
    columns_to_compare = [
        'altitude', 'velocity', 'dynamic_pressure', 'mach_number', 
        'projected_apogee', 'total_arc_length', 'estimated_twr', 'fuel_mass_fraction'
    ]
    
    for col in columns_to_compare:
        comparison_data[f'{col}_base'] = df_base[col].values
        comparison_data[f'{col}_test'] = np.interp(common_time, df_test['time'].values, df_test[col].values)
        comparison_data[f'{col}_delta'] = comparison_data[f'{col}_test'] - comparison_data[f'{col}_base']
        
    return pd.DataFrame(comparison_data)

# =====================================================================
# 3. AUTOMATED TECHNICAL REPORT GENERATOR (PDF)
# =====================================================================
def generate_pdf_report(df_base, df_test, df_delta, output_filename):
    summary_metrics = {
        'Actual Apogee Reached (m)': (df_base['altitude'].max(), df_test['altitude'].max()),
        'Max Velocity (m/s)': (df_base['velocity'].max(), df_test['velocity'].max()),
        'Max-Q Dynamic Pressure (Pa)': (df_base['dynamic_pressure'].max(), df_test['dynamic_pressure'].max()),
        'Peak Stagnation Temp (K)': (df_base['aero_heating_temp_k'].max(), df_test['aero_heating_temp_k'].max()),
        'Max Mach Number': (df_base['mach_number'].max(), df_test['mach_number'].max()),
        'Final Projected Apogee (m)': (df_base['projected_apogee'].iloc[-1], df_test['projected_apogee'].iloc[-1]),
    }

    with PdfPages(output_filename) as pdf:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # --- PAGE 1: EXECUTIVE SUMMARY ---
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.5, 0.95, "ETHEREALX FLIGHT DYNAMICS REPORT", fontsize=18, weight='bold', ha='center', color='#1a365d')
        ax.text(0.5, 0.92, "Automated Post-Flight 18-Parameter Evaluation", fontsize=12, style='italic', ha='center', color='#4a5568')
        ax.plot([0.05, 0.95], [0.89, 0.89], color='#1a365d', lw=2, transform=ax.transAxes)
        
        ax.text(0.05, 0.75, "Performance Milestones", fontsize=14, weight='bold', color='#1a365d')
        
        table_data = [["Metric Target", "Baseline Run", "Test Run", "Absolute Delta", "Variance (%)"]]
        for metric, values in summary_metrics.items():
            base, test = values
            delta = test - base
            pct = (delta / base) * 100 if base != 0 else 0
            table_data.append([metric, f"{base:,.2f}", f"{test:,.2f}", f"{delta:+,.2f}", f"{pct:+.2f}%"])
            
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0.05, 0.45, 0.90, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for col_idx in range(5): table[0, col_idx].set_facecolor('#e2e8f0')
        
        pdf.savefig(fig)
        plt.close()

        # --- PAGE 2: CORE KINEMATICS & TRAJECTORY ---
        fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Core Kinematics & Trajectory", fontsize=16, weight='bold')
        
        # Alt vs Downrange
        axs[0,0].plot(df_base['downrange_distance']/1000, df_base['altitude']/1000, label='Base')
        axs[0,0].plot(df_test['downrange_distance']/1000, df_test['altitude']/1000, label='Test', linestyle='--')
        axs[0,0].set_title("Flight Profile (Alt vs Downrange)")
        axs[0,0].set_ylabel("Altitude (km)"); axs[0,0].set_xlabel("Downrange (km)")
        
        # Velocities
        axs[0,1].plot(df_base['time'], df_base['velocity'], label='Total')
        axs[0,1].plot(df_base['time'], df_base['vertical_velocity'], label='Vertical (Base)', alpha=0.6)
        axs[0,1].plot(df_test['time'], df_test['velocity'], label='Test', linestyle='--')
        axs[0,1].set_title("Velocity Vectors")
        axs[0,1].set_ylabel("Velocity (m/s)")
        
        # Flight Path Angle
        axs[1,0].plot(df_base['time'], df_base['flight_path_angle'])
        axs[1,0].plot(df_test['time'], df_test['flight_path_angle'], linestyle='--')
        axs[1,0].set_title("Flight Path Angle (Pitch)")
        axs[1,0].set_ylabel("Degrees")
        
        # Total Arc Length
        axs[1,1].plot(df_base['time'], df_base['total_arc_length']/1000)
        axs[1,1].plot(df_test['time'], df_test['total_arc_length']/1000, linestyle='--')
        axs[1,1].set_title("Total Distance Flown (Arc Length)")
        axs[1,1].set_ylabel("Distance (km)")
        
        axs[0,0].legend(); axs[0,1].legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

        # --- PAGE 3: ORBITAL MECHANICS & PROPULSION ---
        fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Orbital Mechanics & Propulsion Dynamics", fontsize=16, weight='bold')
        
        # Projected Apogee
        axs[0,0].plot(df_base['time'], df_base['projected_apogee']/1000)
        axs[0,0].plot(df_test['time'], df_test['projected_apogee']/1000, linestyle='--')
        axs[0,0].set_title("Instantaneous Projected Apogee")
        axs[0,0].set_ylabel("Altitude (km)")
        
        # Specific Orbital Energy
        axs[0,1].plot(df_base['time'], df_base['specific_orbital_energy']/1e6)
        axs[0,1].plot(df_test['time'], df_test['specific_orbital_energy']/1e6, linestyle='--')
        axs[0,1].set_title("Specific Orbital Energy")
        axs[0,1].set_ylabel("Energy (MJ/kg)")
        
        # Estimated TWR
        axs[1,0].plot(df_base['time'], df_base['estimated_twr'])
        axs[1,0].plot(df_test['time'], df_test['estimated_twr'], linestyle='--')
        axs[1,0].set_title("Estimated Thrust-to-Weight Ratio")
        axs[1,0].set_ylabel("TWR")
        
        # Fuel Mass Fraction
        axs[1,1].plot(df_base['time'], df_base['fuel_mass_fraction'])
        axs[1,1].plot(df_test['time'], df_test['fuel_mass_fraction'], linestyle='--')
        axs[1,1].set_title("Estimated Fuel Mass Fraction")
        axs[1,1].set_ylabel("Percentage Remaining (%)")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

        # --- PAGE 4: AERODYNAMICS ---
        fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Aerodynamics & Atmospheric Heating", fontsize=16, weight='bold')
        
        # Dynamic Pressure (Q)
        axs[0,0].plot(df_base['time'], df_base['dynamic_pressure'])
        axs[0,0].plot(df_test['time'], df_test['dynamic_pressure'], linestyle='--')
        axs[0,0].set_title("Dynamic Pressure (Max-Q)")
        axs[0,0].set_ylabel("Pascals")
        
        # Mach Number
        axs[0,1].plot(df_base['time'], df_base['mach_number'])
        axs[0,1].plot(df_test['time'], df_test['mach_number'], linestyle='--')
        axs[0,1].set_title("Mach Number")
        axs[0,1].set_ylabel("Mach")
        
        # Aero Heating
        axs[1,0].plot(df_base['time'], df_base['aero_heating_temp_k'])
        axs[1,0].plot(df_test['time'], df_test['aero_heating_temp_k'], linestyle='--')
        axs[1,0].set_title("Stagnation Heating Proxy")
        axs[1,0].set_ylabel("Temperature (K)")
        
        # Gravity
        axs[1,1].plot(df_base['time'], df_base['local_gravity'])
        axs[1,1].plot(df_test['time'], df_test['local_gravity'], linestyle='--')
        axs[1,1].set_title("Local Gravity Field Decline")
        axs[1,1].set_ylabel("Gravity (m/s²)")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

# =====================================================================
# 4. INTERACTIVE FILE UPLOAD ENVIRONMENT (GOOGLE COLAB)
# =====================================================================
if __name__ == '__main__':
    try:
        from google.colab import files
        
        print("🚀 STEP 1: Please upload the BASELINE trajectory JSON file.")
        uploaded_base = files.upload()
        base_filename = list(uploaded_base.keys())[0]
        with open(base_filename, 'r') as file1:
            baseline_raw = json.load(file1)
            
        print("\n🚀 STEP 2: Please upload the TEST RUN trajectory JSON file.")
        uploaded_test = files.upload()
        test_filename = list(uploaded_test.keys())[0]
        with open(test_filename, 'r') as file2:
            test_raw = json.load(file2)
            
        # Extract the nested data and convert to DataFrames
        df_b_raw = pd.DataFrame(baseline_raw['telemetry_data'])
        df_t_raw = pd.DataFrame(test_raw['telemetry_data'])
        
        # Rename your specific JSON keys to match the physics engine
        rename_map = {
            'time_s': 'time',
            'velocity_ms': 'velocity',
            'altitude_m': 'altitude'
        }
        df_b_raw = df_b_raw.rename(columns=rename_map)
        df_t_raw = df_t_raw.rename(columns=rename_map)
        
        # Mathematically derive 'acceleration' since it isn't in your JSON
        df_b_raw['acceleration'] = df_b_raw['velocity'].diff() / df_b_raw['time'].diff()
        df_b_raw['acceleration'] = df_b_raw['acceleration'].fillna(0) # Fill the first empty row
        
        df_t_raw['acceleration'] = df_t_raw['velocity'].diff() / df_t_raw['time'].diff()
        df_t_raw['acceleration'] = df_t_raw['acceleration'].fillna(0)
        
        print("\n⚙️ Processing 18-Parameter Aerospace Metrics...")
        df_baseline_processed = compute_aerospace_metrics(df_b_raw)
        df_test_processed = compute_aerospace_metrics(df_t_raw)
        
        print("⚙️ Evaluating Deviation Matrices...")
        df_deltas = perform_delta_analysis(df_baseline_processed, df_test_processed)
        
        print("📄 Compiling 4-Page Executive Report Portfolio...")
        report_name = f"Comparison_{base_filename.split('.')[0]}_vs_{test_filename.split('.')[0]}.pdf"
        generate_pdf_report(df_baseline_processed, df_test_processed, df_deltas, report_name)
        
        print(f"\n✅ DONE! Please check the folder icon on the left to download: {report_name}")
        
    except ImportError:
        print("Error: This interactive upload feature is specifically designed for Google Colab.")
    except Exception as e:
        print(f"An error occurred: {e}. Please ensure you are uploading valid JSON files.")


        
