import argparse
import json
import numpy as np
import os
import csv
from src.simulator import DetectorSimulator

def main():
    parser = argparse.ArgumentParser(description="Semiconductor Detector Simulator")
    parser.add_argument("--material", type=str, choices=["CZT", "TlBr"], required=True, help="Detector Material")
    parser.add_argument("--energy", type=float, default=60.0, help="Incident X-ray Energy (keV)")
    parser.add_argument("--temp", type=float, default=300.0, help="Temperature (K)")
    parser.add_argument("--bias", type=float, default=500.0, help="Bias Voltage (V)")
    parser.add_argument("--thick", type=float, default=2.0, help="Detector Thickness (mm)")
    parser.add_argument("--output", type=str, default="simulation_result", help="Output filename prefix")
    
    args = parser.parse_args()
    
    print(f"--- Starting Simulation ---")
    print(f"Material: {args.material}")
    print(f"Energy: {args.energy} keV")
    print(f"Temperature: {args.temp} K")
    
    sim = DetectorSimulator(args.material, args.thick, args.bias, args.temp)
    
    # Run Simulation
    result = sim.simulate_spectrum(args.energy)
    
    # Save Results
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Spectrum CSV
    csv_path = os.path.join(output_dir, f"{args.output}_spectrum.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Energy_keV", "Counts_Normalized"])
        for e, c in zip(result["energy_axis"], result["spectrum"]):
            writer.writerow([e, c])
            
    # 2. Save Metadata/Stats JSON
    json_path = os.path.join(output_dir, f"{args.output}_stats.json")
    stats = {
        "parameters": vars(args),
        "results": {
            "interaction_efficiency": result["interaction_efficiency"],
            "noise_properties": result["noise_stats"]
        }
    }
    
    # Convert numpy types to native python types for JSON serialization
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
        
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=4, default=convert)
        
    print(f"--- Simulation Complete ---")
    print(f"Noise (FWHM): {result['noise_stats']['fwhm_noise_keV']:.4f} keV")
    print(f"Dark Current: {result['noise_stats']['dark_current_nA']:.4f} nA")
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
