#

import subprocess
import logging
import os
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL


@dataclass
class MSSpecConfig:
    """Configuration parameters for MSSPEC optimization."""
    input_file: str = 'calc_rho_msspec.py'
    output_file: str = 'rho.out'
    results_dir: str = 'optimization_results'
    max_evals: int = 10
    max_parallel_jobs: int = 3
    timeout: int = 500000  # seconds
    w_range: tuple = (-0.5, 0.5)
    re_modes: list = None

    def __post_init__(self):
        self.re_modes = self.re_modes or ["G_n", "Sigma_n", "Z_n", "Pi_1", "Lowdin"]
        os.makedirs(self.results_dir, exist_ok=True)

class MSSpecOptimizer:
    """MSSPEC parameter optimization handler."""
    
    def __init__(self, config: MSSpecConfig):
        self.config = config
        self.setup_logging()
        self.backup_input_file()
        
    def setup_logging(self):
        """Configure logging with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(self.config.results_dir) / f"optimization_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def backup_input_file(self):
        """Create backup of original input file."""
        backup_path = Path(self.config.input_file).with_suffix('.py.backup')
        if not backup_path.exists():
            with open(self.config.input_file, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
        self.logger.info(f"Created backup of input file: {backup_path}")

    def update_calculation_file(self, params: Dict[str, Any]) -> bool:
        """Update calculation parameters in the input file."""
        try:
            with open(self.config.input_file, 'r') as file:
                filedata = file.readlines()

            modified = False
            for i, line in enumerate(filedata):
                if 'calc.calculation_parameters.renormalization_omega =' in line:
                    filedata[i] = f'calc.calculation_parameters.renormalization_omega = {params["w"].real} + {params["w"].imag}j\n'
                    modified = True
                elif 'calc.calculation_parameters.renormalization_mode' in line:
                    filedata[i] = f'calc.calculation_parameters.renormalization_mode = "{params["re_mode"]}"\n'
                    modified = True

            if not modified:
                self.logger.error("Required parameters not found in input file")
                return False

            with open(self.config.input_file, 'w') as file:
                file.writelines(filedata)
            return True

        except Exception as e:
            self.logger.error(f"Error updating calculation file: {str(e)}")
            return False

    def run_msspec(self, trial_id: int) -> Union[float, None]:
        """Execute MSSPEC calculation with timeout."""
        try:
            output_file = Path(self.config.results_dir) / f"rho_{trial_id}.out"
            
            # Run MSSPEC with timeout
            command = f"msspec -p {self.config.input_file}"
            process = subprocess.Popen(
                command.split(),
                stdout=open(output_file, "w"),
                stderr=subprocess.PIPE
            )
            
            try:
                _, stderr = process.communicate(timeout=self.config.timeout)
                if stderr:
                    self.logger.warning(f"MSSPEC stderr: {stderr.decode()}")
            except subprocess.TimeoutExpired:
                process.kill()
                self.logger.error(f"MSSPEC calculation timed out after {self.config.timeout}s")
                return None

            # Extract maximum modulus
            grep_command = f"grep 'MAXIMUM MODULUS' {output_file} | awk 'END{{print ($5)}}'"
            process = subprocess.Popen(
                grep_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            rho, error = process.communicate()
            
            if error:
                self.logger.error(f"Error extracting result: {error.decode()}")
                return None
                
            return float(rho.decode('utf-8').strip())

        except Exception as e:
            self.logger.error(f"Error in MSSPEC execution: {str(e)}")
            return None
        finally:
            # Cleanup temporary files but keep the output
            if os.path.exists("calc"):
                subprocess.run("rm -r calc", shell=True)

    def objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Objective function for hyperopt optimization."""
        trial_id = len(self.trials.trials)
        
        # Format the complex number w for display
        w_formatted = f"{params['w'].real:.4f} + {params['w'].imag:.4f}j"
        
        # Print progress header for this iteration
        print("\n" + "="*80)
        print(f"Trial {trial_id + 1}/{self.config.max_evals}")
        print(f"Parameters:")
        print(f"  w      = {w_formatted}")
        print(f"  re_mode = {params['re_mode']}")
        print("-"*80)
        
        if not self.update_calculation_file(params):
            print("❌ Failed to update calculation file")
            return {'status': STATUS_FAIL, 'loss': float('inf')}
        
        print("Running MSSPEC calculation...")
        result = self.run_msspec(trial_id)
        
        if result is None:
            print("❌ Calculation failed")
            return {'status': STATUS_FAIL, 'loss': float('inf')}
        
        print(f"✓ Calculation complete")
        print(f"Result: {result:.6f}")
        
        return {
            'status': STATUS_OK,
            'loss': result,
            'trial_id': trial_id,
            'params': params
        }



    def optimize(self) -> Dict[str, Any]:
        """Run the optimization process."""
        print("\n" + "="*80)
        print("Starting MSSPEC Parameter Optimization")
        print(f"Total evaluations: {self.config.max_evals}")
        print(f"Parameter ranges:")
        print(f"  w: {self.config.w_range[0]} to {self.config.w_range[1]} (real and imaginary)")
        print(f"  re_mode: {self.config.re_modes}")
        print("="*80 + "\n")
        
        self.trials = Trials()
        
        # Define the search space
        space = {
            'w': hp.uniform('w_real', self.config.w_range[0], self.config.w_range[1]) + 
                 hp.uniform('w_imag', self.config.w_range[0], self.config.w_range[1]) * 1j,
            're_mode': hp.choice('re_mode', self.config.re_modes)
        }
        
        # Run optimization
        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.config.max_evals,
            trials=self.trials,
            verbose=False  # Set to False to use our custom progress display
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("Optimization Complete")
        print("-"*80)
        print("Best parameters found:")
        print(f"  w      = {best.get('w_real', 0):.4f} + {best.get('w_imag', 0):.4f}j")
        print(f"  re_mode = {self.config.re_modes[best.get('re_mode', 0)]}")
        print(f"Best loss: {self.trials.best_trial['result']['loss']:.6f}")
        print("="*80 + "\n")
        
        # Save results
        self.save_results(best)
        return best



    def save_results(self, best_params: Dict[str, Any]):
        """Save optimization results and statistics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(self.config.results_dir) / f"results_{timestamp}.json"
        
        # Prepare results summary
        results = {
            'best_parameters': best_params,
            'best_loss': self.trials.best_trial['result']['loss'],
            'all_trials': [
                {
                    'trial_id': trial['tid'],
                    'parameters': trial['misc']['vals'],
                    'loss': trial['result']['loss'],
                    'status': trial['result']['status']
                }
                for trial in self.trials.trials
            ],
            'statistics': {
                'mean_loss': np.mean([t['result']['loss'] for t in self.trials.trials]),
                'std_loss': np.std([t['result']['loss'] for t in self.trials.trials]),
                'successful_trials': sum(1 for t in self.trials.trials if t['result']['status'] == STATUS_OK)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best loss: {self.trials.best_trial['result']['loss']}")

def calculate_scattering_order(rho: float, eps_values: list) -> dict:
    """Calculate the scattering order 'n' for each eps value given the best rho."""
    scattering_orders = {}
    for eps in eps_values:
        if rho < 1:
            n = math.log(eps * (1 - rho)) / math.log(rho) - 1
            scattering_orders[eps] = n
        else:
            logging.warning(f"rho={rho} is not valid for calculating n (must be less than 1)")
            scattering_orders[eps] = None
    return scattering_orders

def main():
    """Main execution function."""
    config = MSSpecConfig(
        max_evals=30,            # Increased number of evaluations
        max_parallel_jobs=1,     # Allow parallel processing
        timeout=500000,             # 5-minute timeout per job
        w_range=(-0.5, 0.5)      # Parameter range
    )
    
    optimizer = MSSpecOptimizer(config)
    best_params = optimizer.optimize()
    
    # Calculate scattering_order for different values of eps
    best_rho = optimizer.trials.best_trial['result']['loss']
    eps_values = [0.1, 0.5, 1, 2, 3, 5, 10]
    scattering_orders = calculate_scattering_order(best_rho, eps_values)
    
    # Display scattering orders
    print("\nScattering Order Results:")
    for eps, n in scattering_orders.items():
        if n is not None:
            print(f"eps = {eps}: scattering_order = {n:.4f}")
        else:
            print(f"eps = {eps}: scattering_order calculation invalid (rho >= 1)")

    # Optionally save results in the output JSON
    results_file = Path(config.results_dir) / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({"best_parameters": best_params, "scattering_orders": scattering_orders}, f, indent=2)
    
    return best_params

if __name__ == "__main__":
    main()
