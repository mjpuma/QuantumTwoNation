"""
Enhanced Quantum Trade Policy Analysis Framework - CLEAN VERSION
Fixed indentation and formatting issues

Features:
- 8 quantum models with enhanced differentiation
- Comprehensive probabilistic evaluation
- Parameter exploration utilities
- OECD data integration interface
- Multiple visualization levels with fallbacks
- Extensive error handling
- Easy parameter customization

Author: Enhanced Framework Team
Updated: 2025
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
from scipy import stats
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import json

# MATPLOTLIB BACKEND FIX - Set environment variable first
os.environ['MPLBACKEND'] = 'TkAgg'

# Enhanced matplotlib compatibility setup with smart backend selection
PLOTTING_AVAILABLE = True

try:
    import matplotlib
    
    # Smart backend selection - try GUI backends first, fallback to Agg
    backend_success = False
    backends_to_try = ['TkAgg', 'Qt5Agg', 'QtAgg', 'MacOSX', 'Agg']
    
    for backend in backends_to_try:
        try:
            matplotlib.use(backend, force=True)
            print(f"🔧 Attempting matplotlib backend: {backend}")
            
            # Test if the backend actually works
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Quick test to see if backend is functional
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            test_ax.plot([1, 2], [1, 2])
            
            if backend != 'Agg':
                # For GUI backends, this should work without errors
                plt.close(test_fig)
                print(f"✅ Successfully using matplotlib backend: {backend} (GUI capable)")
                backend_success = True
                break
            else:
                # Agg backend - save only
                plt.close(test_fig)
                print(f"⚠️  Using matplotlib backend: {backend} (save only, no display)")
                backend_success = True
                break
                
        except Exception as e:
            print(f"❌ Backend {backend} failed: {e}")
            continue
    
    if not backend_success:
        print("❌ No working matplotlib backend found")
        PLOTTING_AVAILABLE = False
    else:
        # Continue with matplotlib setup
        plt.ioff()  # Turn off interactive mode
        
        try:
            import seaborn as sns
            style_options = ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'seaborn']
            for style in style_options:
                try:
                    plt.style.use(style)
                    sns.set_palette("husl")
                    print(f"✅ Using matplotlib style: {style}")
                    break
                except Exception:
                    continue
            else:
                plt.style.use('default')
                plt.rcParams['axes.grid'] = True
                plt.rcParams['grid.alpha'] = 0.3
                print("⚠️  Using matplotlib default style")
        except ImportError:
            plt.style.use('default')
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            print("⚠️  Seaborn not available, using matplotlib defaults")
        
        # Enhanced plotting configuration
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['axes.formatter.useoffset'] = False
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.facecolor'] = 'white'
        
        # Report final status
        final_backend = plt.get_backend()
        can_display = final_backend != 'Agg'
        print(f"📊 Matplotlib setup complete:")
        print(f"   Backend: {final_backend}")
        print(f"   Can display plots: {'✅ Yes' if can_display else '⚠️ No (save only)'}")
        print(f"   Plotting available: {'✅ Yes' if PLOTTING_AVAILABLE else '❌ No'}")
    
except ImportError as e:
    print(f"❌ Critical error importing matplotlib: {e}")
    print("📊 Plotting will be disabled - analysis will run with data output only")
    print("💡 Try installing: pip install matplotlib PyQt5")
    PLOTTING_AVAILABLE = False

warnings.filterwarnings('ignore')


# Optional: Add this helper function for enhanced plotting throughout your code
def enhanced_plot_save_and_show(fig, save_path, show_plot=True, close_after=True):
    """
    Enhanced plotting function that always saves and optionally shows plots
    Works with any matplotlib backend
    """
    try:
        # Always save first
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Plot saved: {save_path}")
        
        # Try to show if requested and backend supports it
        if show_plot and PLOTTING_AVAILABLE:
            try:
                backend = plt.get_backend()
                if backend != 'Agg':
                    plt.show()
                    print("✅ Plot displayed")
                else:
                    print("📁 Plot saved (Agg backend - display not available)")
            except Exception as e:
                print(f"⚠️ Could not display plot: {e}")
                print(f"📁 But plot was saved successfully: {save_path}")
        
    except Exception as e:
        print(f"❌ Error with plot: {e}")
    finally:
        if close_after:
            plt.close(fig)


# Quick matplotlib diagnostic function (optional - for debugging)
def check_matplotlib_status():
    """Check and report current matplotlib status"""
    print("\n🔍 MATPLOTLIB DIAGNOSTIC")
    print("=" * 30)
    
    if PLOTTING_AVAILABLE:
        backend = plt.get_backend()
        print(f"✅ Matplotlib available")
        print(f"   Version: {matplotlib.__version__}")
        print(f"   Backend: {backend}")
        print(f"   Interactive: {plt.isinteractive()}")
        print(f"   Can display: {'Yes' if backend != 'Agg' else 'No'}")
        
        # Quick test
        try:
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            test_ax.plot([1, 2], [1, 2])
            plt.close(test_fig)
            print(f"   Basic plotting: ✅ Working")
        except Exception as e:
            print(f"   Basic plotting: ❌ Failed ({e})")
    else:
        print("❌ Matplotlib not available")
        print("💡 Try: pip install matplotlib PyQt5")
        print("🐧 Linux: sudo apt-get install python3-tk")
    
    print("=" * 30)


# Uncomment the next line if you want to see the diagnostic on import
# check_matplotlib_status()


@dataclass
class QuantumSyntheticConfig:
    """Configuration for quantum-appropriate synthetic data generation."""
    
    # Time parameters
    start_year: int = 2008
    end_year: int = 2018
    frequency: str = 'M'  # Monthly data
    
    # Quantum coherence parameters
    coherence_time_country1: float = 2.5  # Years - how long quantum effects persist
    coherence_time_country2: float = 1.8  # Different coherence for each country
    
    # Oscillation parameters (quantum-like)
    primary_frequency_1: float = 0.8  # Primary oscillation frequency (1/years)
    primary_frequency_2: float = 1.2  # Different frequency for country 2
    secondary_frequency_1: float = 2.1  # Higher harmonics
    secondary_frequency_2: float = 1.7
    
    # Decoherence parameters
    decoherence_rate_1: float = 0.15  # How fast quantum effects decay
    decoherence_rate_2: float = 0.25  # Faster decoherence for country 2
    
    # Nonlinear parameters
    nonlinear_strength_1: float = 0.3  # Strength of self-interaction
    nonlinear_strength_2: float = 0.2
    
    # Entanglement parameters
    entanglement_strength: float = 0.4  # Cross-country quantum correlation
    phase_difference: float = np.pi/3  # Phase relationship between countries
    
    # Regime change parameters
    regime_change_times: List[float] = None  # Times of abrupt changes
    regime_change_amplitudes: List[float] = None  # Amplitude of changes
    
    # External driving (Floquet-like)
    external_drive_frequency: float = 0.5  # External periodic influence
    external_drive_amplitude: float = 0.15
    
    # Noise parameters
    quantum_noise_level: float = 0.08  # Quantum measurement noise
    classical_noise_level: float = 0.03  # Classical noise
    
    # Base levels and ranges
    base_level_1: float = 0.45  # Average policy restriction level
    base_level_2: float = 0.55
    amplitude_1: float = 0.35  # Oscillation amplitude
    amplitude_2: float = 0.30
    
    def __post_init__(self):
        """Set default regime changes if not provided."""
        if self.regime_change_times is None:
            self.regime_change_times = [3.0, 7.0]  # Around 2011 and 2015
        if self.regime_change_amplitudes is None:
            self.regime_change_amplitudes = [-0.25, 0.20]  # Magnitude of regime shifts


@dataclass
class QuantumModelConfig:
    """Configuration for quantum model parameters."""
    
    # Individual country parameters
    country_params: Dict[str, Dict[str, float]] = None
    
    # System-wide parameters
    coupling_strength: float = 0.4  # Inter-country coupling
    temperature: float = 0.1  # Effective temperature for thermal effects
    
    # Floquet driving parameters
    floquet_frequency: float = 1.5  # Driving frequency
    floquet_amplitude: float = 0.3  # Driving strength
    floquet_phase: float = 0.0  # Driving phase
    
    # Lindblad operators strengths
    dephasing_rate: float = 0.08  # Dephasing rate
    relaxation_rate: float = 0.05  # Energy relaxation rate
    
    def __post_init__(self):
        """Set default country parameters if not provided."""
        if self.country_params is None:
            self.country_params = {
                'Egypt': {
                    'E0': 2.5,       # Energy scale
                    'Delta': 1.8,    # Tunneling strength
                    'delta': 0.8,    # Detuning
                    'phi': 0.0,      # Phase
                    'gamma': 0.05,   # Individual damping (low - coherent)
                    'coherence_time': 3.0  # Coherence time
                },
                'Russian Federation': {
                    'E0': 2.0,       # Different energy scale
                    'Delta': 1.2,    # Weaker tunneling
                    'delta': -0.6,   # Opposite detuning
                    'phi': np.pi/4,  # Phase shift
                    'gamma': 0.20,   # Higher damping (decoherent)
                    'coherence_time': 1.5  # Shorter coherence
                }
            }


class ConfigurableSyntheticDataGenerator:
    """
    Generates quantum-appropriate synthetic data with configurable parameters.
    """
    
    def __init__(self, config: QuantumSyntheticConfig):
        self.config = config
        
    def generate_quantum_synthetic_data(self, countries: List[str]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Generate synthetic data that exhibits quantum-like behaviors."""
        print("🔄 Generating quantum-appropriate synthetic data...")
        
        # Create time series
        date_range = pd.date_range(
            start=f"{self.config.start_year}-01-01", 
            end=f"{self.config.end_year}-12-31", 
            freq=self.config.frequency
        )
        time_years = np.arange(len(date_range)) / 12  # Convert to years
        
        synthetic_data = {}
        
        for i, country in enumerate(countries):
            print(f"  Generating quantum data for {country}...")
            
            # Select country-specific parameters
            if i == 0:
                coherence_time = self.config.coherence_time_country1
                decoherence_rate = self.config.decoherence_rate_1
                primary_freq = self.config.primary_frequency_1
                secondary_freq = self.config.secondary_frequency_1
                nonlinear_strength = self.config.nonlinear_strength_1
                base_level = self.config.base_level_1
                amplitude = self.config.amplitude_1
            else:
                coherence_time = self.config.coherence_time_country2
                decoherence_rate = self.config.decoherence_rate_2
                primary_freq = self.config.primary_frequency_2
                secondary_freq = self.config.secondary_frequency_2
                nonlinear_strength = self.config.nonlinear_strength_2
                base_level = self.config.base_level_2
                amplitude = self.config.amplitude_2
            
            # Generate quantum-like evolution
            data = self._generate_country_quantum_evolution(
                time_years, coherence_time, decoherence_rate, 
                primary_freq, secondary_freq, nonlinear_strength,
                base_level, amplitude, country_index=i
            )
            
            synthetic_data[country] = data
        
        # Add quantum entanglement effects between countries
        if len(countries) == 2:
            synthetic_data = self._add_quantum_entanglement(
                synthetic_data, time_years, countries
            )
        
        # Add regime changes
        synthetic_data = self._add_regime_changes(synthetic_data, time_years)
        
        # Ensure physical bounds [0, 1]
        for country in countries:
            synthetic_data[country] = np.clip(synthetic_data[country], 0.01, 0.99)
        
        # Create DataFrame
        df_data = {
            'Date': date_range,
            'Year': date_range.year + date_range.month/12
        }
        df_data.update(synthetic_data)
        
        # Print characteristics
        self._print_data_characteristics(synthetic_data, countries)
        
        return pd.DataFrame(df_data), synthetic_data
    
    def _generate_country_quantum_evolution(
        self, time_years: np.ndarray, coherence_time: float, 
        decoherence_rate: float, primary_freq: float, secondary_freq: float,
        nonlinear_strength: float, base_level: float, amplitude: float,
        country_index: int
    ) -> np.ndarray:
        """Generate quantum evolution for a single country."""
        
        # Coherent oscillations with decoherence
        coherent_part = (
            amplitude * np.cos(2 * np.pi * primary_freq * time_years) * 
            np.exp(-time_years / coherence_time) +
            0.3 * amplitude * np.sin(2 * np.pi * secondary_freq * time_years) * 
            np.exp(-time_years / (coherence_time * 0.7))
        )
        
        # Decoherence envelope
        decoherence_envelope = np.exp(-decoherence_rate * time_years)
        
        # Nonlinear self-interaction (creates asymmetry)
        nonlinear_part = nonlinear_strength * coherent_part**2 * np.sign(coherent_part)
        
        # External driving (periodic)
        external_drive = (
            self.config.external_drive_amplitude * 
            np.cos(2 * np.pi * self.config.external_drive_frequency * time_years + 
                   country_index * np.pi/4)  # Different phase for each country
        )
        
        # Quantum measurement noise (more structured than classical)
        quantum_noise = (
            self.config.quantum_noise_level * 
            (np.random.normal(0, 1, len(time_years)) * decoherence_envelope +
             0.5 * np.sin(10 * np.pi * time_years) * 
             np.exp(-2 * time_years / coherence_time))
        )
        
        # Classical noise
        classical_noise = self.config.classical_noise_level * np.random.normal(0, 1, len(time_years))
        
        # Combine all components
        data = (
            base_level + 
            coherent_part * decoherence_envelope + 
            nonlinear_part +
            external_drive +
            quantum_noise + 
            classical_noise
        )
        
        return data
    
    def _add_quantum_entanglement(
        self, synthetic_data: Dict[str, np.ndarray], 
        time_years: np.ndarray, countries: List[str]
    ) -> Dict[str, np.ndarray]:
        """Add quantum entanglement effects between countries."""
        
        country1, country2 = countries[0], countries[1]
        data1, data2 = synthetic_data[country1], synthetic_data[country2]
        
        # Create entangled component
        entangled_signal = (
            self.config.entanglement_strength * 
            np.sin(2 * np.pi * 0.6 * time_years + self.config.phase_difference) *
            np.exp(-0.1 * time_years)  # Entanglement also decoheres
        )
        
        # Add correlated component to both countries (with phase difference)
        synthetic_data[country1] = data1 + entangled_signal
        synthetic_data[country2] = data2 + entangled_signal * np.cos(self.config.phase_difference)
        
        return synthetic_data
    
    def _add_regime_changes(
        self, synthetic_data: Dict[str, np.ndarray], time_years: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Add abrupt regime changes (quantum measurement-like collapses)."""
        
        for i, (change_time, amplitude) in enumerate(
            zip(self.config.regime_change_times, self.config.regime_change_amplitudes)
        ):
            # Find time index
            change_idx = np.argmin(np.abs(time_years - change_time))
            
            # Create smooth but rapid transition
            transition_width = 6  # months
            transition_indices = np.arange(
                max(0, change_idx - transition_width//2),
                min(len(time_years), change_idx + transition_width//2)
            )
            
            # Sigmoid transition
            x = np.arange(len(transition_indices)) - len(transition_indices)//2
            sigmoid = 1 / (1 + np.exp(-x))
            
            for country in synthetic_data.keys():
                # Apply regime change with country-specific response
                country_amplitude = amplitude * (1.0 if 'Egypt' in country else 0.7)
                synthetic_data[country][transition_indices] += country_amplitude * sigmoid
        
        return synthetic_data
    
    def _print_data_characteristics(self, synthetic_data: Dict[str, np.ndarray], countries: List[str]):
        """Print characteristics of generated data."""
        print("📊 Quantum Synthetic Data Characteristics:")
        
        for country, data in synthetic_data.items():
            changes = np.abs(np.diff(data))
            autocorr = np.corrcoef(data[:-1], data[1:])[0,1] if len(data) > 1 else 0
            
            print(f"  {country}:")
            print(f"    Mean: {np.mean(data):.3f}, Std: {np.std(data):.3f}")
            print(f"    Max change: {np.max(changes):.3f}")
            print(f"    Autocorrelation: {autocorr:.3f}")
            print(f"    Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        
        if len(countries) == 2:
            cross_corr = np.corrcoef(
                synthetic_data[countries[0]], 
                synthetic_data[countries[1]]
            )[0,1]
            print(f"  Cross-correlation: {cross_corr:.3f}")


class EnhancedQuantumTradeEvolutionFramework:
    """Enhanced quantum framework with better model differentiation."""
    
    def __init__(self, config: QuantumModelConfig):
        self.config = config
        self.model_names = [
            'schrodinger', 'schrodinger_nonlinear', 'von_neumann', 'lindblad',
            'floquet_schrodinger', 'floquet_schrodinger_nonlinear', 
            'floquet_von_neumann', 'floquet_lindblad'
        ]
        self.model_colors = {
            'schrodinger': '#1f77b4',
            'schrodinger_nonlinear': '#ff7f0e', 
            'von_neumann': '#2ca02c',
            'lindblad': '#d62728',
            'floquet_schrodinger': '#9467bd',
            'floquet_schrodinger_nonlinear': '#8c564b',
            'floquet_von_neumann': '#e377c2',
            'floquet_lindblad': '#7f7f7f'
        }
    
    # MODEL 1: Standard Schrödinger Equation
    def solve_schrodinger(self, H, t_points, initial_state=None):
        """Solve linear Schrödinger equation - Model 1."""
        if initial_state is None:
            initial_state = np.zeros(H.shape[0], dtype=complex)
            initial_state[0] = 1.0

        states = []
        for t in t_points:
            U = expm(-1j * H * t)
            state = U @ initial_state
            states.append(state)

        return np.array(states)
    
    # MODEL 2: Nonlinear Schrödinger Equation  
    def solve_schrodinger_nonlinear(self, H, t_points, nonlinear_strength=None, initial_state=None):
        """Solve nonlinear Schrödinger equation - Model 2."""
        if nonlinear_strength is None:
            nonlinear_strength = 0.2
            
        if initial_state is None:
            initial_state = np.zeros(H.shape[0], dtype=complex)
            initial_state[0] = 1.0

        def nonlinear_schrodinger_rhs(t, psi_vec):
            psi = psi_vec.astype(complex)
            linear_term = -1j * H @ psi
            density = np.abs(psi)**2
            nonlinear_term = -1j * nonlinear_strength * density * psi
            
            if len(psi) > 2:
                total_density = np.sum(density)
                cross_nonlinear = -1j * 0.5 * nonlinear_strength * total_density * psi
                nonlinear_term += cross_nonlinear
            
            return linear_term + nonlinear_term

        try:
            sol = solve_ivp(
                nonlinear_schrodinger_rhs,
                [t_points[0], t_points[-1]],
                initial_state,
                t_eval=t_points,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )
            
            if sol.success:
                states = []
                for psi_vec in sol.y.T:
                    psi = psi_vec / np.linalg.norm(psi_vec)
                    states.append(psi)
                return np.array(states)
            else:
                return self.solve_schrodinger(H, t_points, initial_state)
                
        except Exception:
            return self.solve_schrodinger(H, t_points, initial_state)

    # MODEL 3: von Neumann Equation
    def solve_von_neumann(self, H, t_points, decoherence_rate=None, initial_rho=None):
        """Solve von Neumann equation - Model 3."""
        if decoherence_rate is None:
            decoherence_rate = 0.08
            
        if initial_rho is None:
            initial_state = np.zeros(H.shape[0], dtype=complex)
            initial_state[0] = 1.0
            initial_rho = np.outer(initial_state, np.conj(initial_state))
        
        def von_neumann_rhs(t, rho_vec):
            rho = rho_vec.reshape(H.shape).astype(complex)
            drho_dt = -1j * (H @ rho - rho @ H)
            
            if abs(decoherence_rate) > 1e-12:
                trace_rho = np.trace(rho)
                thermal_state = self._get_thermal_state(H)
                drho_dt -= decoherence_rate * (rho - trace_rho * thermal_state)
            
            return drho_dt.flatten()

        try:
            sol = solve_ivp(
                von_neumann_rhs,
                [t_points[0], t_points[-1]],
                initial_rho.flatten(),
                t_eval=t_points,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12
            )
            
            if sol.success:
                rho_states = []
                for rho_vec in sol.y.T:
                    rho = rho_vec.reshape(H.shape).astype(complex)
                    rho = self._project_to_physical_density_matrix(rho)
                    rho_states.append(rho)
                return np.array(rho_states)
            else:
                return self._solve_von_neumann_fallback(H, t_points, initial_rho, decoherence_rate)
                
        except Exception:
            return self._solve_von_neumann_fallback(H, t_points, initial_rho, decoherence_rate)

    def _get_thermal_state(self, H):
        """Get thermal equilibrium state."""
        try:
            eigenvals, eigenvecs = np.linalg.eigh(H)
            beta = 1.0 / self.config.temperature if self.config.temperature > 0 else 10.0
            thermal_weights = np.exp(-beta * eigenvals)
            thermal_weights /= np.sum(thermal_weights)
            
            thermal_state = np.zeros_like(H, dtype=complex)
            for i, (weight, vec) in enumerate(zip(thermal_weights, eigenvecs.T)):
                thermal_state += weight * np.outer(vec, np.conj(vec))
            
            return thermal_state
        except:
            return np.eye(H.shape[0], dtype=complex) / H.shape[0]

    def _solve_von_neumann_fallback(self, H, t_points, initial_rho, decoherence_rate):
        """Enhanced fallback with higher precision."""
        dt = t_points[1] - t_points[0] if len(t_points) > 1 else 0.1
        
        rho_states = [initial_rho]
        current_rho = initial_rho.copy()
        thermal_state = self._get_thermal_state(H)
        
        for i in range(1, len(t_points)):
            drho_dt = -1j * (H @ current_rho - current_rho @ H)
            
            if abs(decoherence_rate) > 1e-12:
                trace_rho = np.trace(current_rho)
                drho_dt -= decoherence_rate * (current_rho - trace_rho * thermal_state)
            
            current_rho = current_rho + dt * drho_dt
            current_rho = self._project_to_physical_density_matrix(current_rho)
            rho_states.append(current_rho.copy())
        
        return np.array(rho_states)

    # MODEL 4: Lindblad Equation
    def solve_lindblad(self, H, t_points, lindblad_ops=None, initial_rho=None):
        """Solve Lindblad equation - Model 4."""
        if initial_rho is None:
            initial_state = np.zeros(H.shape[0], dtype=complex)
            initial_state[0] = 1.0
            initial_rho = np.outer(initial_state, np.conj(initial_state))

        if lindblad_ops is None:
            lindblad_ops = self._construct_enhanced_lindblad_operators(H.shape[0])

        def lindblad_rhs(t, rho_vec):
            rho = rho_vec.reshape(H.shape).astype(complex)
            drho_dt = -1j * (H @ rho - rho @ H)
            
            for i, L in enumerate(lindblad_ops):
                L_dag = np.conj(L.T)
                rate_factor = 1.0 + 0.2 * np.sin(2.0 * t + i * np.pi/4)
                drho_dt += rate_factor * (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))
            
            return drho_dt.flatten()

        try:
            sol = solve_ivp(
                lindblad_rhs,
                [t_points[0], t_points[-1]],
                initial_rho.flatten(),
                t_eval=t_points,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            
            if sol.success:
                rho_states = []
                for rho_vec in sol.y.T:
                    rho = rho_vec.reshape(H.shape).astype(complex)
                    rho = self._project_to_physical_density_matrix(rho)
                    rho_states.append(rho)
                return np.array(rho_states)
            else:
                return self.solve_von_neumann(H, t_points, initial_rho=initial_rho)
                
        except Exception:
            return self.solve_von_neumann(H, t_points, initial_rho=initial_rho)

    def _construct_enhanced_lindblad_operators(self, dim):
        """Construct enhanced Lindblad operators."""
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        
        lindblad_ops = []
        
        if dim == 4:  # Two-country system
            L1 = np.sqrt(self.config.dephasing_rate) * np.kron(sigma_z, I)
            L2 = np.sqrt(self.config.dephasing_rate * 0.8) * np.kron(I, sigma_z)
            L3 = np.sqrt(self.config.relaxation_rate) * np.kron(sigma_x, I)
            L4 = np.sqrt(self.config.relaxation_rate * 0.9) * np.kron(I, sigma_x)
            L5 = np.sqrt(0.03) * np.kron(sigma_y, sigma_y)
            lindblad_ops = [L1, L2, L3, L4, L5]
        else:  # Single-country system
            L1 = np.sqrt(self.config.dephasing_rate) * sigma_z
            L2 = np.sqrt(self.config.relaxation_rate) * sigma_x
            L3 = np.sqrt(0.02) * sigma_y
            lindblad_ops = [L1, L2, L3]
        
        return lindblad_ops

    # FLOQUET MODELS (5-8)
    def solve_floquet_schrodinger(self, H_static, t_points, drive_frequency=None, drive_amplitude=None, initial_state=None):
        """Solve Floquet Schrödinger equation - Model 5."""
        if drive_frequency is None:
            drive_frequency = self.config.floquet_frequency
        if drive_amplitude is None:
            drive_amplitude = self.config.floquet_amplitude
            
        if initial_state is None:
            initial_state = np.zeros(H_static.shape[0], dtype=complex)
            initial_state[0] = 1.0

        V_drive = self._construct_enhanced_drive_hamiltonian(H_static.shape[0])

        def floquet_schrodinger_rhs(t, psi):
            drive_term = (
                drive_amplitude * np.cos(drive_frequency * t + self.config.floquet_phase) +
                0.3 * drive_amplitude * np.sin(2 * drive_frequency * t) +
                0.1 * drive_amplitude * np.cos(0.5 * drive_frequency * t)
            )
            H_t = H_static + V_drive * drive_term
            return -1j * H_t @ psi

        try:
            sol = solve_ivp(
                floquet_schrodinger_rhs,
                [t_points[0], t_points[-1]],
                initial_state,
                t_eval=t_points,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )
            
            if sol.success:
                states = []
                for psi_vec in sol.y.T:
                    psi = psi_vec / np.linalg.norm(psi_vec)
                    states.append(psi)
                return np.array(states)
            else:
                return self.solve_schrodinger(H_static, t_points, initial_state)
                
        except Exception:
            return self.solve_schrodinger(H_static, t_points, initial_state)

    def _construct_enhanced_drive_hamiltonian(self, dim):
        """Construct enhanced driving Hamiltonian."""
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        I = np.eye(2, dtype=complex)
        
        if dim == 4:
            V_x = np.kron(sigma_x, I) + np.kron(I, sigma_x)
            V_y = 0.5 * (np.kron(sigma_y, I) - np.kron(I, sigma_y))
            V_xy = 0.3 * np.kron(sigma_x, sigma_y)
            return V_x + V_y + V_xy
        else:
            return sigma_x + 0.3 * sigma_y

    def solve_floquet_schrodinger_nonlinear(self, H_static, t_points, drive_frequency=None, drive_amplitude=None, nonlinear_strength=None, initial_state=None):
        """Solve Floquet nonlinear Schrödinger equation - Model 6."""
        if drive_frequency is None:
            drive_frequency = self.config.floquet_frequency
        if drive_amplitude is None:
            drive_amplitude = self.config.floquet_amplitude
        if nonlinear_strength is None:
            nonlinear_strength = 0.25
            
        if initial_state is None:
            initial_state = np.zeros(H_static.shape[0], dtype=complex)
            initial_state[0] = 1.0

        V_drive = self._construct_enhanced_drive_hamiltonian(H_static.shape[0])

        def floquet_nonlinear_rhs(t, psi_vec):
            psi = psi_vec.astype(complex)
            
            drive_term = (
                drive_amplitude * np.cos(drive_frequency * t + self.config.floquet_phase) +
                0.3 * drive_amplitude * np.sin(2 * drive_frequency * t)
            )
            H_t = H_static + V_drive * drive_term
            
            linear_term = -1j * H_t @ psi
            effective_nonlinearity = nonlinear_strength * (1 + 0.2 * np.abs(drive_term))
            density = np.abs(psi)**2
            nonlinear_term = -1j * effective_nonlinearity * density * psi
            
            return linear_term + nonlinear_term

        try:
            sol = solve_ivp(
                floquet_nonlinear_rhs,
                [t_points[0], t_points[-1]],
                initial_state,
                t_eval=t_points,
                method='RK45',
                rtol=1e-8,
                atol=1e-10
            )
            
            if sol.success:
                states = []
                for psi_vec in sol.y.T:
                    psi = psi_vec / np.linalg.norm(psi_vec)
                    states.append(psi)
                return np.array(states)
            else:
                return self.solve_floquet_schrodinger(H_static, t_points, drive_frequency, drive_amplitude, initial_state)
                
        except Exception:
            return self.solve_floquet_schrodinger(H_static, t_points, drive_frequency, drive_amplitude, initial_state)

    def solve_floquet_von_neumann(self, H_static, t_points, drive_frequency=None, drive_amplitude=None, decoherence_rate=None, initial_rho=None):
        """Solve Floquet von Neumann equation - Model 7."""
        if drive_frequency is None:
            drive_frequency = self.config.floquet_frequency
        if drive_amplitude is None:
            drive_amplitude = self.config.floquet_amplitude
        if decoherence_rate is None:
            decoherence_rate = 0.08
            
        if initial_rho is None:
            initial_state = np.zeros(H_static.shape[0], dtype=complex)
            initial_state[0] = 1.0
            initial_rho = np.outer(initial_state, np.conj(initial_state))

        V_drive = self._construct_enhanced_drive_hamiltonian(H_static.shape[0])
        thermal_state = self._get_thermal_state(H_static)

        def floquet_von_neumann_rhs(t, rho_vec):
            rho = rho_vec.reshape(H_static.shape).astype(complex)
            
            drive_term = (
                drive_amplitude * np.cos(drive_frequency * t + self.config.floquet_phase) +
                0.3 * drive_amplitude * np.sin(2 * drive_frequency * t)
            )
            H_t = H_static + V_drive * drive_term
            
            drho_dt = -1j * (H_t @ rho - rho @ H_t)
            
            if abs(decoherence_rate) > 1e-12:
                trace_rho = np.trace(rho)
                effective_rate = decoherence_rate * (1 + 0.1 * np.abs(drive_term))
                drho_dt -= effective_rate * (rho - trace_rho * thermal_state)
            
            return drho_dt.flatten()

        try:
            sol = solve_ivp(
                floquet_von_neumann_rhs,
                [t_points[0], t_points[-1]],
                initial_rho.flatten(),
                t_eval=t_points,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12
            )
            
            if sol.success:
                rho_states = []
                for rho_vec in sol.y.T:
                    rho = rho_vec.reshape(H_static.shape).astype(complex)
                    rho = self._project_to_physical_density_matrix(rho)
                    rho_states.append(rho)
                return np.array(rho_states)
            else:
                return self.solve_von_neumann(H_static, t_points, decoherence_rate, initial_rho)
                
        except Exception:
            return self.solve_von_neumann(H_static, t_points, decoherence_rate, initial_rho)

    def solve_floquet_lindblad(self, H_static, t_points, drive_frequency=None, drive_amplitude=None, lindblad_ops=None, initial_rho=None):
        """Solve Floquet Lindblad equation - Model 8."""
        if drive_frequency is None:
            drive_frequency = self.config.floquet_frequency
        if drive_amplitude is None:
            drive_amplitude = self.config.floquet_amplitude
            
        if initial_rho is None:
            initial_state = np.zeros(H_static.shape[0], dtype=complex)
            initial_state[0] = 1.0
            initial_rho = np.outer(initial_state, np.conj(initial_state))

        if lindblad_ops is None:
            lindblad_ops = self._construct_enhanced_lindblad_operators(H_static.shape[0])

        V_drive = self._construct_enhanced_drive_hamiltonian(H_static.shape[0])

        def floquet_lindblad_rhs(t, rho_vec):
            rho = rho_vec.reshape(H_static.shape).astype(complex)
            
            drive_term = (
                drive_amplitude * np.cos(drive_frequency * t + self.config.floquet_phase) +
                0.3 * drive_amplitude * np.sin(2 * drive_frequency * t)
            )
            H_t = H_static + V_drive * drive_term
            
            drho_dt = -1j * (H_t @ rho - rho @ H_t)
            
            for i, L in enumerate(lindblad_ops):
                L_dag = np.conj(L.T)
                rate_factor = 1.0 + 0.15 * np.sin(drive_frequency * t + i * np.pi/3)
                drho_dt += rate_factor * (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L))
            
            return drho_dt.flatten()

        try:
            sol = solve_ivp(
                floquet_lindblad_rhs,
                [t_points[0], t_points[-1]],
                initial_rho.flatten(),
                t_eval=t_points,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            
            if sol.success:
                rho_states = []
                for rho_vec in sol.y.T:
                    rho = rho_vec.reshape(H_static.shape).astype(complex)
                    rho = self._project_to_physical_density_matrix(rho)
                    rho_states.append(rho)
                return np.array(rho_states)
            else:
                return self.solve_lindblad(H_static, t_points, lindblad_ops, initial_rho)
                
        except Exception:
            return self.solve_lindblad(H_static, t_points, lindblad_ops, initial_rho)

    # OBSERVABLE CALCULATION
    def calculate_observables_pure(self, states, countries):
        """Calculate observables from pure states."""
        n_countries = len(countries)
        sigma_z = np.array([[1, 0], [0, -1]])
        I = np.eye(2)

        observables = {}

        for i, country in enumerate(countries):
            country_params = self.config.country_params.get(country, {})
            phi = country_params.get('phi', 0.0)
            
            ops = [I] * n_countries
            ops[i] = sigma_z * np.cos(phi) + np.array([[0, 1], [1, 0]]) * np.sin(phi)

            M = ops[0]
            for op in ops[1:]:
                M = np.kron(M, op)

            expectations = []
            for state in states:
                state = state / np.linalg.norm(state)
                exp_val = np.real(np.conj(state) @ M @ state)
                prob = 1 / (1 + np.exp(-2 * exp_val))
                prob = np.clip(prob, 0.01, 0.99)
                expectations.append(prob)

            observables[country] = np.array(expectations)

        return observables

    def calculate_observables_mixed(self, rho_states, countries):
        """Calculate observables from density matrices."""
        n_countries = len(countries)
        sigma_z = np.array([[1, 0], [0, -1]])
        I = np.eye(2)

        observables = {}

        for i, country in enumerate(countries):
            country_params = self.config.country_params.get(country, {})
            phi = country_params.get('phi', 0.0)
            
            ops = [I] * n_countries
            ops[i] = sigma_z * np.cos(phi) + np.array([[0, 1], [1, 0]]) * np.sin(phi)

            M = ops[0]
            for op in ops[1:]:
                M = np.kron(M, op)

            expectations = []
            for rho in rho_states:
                trace_rho = np.trace(rho)
                if abs(trace_rho) > 1e-10:
                    rho = rho / trace_rho
                
                exp_val = np.real(np.trace(M @ rho))
                gamma = country_params.get('gamma', 0.1)
                prob = 1 / (1 + np.exp(-2 * exp_val * (1 + gamma)))
                prob = np.clip(prob, 0.01, 0.99)
                expectations.append(prob)

            observables[country] = np.array(expectations)

        return observables

    # 8-MODEL COMPARISON
    def run_all_models(self, H, t_points, countries):
        """Run all 8 quantum models with enhanced differentiation."""
        print("\n🔬 Running Enhanced 8-Model Quantum Comparison...")
        all_predictions = {}
        
        print("  Running Models 1-4 (Standard)...")
        
        # Model 1: Schrödinger
        states_1 = self.solve_schrodinger(H, t_points)
        all_predictions['schrodinger'] = self.calculate_observables_pure(states_1, countries)
        
        # Model 2: Nonlinear Schrödinger  
        states_2 = self.solve_schrodinger_nonlinear(H, t_points, nonlinear_strength=0.25)
        all_predictions['schrodinger_nonlinear'] = self.calculate_observables_pure(states_2, countries)
        
        # Model 3: von Neumann
        rho_states_3 = self.solve_von_neumann(H, t_points, decoherence_rate=0.08)
        all_predictions['von_neumann'] = self.calculate_observables_mixed(rho_states_3, countries)
        
        # Model 4: Lindblad
        rho_states_4 = self.solve_lindblad(H, t_points)
        all_predictions['lindblad'] = self.calculate_observables_mixed(rho_states_4, countries)
        
        print("  Running Models 5-8 (Floquet)...")
        
        # Model 5: Floquet Schrödinger
        states_5 = self.solve_floquet_schrodinger(H, t_points)
        all_predictions['floquet_schrodinger'] = self.calculate_observables_pure(states_5, countries)
        
        # Model 6: Floquet Nonlinear Schrödinger
        states_6 = self.solve_floquet_schrodinger_nonlinear(H, t_points, nonlinear_strength=0.25)
        all_predictions['floquet_schrodinger_nonlinear'] = self.calculate_observables_pure(states_6, countries)
        
        # Model 7: Floquet von Neumann
        rho_states_7 = self.solve_floquet_von_neumann(H, t_points, decoherence_rate=0.08)
        all_predictions['floquet_von_neumann'] = self.calculate_observables_mixed(rho_states_7, countries)
        
        # Model 8: Floquet Lindblad
        rho_states_8 = self.solve_floquet_lindblad(H, t_points)
        all_predictions['floquet_lindblad'] = self.calculate_observables_mixed(rho_states_8, countries)
        
        print("  ✅ All 8 enhanced models completed!")
        return all_predictions

    def _project_to_physical_density_matrix(self, rho):
        """Project density matrix to physical subspace."""
        rho = (rho + np.conj(rho.T)) / 2
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        eigenvals = np.maximum(eigenvals, 0)
        rho = eigenvecs @ np.diag(eigenvals) @ np.conj(eigenvecs.T)
        trace_rho = np.trace(rho)
        if abs(trace_rho) > 1e-12:
            rho = rho / trace_rho
        else:
            rho = np.eye(rho.shape[0]) / rho.shape[0]
        return rho


class EnhancedProbabilisticModelEvaluator:
    """Enhanced probabilistic evaluation with improved metrics."""
    
    def __init__(self):
        self.evaluation_methods = [
            'log_likelihood', 'brier_score', 'calibration_slope', 
            'regime_detection', 'distributional_validation', 'coupling_validation',
            'quantum_coherence_metrics', 'model_differentiation'
        ]
    
    def comprehensive_evaluation(self, all_predictions, observed_data_continuous):
        """Comprehensive evaluation with enhanced metrics."""
        print("\n📊 ENHANCED PROBABILISTIC MODEL EVALUATION")
        print("=" * 60)
        
        observed_data_binary = {}
        for country, obs in observed_data_continuous.items():
            threshold = np.median(obs)
            observed_data_binary[country] = (obs > threshold).astype(int)
            print(f"  {country}: Binary threshold = {threshold:.3f}")
        
        model_metrics = {}
        
        # Model differentiation analysis
        differentiation_analysis = self.model_differentiation_analysis(all_predictions, list(observed_data_binary.keys()))
        print(f"\n  Model Differentiation Analysis:")
        print(f"    Average RMS difference: {differentiation_analysis['average_differentiation']:.3f}")
        print(f"    Floquet vs Standard: {differentiation_analysis['floquet_vs_standard_diff']:.3f}")
        
        for model_name, predictions in all_predictions.items():
            print(f"\n  Evaluating {model_name.replace('_', ' ').title()}...")
            
            metrics = {}
            
            for country in observed_data_binary.keys():
                if country in predictions:
                    pred = predictions[country]
                    obs_binary = observed_data_binary[country]
                    obs_continuous = observed_data_continuous[country]
                    
                    country_metrics = {
                        'log_likelihood': self.log_likelihood_score(pred, obs_binary),
                        'brier_score': self.brier_score(pred, obs_binary),
                        'calibration_slope': self.assess_calibration_slope(pred, obs_binary),
                        'regime_detection': self.regime_change_detection(pred, obs_continuous),
                        'distributional': self.distributional_validation(pred, obs_continuous),
                        'quantum_coherence': self.quantum_coherence_metrics(pred, obs_continuous)
                    }
                    
                    metrics[country] = country_metrics
            
            # Cross-country coupling
            if len(observed_data_binary) == 2:
                countries = list(observed_data_binary.keys())
                coupling_metrics = self.coupling_validation(
                    predictions[countries[0]], predictions[countries[1]],
                    observed_data_binary[countries[0]], observed_data_binary[countries[1]]
                )
                metrics['coupling'] = coupling_metrics
            
            # Overall scores
            if len(metrics) >= 2:
                country_metrics = [m for k, m in metrics.items() if isinstance(m, dict) and 'log_likelihood' in m]
                
                if country_metrics:
                    metrics['overall'] = {
                        'avg_log_likelihood': np.mean([m['log_likelihood'] for m in country_metrics]),
                        'avg_brier_score': np.mean([m['brier_score'] for m in country_metrics]),
                        'avg_calibration_slope': np.mean([m['calibration_slope'] for m in country_metrics]),
                        'avg_regime_f1': np.mean([m['regime_detection']['f1'] for m in country_metrics]),
                        'avg_distributional_similarity': np.mean([m['distributional']['overall_similarity'] for m in country_metrics]),
                        'avg_quantum_score': np.mean([m['quantum_coherence']['overall_quantum_score'] for m in country_metrics])
                    }
                    
                    # Enhanced composite score
                    ll_norm = (metrics['overall']['avg_log_likelihood'] + 2) / 2
                    brier_norm = 1 - metrics['overall']['avg_brier_score']
                    calib_norm = 1 - abs(1 - metrics['overall']['avg_calibration_slope'])
                    
                    composite = (ll_norm + brier_norm + calib_norm + 
                               metrics['overall']['avg_regime_f1'] + 
                               metrics['overall']['avg_distributional_similarity'] +
                               metrics['overall']['avg_quantum_score']) / 6
                    
                    metrics['overall']['composite_score'] = composite
            
            model_metrics[model_name] = metrics
        
        # Add differentiation metrics to overall results
        model_metrics['_differentiation_analysis'] = differentiation_analysis
        
        # Print enhanced summary
        self._print_enhanced_summary(model_metrics)
        
        return model_metrics

    def model_differentiation_analysis(self, all_predictions, countries):
        """Analyze how well different models produce distinguishable outputs."""
        model_names = list(all_predictions.keys())
        differentiation_matrix = np.zeros((len(model_names), len(model_names)))
        
        for country in countries:
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if model1 in all_predictions and model2 in all_predictions:
                        if country in all_predictions[model1] and country in all_predictions[model2]:
                            pred1 = all_predictions[model1][country]
                            pred2 = all_predictions[model2][country]
                            rms_diff = np.sqrt(np.mean((pred1 - pred2)**2))
                            differentiation_matrix[i, j] = rms_diff
        
        # Average differentiation
        avg_differentiation = np.mean(differentiation_matrix[np.triu_indices_from(differentiation_matrix, k=1)])
        
        # Floquet vs non-Floquet differentiation
        floquet_models = [m for m in model_names if 'floquet' in m]
        non_floquet_models = [m for m in model_names if 'floquet' not in m]
        
        floquet_differentiation = 0
        if floquet_models and non_floquet_models:
            floquet_diffs = []
            for f_model in floquet_models:
                for nf_model in non_floquet_models:
                    if f_model in all_predictions and nf_model in all_predictions:
                        for country in countries:
                            if (country in all_predictions[f_model] and 
                                country in all_predictions[nf_model]):
                                pred_f = all_predictions[f_model][country]
                                pred_nf = all_predictions[nf_model][country]
                                diff = np.sqrt(np.mean((pred_f - pred_nf)**2))
                                floquet_diffs.append(diff)
            
            floquet_differentiation = np.mean(floquet_diffs) if floquet_diffs else 0
        
        return {
            'differentiation_matrix': differentiation_matrix,
            'average_differentiation': avg_differentiation,
            'floquet_vs_standard_diff': floquet_differentiation,
            'model_names': model_names
        }

    def log_likelihood_score(self, predicted_probs, observed_outcomes):
        """Log-likelihood score for probabilistic predictions."""
        pred_clipped = np.clip(predicted_probs, 1e-10, 1-1e-10)
        ll = observed_outcomes * np.log(pred_clipped) + (1 - observed_outcomes) * np.log(1 - pred_clipped)
        return np.mean(ll)
    
    def brier_score(self, predicted_probs, observed_outcomes):
        """Brier score for probabilistic predictions."""
        return np.mean((predicted_probs - observed_outcomes) ** 2)
    
    def assess_calibration_slope(self, predicted_probs, observed_outcomes, n_bins=10):
        """Assess calibration quality using binned approach."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_probs = []
        bin_freqs = []
        
        for i in range(n_bins):
            mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i+1])
            if mask.sum() > 3:
                bin_prob = predicted_probs[mask].mean()
                bin_freq = observed_outcomes[mask].mean()
                bin_probs.append(bin_prob)
                bin_freqs.append(bin_freq)
        
        if len(bin_probs) < 3:
            return 0.0
        
        try:
            slope = np.polyfit(bin_probs, bin_freqs, 1)[0]
            return slope
        except:
            return 0.0
    
    def regime_change_detection(self, predicted_probs, observed_outcomes, threshold=None):
        """Enhanced regime detection with adaptive thresholds."""
        if len(predicted_probs) < 3:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if threshold is None:
            obs_changes = np.abs(np.diff(observed_outcomes))
            threshold = max(np.percentile(obs_changes, 75), 0.05)
        
        obs_changes = np.abs(np.diff(observed_outcomes)) > threshold
        pred_changes = np.abs(np.diff(predicted_probs)) > threshold
        
        if np.sum(pred_changes) == 0:
            precision = 0.0
        else:
            precision = np.sum(obs_changes & pred_changes) / np.sum(pred_changes)
        
        if np.sum(obs_changes) == 0:
            recall = 1.0 if np.sum(pred_changes) == 0 else 0.0
        else:
            recall = np.sum(obs_changes & pred_changes) / np.sum(obs_changes)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    def distributional_validation(self, predicted_probs, observed_outcomes):
        """Compare statistical properties of predicted vs observed series."""
        results = {}
        
        pred_vol = np.std(predicted_probs)
        obs_vol = np.std(observed_outcomes)
        results['volatility_ratio'] = pred_vol / obs_vol if obs_vol > 0 else 0
        
        if len(predicted_probs) > 1:
            pred_acf = np.corrcoef(predicted_probs[:-1], predicted_probs[1:])[0,1]
            obs_acf = np.corrcoef(observed_outcomes[:-1], observed_outcomes[1:])[0,1]
            if not np.isnan(pred_acf) and not np.isnan(obs_acf) and obs_acf != 0:
                results['persistence_ratio'] = pred_acf / obs_acf
            else:
                results['persistence_ratio'] = 0
        else:
            results['persistence_ratio'] = 0
        
        try:
            results['skew_diff'] = abs(skew(predicted_probs) - skew(observed_outcomes))
            results['kurtosis_diff'] = abs(kurtosis(predicted_probs) - kurtosis(observed_outcomes))
        except:
            results['skew_diff'] = 0
            results['kurtosis_diff'] = 0
        
        vol_score = abs(1 - results['volatility_ratio'])
        pers_score = abs(1 - results['persistence_ratio'])
        shape_score = (results['skew_diff'] + results['kurtosis_diff']) / 2
        
        results['overall_similarity'] = 1 / (1 + vol_score + pers_score + shape_score)
        
        return results
    
    def coupling_validation(self, country1_probs, country2_probs, country1_obs, country2_obs):
        """Compare predicted vs observed cross-country coupling."""
        pred_coupling = np.corrcoef(country1_probs, country2_probs)[0,1]
        obs_coupling = np.corrcoef(country1_obs, country2_obs)[0,1]
        
        if np.isnan(pred_coupling):
            pred_coupling = 0
        if np.isnan(obs_coupling):
            obs_coupling = 0
        
        coupling_error = abs(pred_coupling - obs_coupling)
        
        window = min(12, len(country1_probs) // 3)
        if window >= 3:
            pred_dynamic = []
            obs_dynamic = []
            
            for i in range(window, len(country1_probs)):
                pred_corr = np.corrcoef(country1_probs[i-window:i], country2_probs[i-window:i])[0,1]
                obs_corr = np.corrcoef(country1_obs[i-window:i], country2_obs[i-window:i])[0,1]
                
                if not np.isnan(pred_corr) and not np.isnan(obs_corr):
                    pred_dynamic.append(pred_corr)
                    obs_dynamic.append(obs_corr)
            
            if len(pred_dynamic) > 3:
                dynamic_similarity = np.corrcoef(pred_dynamic, obs_dynamic)[0,1]
                if np.isnan(dynamic_similarity):
                    dynamic_similarity = 0
            else:
                dynamic_similarity = 0
        else:
            dynamic_similarity = 0
        
        return {
            'static_coupling_error': coupling_error,
            'dynamic_coupling_similarity': dynamic_similarity,
            'predicted_coupling': pred_coupling,
            'observed_coupling': obs_coupling
        }
    
    def quantum_coherence_metrics(self, predicted_probs, observed_outcomes):
        """Metrics specifically designed for quantum-like behavior evaluation."""
        # Oscillation coherence
        fft_pred = np.fft.fft(predicted_probs)
        fft_obs = np.fft.fft(observed_outcomes)
        freq_similarity = np.abs(np.corrcoef(np.abs(fft_pred), np.abs(fft_obs))[0,1])
        if np.isnan(freq_similarity):
            freq_similarity = 0
        
        # Phase coherence
        phase_pred = np.angle(fft_pred)
        phase_obs = np.angle(fft_obs)
        phase_similarity = np.cos(np.mean(np.abs(phase_pred - phase_obs)))
        
        # Decoherence pattern matching
        envelope_pred = gaussian_filter1d(np.abs(predicted_probs - np.mean(predicted_probs)), sigma=2)
        envelope_obs = gaussian_filter1d(np.abs(observed_outcomes - np.mean(observed_outcomes)), sigma=2)
        envelope_similarity = np.corrcoef(envelope_pred, envelope_obs)[0,1]
        if np.isnan(envelope_similarity):
            envelope_similarity = 0
        
        return {
            'frequency_similarity': freq_similarity,
            'phase_similarity': phase_similarity,
            'envelope_similarity': envelope_similarity,
            'overall_quantum_score': (freq_similarity + phase_similarity + envelope_similarity) / 3
        }
    
    def _print_enhanced_summary(self, model_metrics):
        """Print enhanced summary with model differentiation."""
        print(f"\n📋 ENHANCED MODEL PERFORMANCE SUMMARY")
        print("=" * 50)
        
        ranked_models = []
        for model_name, metrics in model_metrics.items():
            if model_name.startswith('_'):
                continue
            if 'overall' in metrics and 'composite_score' in metrics['overall']:
                score = metrics['overall']['composite_score']
                ranked_models.append((model_name, score))
        
        ranked_models.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, score) in enumerate(ranked_models):
            display_name = model_name.replace('_', ' ').title()
            print(f"  {i+1}. {display_name}: {score:.3f}")
            
            metrics = model_metrics[model_name]
            if 'overall' in metrics:
                print(f"     Quantum Score: {metrics['overall']['avg_quantum_score']:.3f}")
                print(f"     Regime Detection: {metrics['overall']['avg_regime_f1']:.3f}")
                print(f"     Calibration: {metrics['overall']['avg_calibration_slope']:.3f}")
        
        # Model differentiation summary
        if '_differentiation_analysis' in model_metrics:
            diff_analysis = model_metrics['_differentiation_analysis']
            print(f"\n🔄 MODEL DIFFERENTIATION:")
            print(f"  Average difference: {diff_analysis['average_differentiation']:.3f}")
            print(f"  Floquet advantage: {diff_analysis['floquet_vs_standard_diff']:.3f}")
            
            if diff_analysis['average_differentiation'] > 0.05:
                print("  ✅ Models are well-differentiated")
            else:
                print("  ⚠️  Models show similar behavior")


class EnhancedQuantumTradeAnalysis:
    """Main analysis class with enhanced configurability."""

    def __init__(self, output_dir="enhanced_quantum_configurable", 
                 synthetic_config: Optional[QuantumSyntheticConfig] = None,
                 quantum_config: Optional[QuantumModelConfig] = None,
                 enable_plotting: bool = True):
        self.output_dir = output_dir
        self.enable_plotting = enable_plotting and PLOTTING_AVAILABLE
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/plots/essential", exist_ok=True)
        os.makedirs(f"{output_dir}/plots/detailed", exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        os.makedirs(f"{output_dir}/configs", exist_ok=True)
        print(f"📁 Output directory: {output_dir}/")
        
        if not self.enable_plotting:
            print("⚠️  Plotting disabled - will create text-based alternatives")
        
        # Use provided configs or create defaults
        self.synthetic_config = synthetic_config or QuantumSyntheticConfig()
        self.quantum_config = quantum_config or QuantumModelConfig()
        
        # Initialize components
        self.synthetic_generator = ConfigurableSyntheticDataGenerator(self.synthetic_config)
        self.quantum_framework = EnhancedQuantumTradeEvolutionFramework(self.quantum_config)
        self.evaluator = EnhancedProbabilisticModelEvaluator()
        
        # Save configurations
        self.save_configurations()

    def save_configurations(self):
        """Save current configurations to JSON files for reproducibility."""
        try:
            with open(f'{self.output_dir}/configs/synthetic_config.json', 'w') as f:
                config_dict = {
                    'start_year': self.synthetic_config.start_year,
                    'end_year': self.synthetic_config.end_year,
                    'coherence_time_country1': self.synthetic_config.coherence_time_country1,
                    'coherence_time_country2': self.synthetic_config.coherence_time_country2,
                    'entanglement_strength': self.synthetic_config.entanglement_strength,
                    'regime_change_times': self.synthetic_config.regime_change_times,
                    'regime_change_amplitudes': self.synthetic_config.regime_change_amplitudes,
                    'quantum_noise_level': self.synthetic_config.quantum_noise_level
                }
                json.dump(config_dict, f, indent=2)
            
            with open(f'{self.output_dir}/configs/quantum_config.json', 'w') as f:
                config_dict = {
                    'coupling_strength': self.quantum_config.coupling_strength,
                    'temperature': self.quantum_config.temperature,
                    'floquet_frequency': self.quantum_config.floquet_frequency,
                    'floquet_amplitude': self.quantum_config.floquet_amplitude,
                    'country_params': self.quantum_config.country_params
                }
                json.dump(config_dict, f, indent=2)
            
            print("📋 Configurations saved to configs/ directory")
        except Exception as e:
            print(f"⚠️  Could not save configurations: {e}")

    def load_data(self, countries: List[str]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Load quantum-appropriate synthetic data."""
        print("🔄 Generating quantum-appropriate synthetic data...")
        return self.synthetic_generator.generate_quantum_synthetic_data(countries)

    def construct_hamiltonian(self, countries: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Construct Hamiltonian from quantum configuration."""
        print("⚛️ Constructing enhanced Hamiltonian...")
        
        n_countries = len(countries)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        H = np.zeros((2**n_countries, 2**n_countries), dtype=complex)

        # Individual country Hamiltonians with enhanced parameters
        for i, country in enumerate(countries):
            params = self.quantum_config.country_params[country]
            
            # Enhanced single-country Hamiltonian
            H_country = (params['delta'] * sigma_z + 
                        params['Delta'] * (sigma_x * np.cos(params['phi']) + 
                                         sigma_y * np.sin(params['phi'])))

            ops = [I] * n_countries
            ops[i] = H_country
            H_i = ops[0]
            for op in ops[1:]:
                H_i = np.kron(H_i, op)

            H += H_i

        # Enhanced coupling between countries
        if n_countries == 2:
            # Multiple coupling terms for richer dynamics
            H_coupling_xx = np.kron(sigma_x, sigma_x)
            H_coupling_yy = np.kron(sigma_y, sigma_y)
            H_coupling_zz = np.kron(sigma_z, sigma_z)
            
            H += (self.quantum_config.coupling_strength * H_coupling_xx +
                  0.3 * self.quantum_config.coupling_strength * H_coupling_yy +
                  0.1 * self.quantum_config.coupling_strength * H_coupling_zz)

        print(f"  Hamiltonian: {H.shape[0]}×{H.shape[0]} matrix")
        print(f"  Coupling strength: {self.quantum_config.coupling_strength:.2f}")
        
        return H, countries

    def create_comparison_plots(self, years, observed_data, all_predictions, all_metrics):
        """Create comprehensive comparison plots with robust error handling."""
        if not self.enable_plotting:
            print("📊 Plotting disabled - creating text-based alternatives")
            self._create_text_alternatives(years, observed_data, all_predictions, all_metrics)
            return
        
        print("🎨 Creating enhanced comparison plots...")
        
        try:
            # Main evolution plot
            print("  Creating evolution plot...")
            self._create_evolution_plot(years, observed_data, all_predictions)
            
            # Model differentiation plot  
            print("  Creating differentiation plot...")
            self._create_differentiation_plot(all_predictions, all_metrics)
            
            # Performance metrics plot
            print("  Creating performance plot...")
            self._create_performance_plot(all_metrics)
            
            print("📊 Enhanced plots created!")
            
        except Exception as e:
            print(f"❌ Critical plotting error: {e}")
            print("  📄 Analysis will continue with data output only")
            self._create_text_alternatives(years, observed_data, all_predictions, all_metrics)

    def _create_evolution_plot(self, years, observed_data, all_predictions):
        """Create main evolution comparison plot."""
        try:
            fig = plt.figure(figsize=(20, 10))
            countries = list(observed_data.keys())
            
            model_names = ['schrodinger', 'schrodinger_nonlinear', 'von_neumann', 'lindblad',
                          'floquet_schrodinger', 'floquet_schrodinger_nonlinear', 
                          'floquet_von_neumann', 'floquet_lindblad']
            
            for idx, model in enumerate(model_names):
                ax = fig.add_subplot(2, 4, idx + 1)
                
                # Plot observed data
                for i, country in enumerate(countries):
                    style = '-' if i == 0 else '--'
                    ax.plot(years, observed_data[country], 'k' + style, 
                           linewidth=2, alpha=0.8, label=f'{country} (Obs)')
                
                # Plot model predictions
                if model in all_predictions:
                    for i, country in enumerate(countries):
                        if country in all_predictions[model]:
                            style = '-' if i == 0 else '--'
                            color = self.quantum_framework.model_colors[model]
                            pred = all_predictions[model][country]
                            ax.plot(years, pred, color=color, linestyle=style,
                                   alpha=0.7, linewidth=2, 
                                   label=f'{country} (Pred)' if i == 0 else "")
                
                ax.set_title(model.replace('_', ' ').title(), fontweight='bold')
                ax.set_ylabel('Policy Restriction Probability')
                ax.grid(True, alpha=0.3)
                if idx == 0:
                    ax.legend()
            
            plt.suptitle('Enhanced 8-Model Quantum Evolution Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/essential/enhanced_evolution_comparison.png',
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            print(f"  ⚠️  Error creating evolution plot: {e}")
            self._create_fallback_evolution_plot(years, observed_data, all_predictions)

    def _create_fallback_evolution_plot(self, years, observed_data, all_predictions):
        """Create simplified fallback plot when main plotting fails."""
        try:
            countries = list(observed_data.keys())
            model_names = ['schrodinger', 'von_neumann', 'lindblad', 'floquet_schrodinger']
            
            plt.figure(figsize=(15, 8))
            
            for i, country in enumerate(countries):
                plt.subplot(1, len(countries), i + 1)
                
                # Plot observed
                plt.plot(years, observed_data[country], 'k-', 
                        linewidth=3, alpha=0.8, label=f'{country} (Observed)')
                
                # Plot key models
                colors = ['blue', 'green', 'red', 'purple']
                for j, model in enumerate(model_names):
                    if model in all_predictions and country in all_predictions[model]:
                        plt.plot(years, all_predictions[model][country], 
                               color=colors[j], alpha=0.7, linewidth=2,
                               label=model.replace('_', ' ').title())
                
                plt.title(f'{country}', fontweight='bold')
                plt.ylabel('Policy Probability')
                plt.xlabel('Year')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.suptitle('Key Quantum Models Comparison (Fallback)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/essential/fallback_evolution_comparison.png',
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print("  ✅ Fallback plot created successfully")
            
        except Exception as e:
            print(f"  ❌ Even fallback plot failed: {e}")

    def _create_differentiation_plot(self, all_predictions, all_metrics):
        """Create model differentiation analysis plot."""
        try:
            if '_differentiation_analysis' not in all_metrics:
                print("  ⚠️  No differentiation analysis data available")
                return
            
            diff_analysis = all_metrics['_differentiation_analysis']
            diff_matrix = diff_analysis['differentiation_matrix']
            model_names = diff_analysis['model_names']
            
            fig = plt.figure(figsize=(15, 6))
            
            # Subplot 1: Differentiation matrix heatmap
            ax1 = fig.add_subplot(1, 2, 1)
            im = ax1.imshow(diff_matrix, cmap='viridis', aspect='auto')
            ax1.set_xticks(range(len(model_names)))
            ax1.set_yticks(range(len(model_names)))
            ax1.set_xticklabels([m.replace('_', '\n') for m in model_names], rotation=45)
            ax1.set_yticklabels([m.replace('_', '\n') for m in model_names])
            ax1.set_title('Model Differentiation Matrix\n(RMS Differences)', fontweight='bold')
            
            try:
                plt.colorbar(im, ax=ax1)
            except Exception:
                pass
            
            # Subplot 2: Summary plot
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.axis('off')
            
            summary_text = "MODEL DIFFERENTIATION SUMMARY\n" + "="*25 + "\n\n"
            summary_text += f"Average difference: {diff_analysis['average_differentiation']:.3f}\n"
            summary_text += f"Floquet vs Standard: {diff_analysis['floquet_vs_standard_diff']:.3f}\n\n"
            
            if diff_analysis['average_differentiation'] > 0.05:
                summary_text += "✅ Models are well-differentiated\n"
            else:
                summary_text += "⚠️  Models show similar behavior\n"
            
            ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/detailed/model_differentiation.png',
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            print(f"  ⚠️  Error creating differentiation plot: {e}")

    def _create_performance_plot(self, all_metrics):
        """Create performance metrics comparison plot."""
        try:
            fig = plt.figure(figsize=(15, 12))
            
            model_names = [m for m in all_metrics.keys() if not m.startswith('_')]
            
            if not model_names:
                print("  ⚠️  No model metrics available for plotting")
                return
            
            # Extract scores
            composite_scores = []
            quantum_scores = []
            regime_f1_scores = []
            
            for model in model_names:
                if 'overall' in all_metrics[model]:
                    overall = all_metrics[model]['overall']
                    composite_scores.append(overall.get('composite_score', 0))
                    quantum_scores.append(overall.get('avg_quantum_score', 0))
                    regime_f1_scores.append(overall.get('avg_regime_f1', 0))
                else:
                    composite_scores.append(0)
                    quantum_scores.append(0)
                    regime_f1_scores.append(0)
            
            # Plot 1: Composite scores
            ax1 = fig.add_subplot(2, 2, 1)
            bars1 = ax1.bar(range(len(model_names)), composite_scores, alpha=0.7)
            ax1.set_title('Model Composite Scores', fontweight='bold')
            ax1.set_ylabel('Composite Score')
            ax1.set_xticks(range(len(model_names)))
            ax1.set_xticklabels([m.replace('_', '\n') for m in model_names], rotation=45)
            
            # Plot 2: Quantum coherence scores
            ax2 = fig.add_subplot(2, 2, 2)
            bars2 = ax2.bar(range(len(model_names)), quantum_scores, alpha=0.7, color='green')
            ax2.set_title('Quantum Coherence Scores', fontweight='bold')
            ax2.set_ylabel('Quantum Score')
            ax2.set_xticks(range(len(model_names)))
            ax2.set_xticklabels([m.replace('_', '\n') for m in model_names], rotation=45)
            
            # Plot 3: Regime detection F1
            ax3 = fig.add_subplot(2, 2, 3)
            bars3 = ax3.bar(range(len(model_names)), regime_f1_scores, alpha=0.7, color='red')
            ax3.set_title('Regime Detection F1 Scores', fontweight='bold')
            ax3.set_ylabel('F1 Score')
            ax3.set_xticks(range(len(model_names)))
            ax3.set_xticklabels([m.replace('_', '\n') for m in model_names], rotation=45)
            
            # Plot 4: Summary text
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis('off')
            
            summary_text = "ENHANCED ANALYSIS SUMMARY\n" + "="*25 + "\n\n"
            
            if composite_scores and max(composite_scores) > 0:
                best_idx = np.argmax(composite_scores)
                best_model = model_names[best_idx]
                summary_text += f"Best Overall: {best_model.replace('_', ' ').title()}\n"
                summary_text += f"Score: {max(composite_scores):.3f}\n\n"
            
            avg_differentiation = 0
            if '_differentiation_analysis' in all_metrics:
                avg_differentiation = all_metrics['_differentiation_analysis'].get('average_differentiation', 0)
            
            summary_text += f"Model Differentiation: {avg_differentiation:.3f}\n"
            summary_text += f"{'Good' if avg_differentiation > 0.05 else 'Needs Improvement'}\n\n"
            
            if regime_f1_scores:
                summary_text += f"Avg Regime Detection: {np.mean(regime_f1_scores):.3f}\n"
            if quantum_scores:
                summary_text += f"Avg Quantum Score: {np.mean(quantum_scores):.3f}\n\n"
            
            summary_text += "Check console output for\ndetailed results!"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/essential/performance_summary.png',
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            print(f"  ⚠️  Error creating performance plot: {e}")
            self._create_text_summary(all_metrics)

    def _create_text_summary(self, all_metrics):
        """Create text-based summary when plotting fails completely."""
        try:
            summary_path = f'{self.output_dir}/plots/essential/text_summary.txt'
            
            with open(summary_path, 'w') as f:
                f.write("PERFORMANCE SUMMARY (Text Version)\n")
                f.write("=" * 40 + "\n\n")
                
                model_names = [m for m in all_metrics.keys() if not m.startswith('_')]
                
                for model in model_names:
                    if 'overall' in all_metrics[model]:
                        overall = all_metrics[model]['overall']
                        f.write(f"{model.replace('_', ' ').title()}:\n")
                        f.write(f"  Composite: {overall.get('composite_score', 0):.3f}\n")
                        f.write(f"  Quantum: {overall.get('avg_quantum_score', 0):.3f}\n")
                        f.write(f"  Regime F1: {overall.get('avg_regime_f1', 0):.3f}\n\n")
            
            print(f"  ✅ Text summary created: {summary_path}")
            
        except Exception as e:
            print(f"  ❌ Could not create text summary: {e}")

    def _create_text_alternatives(self, years, observed_data, all_predictions, all_metrics):
        """Create text-based analysis when plotting completely fails."""
        try:
            alt_path = f'{self.output_dir}/text_analysis_alternative.txt'
            
            with open(alt_path, 'w') as f:
                f.write("QUANTUM TRADE ANALYSIS - TEXT ALTERNATIVE\n")
                f.write("=" * 50 + "\n\n")
                
                # Data summary
                f.write("DATA SUMMARY\n")
                f.write("-" * 15 + "\n")
                for country, data in observed_data.items():
                    f.write(f"{country}:\n")
                    f.write(f"  Mean: {np.mean(data):.3f}\n")
                    f.write(f"  Std: {np.std(data):.3f}\n")
                    f.write(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]\n\n")
                
                # Model comparison
                f.write("MODEL PREDICTIONS SUMMARY\n")
                f.write("-" * 25 + "\n")
                for model, predictions in all_predictions.items():
                    f.write(f"\n{model.replace('_', ' ').title()}:\n")
                    for country, pred in predictions.items():
                        f.write(f"  {country}: mean={np.mean(pred):.3f}, std={np.std(pred):.3f}\n")
                
                # Performance metrics
                if all_metrics:
                    f.write("\nPERFORMANCE METRICS\n")
                    f.write("-" * 20 + "\n")
                    
                    model_scores = []
                    for model, metrics in all_metrics.items():
                        if not model.startswith('_') and 'overall' in metrics:
                            overall = metrics['overall']
                            score = overall.get('composite_score', 0)
                            model_scores.append((model, score))
                            
                            f.write(f"\n{model.replace('_', ' ').title()}:\n")
                            f.write(f"  Composite Score: {score:.3f}\n")
                            f.write(f"  Regime Detection F1: {overall.get('avg_regime_f1', 0):.3f}\n")
                            f.write(f"  Quantum Score: {overall.get('avg_quantum_score', 0):.3f}\n")
                    
                    # Best model
                    if model_scores:
                        best_model, best_score = max(model_scores, key=lambda x: x[1])
                        f.write(f"\nBEST MODEL: {best_model.replace('_', ' ').title()}\n")
                        f.write(f"Score: {best_score:.3f}\n")
            
            print(f"✅ Text alternative created: {alt_path}")
            
        except Exception as e:
            print(f"❌ Could not create text alternative: {e}")

    def save_results(self, years, observed_data, all_predictions, all_metrics):
        """Save all results and configurations."""
        print("\n💾 Saving enhanced results...")
        
        try:
            # Save observed data
            data_df = pd.DataFrame({'Year': years})
            data_df.update(observed_data)
            data_df.to_csv(f'{self.output_dir}/data/observed_data.csv', index=False)
            
            # Save model predictions
            for model_name, predictions in all_predictions.items():
                pred_df = pd.DataFrame({'Year': years})
                pred_df.update(predictions)
                pred_df.to_csv(f'{self.output_dir}/data/{model_name}_predictions.csv', index=False)
            
            # Save metrics
            metrics_serializable = {}
            for model, metrics in all_metrics.items():
                metrics_serializable[model] = self._make_json_serializable(metrics)
            
            with open(f'{self.output_dir}/data/evaluation_metrics.json', 'w') as f:
                json.dump(metrics_serializable, f, indent=2)
            
            # Save summary report
            self._create_summary_report(all_metrics)
            
            print(f"📁 Enhanced results saved to: {self.output_dir}/")
            
        except Exception as e:
            print(f"  ⚠️  Error saving results: {e}")

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and complex numbers to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': float(obj.real), 'imag': float(obj.imag)}
        elif np.isnan(obj) if isinstance(obj, (float, np.floating)) else False:
            return None
        else:
            return obj

    def _create_summary_report(self, all_metrics):
        """Create a comprehensive summary report."""
        report_path = f'{self.output_dir}/enhanced_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("ENHANCED QUANTUM TRADE POLICY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Configuration summary
            f.write("CONFIGURATION SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Synthetic Data Config:\n")
            f.write(f"  Coherence times: {self.synthetic_config.coherence_time_country1:.1f}, {self.synthetic_config.coherence_time_country2:.1f} years\n")
            f.write(f"  Entanglement strength: {self.synthetic_config.entanglement_strength:.2f}\n")
            f.write(f"  Regime changes at: {self.synthetic_config.regime_change_times}\n")
            f.write(f"  External drive frequency: {self.synthetic_config.external_drive_frequency:.2f}\n\n")
            
            f.write(f"Quantum Model Config:\n")
            f.write(f"  Coupling strength: {self.quantum_config.coupling_strength:.2f}\n")
            f.write(f"  Floquet frequency: {self.quantum_config.floquet_frequency:.2f}\n")
            f.write(f"  Temperature: {self.quantum_config.temperature:.2f}\n\n")
            
            # Model performance ranking
            f.write("MODEL PERFORMANCE RANKING\n")
            f.write("-" * 25 + "\n")
            
            ranked_models = []
            for model_name, metrics in all_metrics.items():
                if model_name.startswith('_'):
                    continue
                if 'overall' in metrics and 'composite_score' in metrics['overall']:
                    score = metrics['overall']['composite_score']
                    ranked_models.append((model_name, score, metrics['overall']))
            
            ranked_models.sort(key=lambda x: x[1], reverse=True)
            
            for i, (model_name, score, overall_metrics) in enumerate(ranked_models):
                f.write(f"{i+1}. {model_name.replace('_', ' ').title()}\n")
                f.write(f"   Composite Score: {score:.3f}\n")
                f.write(f"   Log-Likelihood: {overall_metrics.get('avg_log_likelihood', 0):.3f}\n")
                f.write(f"   Brier Score: {overall_metrics.get('avg_brier_score', 0):.3f}\n")
                f.write(f"   Regime Detection F1: {overall_metrics.get('avg_regime_f1', 0):.3f}\n")
                f.write(f"   Quantum Score: {overall_metrics.get('avg_quantum_score', 0):.3f}\n\n")
            
            # Model differentiation analysis
            if '_differentiation_analysis' in all_metrics:
                diff_analysis = all_metrics['_differentiation_analysis']
                f.write("MODEL DIFFERENTIATION ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average RMS difference between models: {diff_analysis['average_differentiation']:.3f}\n")
                f.write(f"Floquet vs Standard difference: {diff_analysis['floquet_vs_standard_diff']:.3f}\n")
                
                if diff_analysis['average_differentiation'] > 0.05:
                    f.write("✅ Models are well-differentiated\n\n")
                else:
                    f.write("⚠️ Models show similar behavior - consider parameter adjustment\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            if ranked_models:
                best_model = ranked_models[0][0]
                f.write(f"• Best performing model: {best_model.replace('_', ' ').title()}\n")
                
                if 'floquet' in best_model:
                    f.write("• Floquet (driven) models perform best - periodic external influences are important\n")
                else:
                    f.write("• Standard models perform best - system may be dominated by intrinsic dynamics\n")
            
            avg_regime_f1 = np.mean([m[2].get('avg_regime_f1', 0) for m in ranked_models])
            if avg_regime_f1 > 0.3:
                f.write("• Good regime detection capability - models capture policy transitions well\n")
            else:
                f.write("• Poor regime detection - consider adjusting synthetic data parameters\n")
            
            if '_differentiation_analysis' in all_metrics:
                if all_metrics['_differentiation_analysis']['average_differentiation'] < 0.05:
                    f.write("• Increase model parameter differences for better differentiation\n")
                    f.write("• Consider stronger nonlinear effects or coupling strengths\n")
            
            f.write("\n")
            f.write("For detailed results, see CSV files and plots in the output directory.\n")
        
        print(f"📄 Summary report saved to: {report_path}")

    def run_enhanced_analysis(self, countries: List[str] = None) -> Dict:
        """
        Run the complete enhanced quantum trade policy analysis.
        
        Args:
            countries: List of countries to analyze (default: ['Egypt', 'Russian Federation'])
        
        Returns:
            Dictionary containing all analysis results
        """
        print("🚀 ENHANCED CONFIGURABLE QUANTUM TRADE ANALYSIS")
        print("=" * 70)
        print("🎯 Key Features:")
        print("  ✅ Configurable synthetic data generation")
        print("  ✅ Quantum-appropriate synthetic patterns")
        print("  ✅ Enhanced model differentiation")
        print("  ✅ Comprehensive evaluation metrics")
        if not self.enable_plotting:
            print("  ⚠️  Text-based output (matplotlib unavailable)")
        print("")

        # Default countries if not specified
        if countries is None:
            countries = ['Egypt', 'Russian Federation']

        try:
            # Stage 1: Data Loading/Generation
            print("📊 STAGE 1: Data Loading/Generation")
            print("-" * 40)
            
            ts_df, observed_data = self.load_data(countries)
            years = ts_df['Year'].values
            t_points = np.linspace(0, years[-1] - years[0], len(ts_df))

            print(f"  Data loaded: {len(ts_df)} time points")
            print(f"  Time range: {years[0]:.1f} - {years[-1]:.1f}")
            
            # Print data characteristics
            for country, data in observed_data.items():
                print(f"  {country}: mean={np.mean(data):.3f}, std={np.std(data):.3f}")

            # Stage 2: Quantum System Construction
            print("\n⚛️ STAGE 2: Quantum System Construction")
            print("-" * 40)
            
            H, countries = self.construct_hamiltonian(countries)
            
            # Print Hamiltonian characteristics
            eigenvals, _ = np.linalg.eigh(H)
            print(f"  Energy spectrum: [{np.min(eigenvals):.2f}, {np.max(eigenvals):.2f}]")
            print(f"  Energy gap: {np.min(np.diff(np.sort(eigenvals))):.3f}")

            # Stage 3: Model Parameter Summary
            print("\n🔧 STAGE 3: Model Parameter Summary")
            print("-" * 40)
            
            for country in countries:
                params = self.quantum_config.country_params[country]
                print(f"  {country}:")
                print(f"    Energy scale (E0): {params['E0']:.2f}")
                print(f"    Tunneling (Δ): {params['Delta']:.2f}")
                print(f"    Detuning (δ): {params['delta']:.2f}")
                print(f"    Damping (γ): {params['gamma']:.3f}")
                print(f"    Phase (φ): {params['phi']:.2f}")

            # Stage 4: 8-Model Quantum Evolution
            print("\n🔬 STAGE 4: 8-Model Quantum Evolution")
            print("-" * 40)
            
            print(f"  Floquet parameters:")
            print(f"    Drive frequency: {self.quantum_config.floquet_frequency:.2f}")
            print(f"    Drive amplitude: {self.quantum_config.floquet_amplitude:.2f}")
            print(f"    Drive phase: {self.quantum_config.floquet_phase:.2f}")
            
            all_predictions = self.quantum_framework.run_all_models(H, t_points, countries)

            # Stage 5: Enhanced Evaluation
            print("\n📊 STAGE 5: Enhanced Probabilistic Evaluation")
            print("-" * 40)
            
            all_metrics = self.evaluator.comprehensive_evaluation(all_predictions, observed_data)

            # Stage 6: Visualization
            print("\n🎨 STAGE 6: Enhanced Visualization Suite")
            print("-" * 50)
            
            # Original plots
            self.create_comparison_plots(years, observed_data, all_predictions, all_metrics)
            
            # NEW: Enhanced quantum expert and time series plots
            self.create_enhanced_quantum_and_timeseries_plots(years, observed_data, all_predictions, all_metrics, H, t_points)


            # Stage 7: Results Saving
            print("\n💾 STAGE 7: Saving Results")
            print("-" * 40)
            
            self.save_results(years, observed_data, all_predictions, all_metrics)

            # Stage 8: Summary and Recommendations
            print("\n📋 STAGE 8: Analysis Summary")
            print("-" * 40)
            
            self._print_final_summary(all_metrics)

            print("\n" + "="*70)
            print("🎉 ENHANCED ANALYSIS COMPLETE!")
            print("="*70)
            print(f"📁 All results saved to: {self.output_dir}/")
            print("📊 Key outputs:")
            if self.enable_plotting:
                print("  - enhanced_evolution_comparison.png (main results)")
                print("  - performance_summary.png (model comparison)")
                print("  - model_differentiation.png (detailed analysis)")
            else:
                print("  - text_analysis_alternative.txt (text-based results)")
            print("  - enhanced_analysis_report.txt (summary)")
            print("  - CSV files with all data and predictions")
            print("  - JSON configs for reproducibility")

            return {
                'observed_data': observed_data,
                'quantum_config': self.quantum_config,
                'synthetic_config': self.synthetic_config,
                'all_predictions': all_predictions,
                'all_metrics': all_metrics,
                'years': years,
                'hamiltonian': H,
                'plotting_available': self.enable_plotting
            }
            
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in analysis: {e}")
            print("Analysis failed. Check your environment and dependencies.")
            import traceback
            print(f"Full error trace:\n{traceback.format_exc()}")
            
            return {
                'error': str(e),
                'plotting_available': self.enable_plotting,
                'status': 'failed'
            }

    # =========================================================================
    #  Stage 6  
    def create_enhanced_quantum_and_timeseries_plots(self, years, observed_data, all_predictions, all_metrics, H, t_points):
        """
        Enhanced plotting suite - BOTH COUNTRIES VERSION
        """
        print("🔬 Creating enhanced plots for ALL countries (save-only mode)...")
        
        countries = list(observed_data.keys())
        print(f"📊 Countries in analysis: {countries}")
        
        # Force non-interactive backend to prevent hanging
        plt.ioff()
        
        # ========================================================================
        # PLOT 1: QUANTUM EXPERT ANALYSIS (BOTH COUNTRIES)
        # ========================================================================
        try:
            print("  📊 Creating quantum expert analysis for both countries...")
            
            fig1 = plt.figure(figsize=(24, 20))  # Even larger for both countries
            gs1 = GridSpec(5, 3, figure=fig1, hspace=0.4, wspace=0.35)
            
            # 1. Quantum Coherence Analysis
            ax1 = fig1.add_subplot(gs1[0, 0])
            ax1.set_title('Quantum Coherence Analysis', fontweight='bold', fontsize=12)
            
            try:
                for model_name, predictions in all_predictions.items():
                    if 'floquet' not in model_name and len(countries) >= 2:
                        country1_data = predictions[countries[0]]
                        country2_data = predictions[countries[1]]
                        window = 12
                        coherence_proxy = []
                        
                        for i in range(len(years)):
                            if i >= window:
                                try:
                                    corr = np.corrcoef(country1_data[i-window:i+1], 
                                                     country2_data[i-window:i+1])[0,1]
                                    coherence_proxy.append(abs(corr) if not np.isnan(corr) else 0)
                                except:
                                    coherence_proxy.append(0)
                            else:
                                coherence_proxy.append(1.0)
                        
                        if coherence_proxy:
                            color = self.quantum_framework.model_colors.get(model_name, 'blue')
                            ax1.plot(years[:len(coherence_proxy)], coherence_proxy, 
                                   label=model_name.replace('_', ' ').title(), 
                                   color=color, linewidth=2, alpha=0.8)
                
                ax1.set_xlabel('Year', fontsize=10)
                ax1.set_ylabel('Coherence Measure', fontsize=10)
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
            except Exception as e:
                ax1.text(0.5, 0.5, f'Coherence analysis\nerror: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=10)
            
            # 2. Entanglement Dynamics
            ax2 = fig1.add_subplot(gs1[0, 1])
            ax2.set_title('Cross-Country Entanglement', fontweight='bold', fontsize=12)
            
            try:
                if len(countries) >= 2:
                    window = 12
                    obs_corr = []
                    pred_corrs = {model: [] for model in list(all_predictions.keys())[:3]}
                    
                    for i in range(window, min(len(years), 60)):
                        # Observed correlation
                        try:
                            corr = np.corrcoef(observed_data[countries[0]][i-window:i], 
                                             observed_data[countries[1]][i-window:i])[0,1]
                            obs_corr.append(corr if not np.isnan(corr) else 0)
                        except:
                            obs_corr.append(0)
                        
                        # Model correlations
                        for model in pred_corrs.keys():
                            try:
                                corr = np.corrcoef(all_predictions[model][countries[0]][i-window:i],
                                                 all_predictions[model][countries[1]][i-window:i])[0,1]
                                pred_corrs[model].append(corr if not np.isnan(corr) else 0)
                            except:
                                pred_corrs[model].append(0)
                    
                    # Plot observed
                    time_subset = years[window:window+len(obs_corr)]
                    ax2.plot(time_subset, obs_corr, 'k-', linewidth=3, label='Observed', alpha=0.8)
                    
                    # Plot top models
                    for i, (model, corrs) in enumerate(pred_corrs.items()):
                        if corrs:
                            color = self.quantum_framework.model_colors.get(model, f'C{i}')
                            ax2.plot(time_subset[:len(corrs)], corrs, 
                                    label=model.replace('_', ' ').title(),
                                    color=color, alpha=0.7, linewidth=2)
                
                ax2.set_xlabel('Year', fontsize=10)
                ax2.set_ylabel('Cross-Country Correlation', fontsize=10)
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
            except Exception as e:
                ax2.text(0.5, 0.5, f'Entanglement analysis\nerror: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            
            # 3. Energy Spectrum
            ax3 = fig1.add_subplot(gs1[0, 2])
            ax3.set_title('Hamiltonian Energy Spectrum', fontweight='bold', fontsize=12)
            
            try:
                eigenvals, eigenvecs = np.linalg.eigh(H)
                colors = plt.cm.viridis(np.linspace(0, 1, len(eigenvals)))
                
                for i, (energy, color) in enumerate(zip(eigenvals, colors)):
                    ax3.axhline(y=energy, color=color, linewidth=4, alpha=0.8)
                    ax3.text(0.1, energy, f'E{i}', fontsize=9, verticalalignment='center')
                
                gaps = np.diff(np.sort(eigenvals))
                ax3.text(0.6, 0.95, f'Energy Gaps:\nMin: {np.min(gaps):.3f}\nMax: {np.max(gaps):.3f}', 
                        transform=ax3.transAxes, verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
                
                ax3.set_xlim(0, 1)
                ax3.set_ylabel('Energy', fontsize=10)
                ax3.set_xlabel('(Energy Levels)', fontsize=10)
                ax3.grid(True, alpha=0.3)
            except Exception as e:
                ax3.text(0.5, 0.5, f'Energy spectrum\nerror: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=10)
            
            # 4. Quantum vs Classical FOR BOTH COUNTRIES (TWO SUBPLOTS)
            for country_idx, country in enumerate(countries):
                ax4 = fig1.add_subplot(gs1[1, country_idx])
                ax4.set_title(f'Quantum vs Classical - {country}', fontweight='bold', fontsize=12)
                
                try:
                    observed = observed_data[country]
                    ax4.plot(years, observed, 'k-', linewidth=3, alpha=0.9, label='Observed', zorder=10)
                    
                    # Quantum models
                    quantum_models = ['von_neumann', 'lindblad', 'floquet_von_neumann', 'floquet_lindblad']
                    quantum_preds = []
                    for model in quantum_models:
                        if model in all_predictions and country in all_predictions[model]:
                            quantum_preds.append(all_predictions[model][country])
                    
                    if quantum_preds:
                        quantum_avg = np.mean(quantum_preds, axis=0)
                        quantum_std = np.std(quantum_preds, axis=0)
                        ax4.plot(years, quantum_avg, 'b-', linewidth=2, label='Quantum Avg', alpha=0.8)
                        ax4.fill_between(years, quantum_avg - quantum_std, quantum_avg + quantum_std, 
                                       alpha=0.25, color='blue')
                    
                    # Classical models
                    classical_models = ['schrodinger', 'schrodinger_nonlinear']
                    classical_preds = []
                    for model in classical_models:
                        if model in all_predictions and country in all_predictions[model]:
                            classical_preds.append(all_predictions[model][country])
                    
                    if classical_preds:
                        classical_avg = np.mean(classical_preds, axis=0)
                        classical_std = np.std(classical_preds, axis=0)
                        ax4.plot(years, classical_avg, 'r-', linewidth=2, label='Classical Avg', alpha=0.8)
                        ax4.fill_between(years, classical_avg - classical_std, classical_avg + classical_std, 
                                       alpha=0.25, color='red')
                    
                    ax4.set_xlabel('Year', fontsize=10)
                    ax4.set_ylabel('Policy Probability', fontsize=10)
                    ax4.legend(fontsize=9)
                    ax4.grid(True, alpha=0.3)
                except Exception as e:
                    ax4.text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                            ha='center', va='center', transform=ax4.transAxes, fontsize=10)
            
            # 5. COUNTRY COMPARISON (spans remaining space)
            ax5 = fig1.add_subplot(gs1[1, 2])
            ax5.set_title('Country Comparison', fontweight='bold', fontsize=12)
            
            try:
                # Plot both countries observed data
                for i, country in enumerate(countries):
                    observed = observed_data[country]
                    color = f'C{i}'
                    ax5.plot(years, observed, color=color, linewidth=3, alpha=0.8, 
                            label=f'{country} (Observed)', linestyle='-')
                    
                    # Add best model prediction for this country
                    if all_predictions:
                        # Find best model for this country
                        best_mse = float('inf')
                        best_pred = None
                        for model_name, predictions in all_predictions.items():
                            pred = predictions[country]
                            mse = np.mean((observed - pred)**2)
                            if mse < best_mse:
                                best_mse = mse
                                best_pred = pred
                        
                        if best_pred is not None:
                            ax5.plot(years, best_pred, color=color, linewidth=2, alpha=0.6,
                                    label=f'{country} (Best Model)', linestyle='--')
                
                ax5.set_xlabel('Year', fontsize=10)
                ax5.set_ylabel('Policy Probability', fontsize=10)
                ax5.legend(fontsize=9)
                ax5.grid(True, alpha=0.3)
            except Exception as e:
                ax5.text(0.5, 0.5, f'Country comparison\nerror: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax5.transAxes, fontsize=10)
            
            # 6-9. Additional quantum analysis plots
            remaining_titles = [
                'Phase Space Dynamics',
                'Decoherence Analysis', 
                'Quantum Fidelity'
            ]
            
            positions = [(2, 0), (2, 1), (2, 2)]
            
            for i, (title, pos) in enumerate(zip(remaining_titles, positions)):
                ax = fig1.add_subplot(gs1[pos[0], pos[1]])
                ax.set_title(title, fontweight='bold', fontsize=12)
                
                if i == 0 and len(countries) >= 2:  # Phase space with both countries
                    try:
                        for j, (model_name, predictions) in enumerate(list(all_predictions.items())[:4]):
                            x = predictions[countries[0]]
                            y = predictions[countries[1]]
                            color = self.quantum_framework.model_colors.get(model_name, f'C{j}')
                            ax.plot(x, y, alpha=0.7, linewidth=1.5, color=color,
                                   label=model_name.replace('_', ' ').title())
                            ax.plot(x[0], y[0], 'o', color=color, markersize=6)
                            ax.plot(x[-1], y[-1], 's', color=color, markersize=6)
                        
                        ax.set_xlabel(f'{countries[0]}', fontsize=10)
                        ax.set_ylabel(f'{countries[1]}', fontsize=10)
                        ax.legend(fontsize=8)
                    except:
                        ax.text(0.5, 0.5, 'Phase Space:\nPolicy trajectories\nin 2D space', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=11)
                elif i == 1:  # Decoherence
                    ax.text(0.5, 0.5, 'Decoherence:\nQuantum coherence\ndecay analysis', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=11)
                else:  # Fidelity
                    ax.text(0.5, 0.5, 'Quantum Fidelity:\nState preservation\nmeasures', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=11)
                
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('🔬 Quantum Trade Policy Analysis: Both Countries', 
                         fontsize=18, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # SAVE
            plot_path1 = f'{self.output_dir}/plots/detailed/quantum_expert_analysis.png'
            fig1.savefig(plot_path1, dpi=200, bbox_inches='tight', facecolor='white', 
                        edgecolor='none', pad_inches=0.2)
            plt.close(fig1)
            print(f"  ✅ Quantum expert plots saved: {plot_path1}")
            
        except Exception as e:
            print(f"  ❌ Error creating quantum expert plots: {e}")
            try:
                plt.close(fig1)
            except:
                pass
        
        # ========================================================================
        # PLOT 2: TIME SERIES EVALUATION (BOTH COUNTRIES)
        # ========================================================================
        try:
            print("  📈 Creating time series evaluation for both countries...")
            
            fig2 = plt.figure(figsize=(24, 20))
            gs2 = GridSpec(4, 3, figure=fig2, hspace=0.4, wspace=0.35)
            
            # Get top performing models for each country
            country_best_models = {}
            for country in countries:
                observed = observed_data[country]
                model_scores = {}
                for model_name, predictions in all_predictions.items():
                    pred = predictions[country]
                    mse = np.mean((observed - pred)**2)
                    model_scores[model_name] = mse
                
                country_best_models[country] = sorted(model_scores.items(), key=lambda x: x[1])[:4]
            
            # 1. PREDICTIONS FOR BOTH COUNTRIES (TOP ROW)
            for country_idx, country in enumerate(countries):
                ax1 = fig2.add_subplot(gs2[0, country_idx])
                ax1.set_title(f'🎯 Top Models - {country}', fontweight='bold', fontsize=14)
                
                observed = observed_data[country]
                sorted_models = country_best_models[country]
                
                # Plot observed
                ax1.plot(years, observed, 'k-', linewidth=4, alpha=0.9, label='Observed', zorder=10)
                
                # Plot top 3 models for this country
                for i, (model_name, mse) in enumerate(sorted_models[:3]):
                    pred = all_predictions[model_name][country]
                    color = self.quantum_framework.model_colors.get(model_name, f'C{i}')
                    
                    ax1.plot(years, pred, color=color, linewidth=2, alpha=0.8,
                           label=f'{model_name.replace("_", " ").title()}\n(MSE: {mse:.4f})')
                    
                    # Confidence bands
                    residuals = observed - pred
                    std_error = np.std(residuals)
                    ax1.fill_between(years, pred - 1.96*std_error, pred + 1.96*std_error,
                                   alpha=0.15, color=color)
                
                ax1.set_xlabel('Year', fontsize=11)
                ax1.set_ylabel('Policy Probability', fontsize=11)
                ax1.legend(fontsize=9, loc='best')
                ax1.grid(True, alpha=0.3)
            
            # 2. COUNTRY COMPARISON (TOP RIGHT)
            ax_comp = fig2.add_subplot(gs2[0, 2])
            ax_comp.set_title('🌍 Country Comparison', fontweight='bold', fontsize=14)
            
            for i, country in enumerate(countries):
                observed = observed_data[country]
                color = f'C{i}'
                ax_comp.plot(years, observed, color=color, linewidth=3, alpha=0.8, 
                            label=f'{country} (Observed)')
                
                # Best model for each country
                best_model, best_mse = country_best_models[country][0]
                best_pred = all_predictions[best_model][country]
                ax_comp.plot(years, best_pred, color=color, linewidth=2, alpha=0.6,
                            label=f'{country} (Best: {best_model.replace("_", " ").title()})', 
                            linestyle='--')
            
            ax_comp.set_xlabel('Year', fontsize=11)
            ax_comp.set_ylabel('Policy Probability', fontsize=11)
            ax_comp.legend(fontsize=10)
            ax_comp.grid(True, alpha=0.3)
            
            # 3. RESIDUAL ANALYSIS FOR BOTH COUNTRIES
            for country_idx, country in enumerate(countries):
                ax3 = fig2.add_subplot(gs2[1, country_idx])
                ax3.set_title(f'📊 Residuals - {country}', fontweight='bold', fontsize=12)
                
                observed = observed_data[country]
                sorted_models = country_best_models[country]
                
                residuals_data = []
                model_labels = []
                
                for model_name, _ in sorted_models[:3]:
                    pred = all_predictions[model_name][country]
                    residuals = observed - pred
                    residuals_data.append(residuals)
                    model_labels.append(model_name.replace('_', '\n'))
                
                bp = ax3.boxplot(residuals_data, labels=model_labels, patch_artist=True)
                
                for i, box in enumerate(bp['boxes']):
                    model_name = model_labels[i].replace('\n', '_')
                    color = self.quantum_framework.model_colors.get(model_name, f'C{i}')
                    box.set_facecolor(color)
                    box.set_alpha(0.7)
                
                ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax3.set_ylabel('Prediction Error', fontsize=10)
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax3.grid(True, alpha=0.3)
            
            # 4. PERFORMANCE COMPARISON
            ax4 = fig2.add_subplot(gs2[1, 2])
            ax4.set_title('🏆 Performance Comparison', fontweight='bold', fontsize=12)
            
            if all_metrics:
                # Show composite scores for top models of each country
                all_top_models = set()
                for country in countries:
                    for model_name, _ in country_best_models[country][:2]:
                        all_top_models.add(model_name)
                
                composite_scores = []
                perf_model_names = []
                
                for model_name in all_top_models:
                    if model_name in all_metrics and 'overall' in all_metrics[model_name]:
                        overall = all_metrics[model_name]['overall']
                        composite_scores.append(overall.get('composite_score', 0))
                        perf_model_names.append(model_name.replace('_', '\n'))
                
                if composite_scores:
                    bars = ax4.bar(range(len(perf_model_names)), composite_scores, alpha=0.8)
                    
                    for i, bar in enumerate(bars):
                        model_name = perf_model_names[i].replace('\n', '_')
                        color = self.quantum_framework.model_colors.get(model_name, f'C{i}')
                        bar.set_color(color)
                        # Add value labels
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                    
                    ax4.set_xticks(range(len(perf_model_names)))
                    ax4.set_xticklabels(perf_model_names, rotation=45, ha='right', fontsize=9)
                    ax4.set_ylabel('Composite Score', fontsize=10)
                    ax4.grid(True, alpha=0.3)
            
            # 5. ROLLING PERFORMANCE FOR BOTH COUNTRIES
            for country_idx, country in enumerate(countries):
                ax5 = fig2.add_subplot(gs2[2, country_idx])
                ax5.set_title(f'📈 Rolling Performance - {country}', fontweight='bold', fontsize=12)
                
                observed = observed_data[country]
                sorted_models = country_best_models[country]
                window = 24
                
                for model_name, _ in sorted_models[:3]:
                    pred = all_predictions[model_name][country]
                    rolling_mse = []
                    
                    for i in range(window, len(observed)):
                        obs_window = observed[i-window:i]
                        pred_window = pred[i-window:i]
                        mse = np.mean((obs_window - pred_window)**2)
                        rolling_mse.append(mse)
                    
                    color = self.quantum_framework.model_colors.get(model_name, 'blue')
                    ax5.plot(years[window:], rolling_mse, 
                           label=model_name.replace('_', ' ').title(),
                           color=color, linewidth=2, alpha=0.8)
                
                ax5.set_xlabel('Year', fontsize=10)
                ax5.set_ylabel('Rolling MSE', fontsize=10)
                ax5.legend(fontsize=8)
                ax5.grid(True, alpha=0.3)
            
            # 6. CROSS-COUNTRY CORRELATION
            ax6 = fig2.add_subplot(gs2[2, 2])
            ax6.set_title('🔗 Cross-Country Analysis', fontweight='bold', fontsize=12)
            
            if len(countries) >= 2:
                # Observed vs predicted correlations
                obs_corr = np.corrcoef([observed_data[c] for c in countries])[0,1]
                
                model_corrs = []
                model_names = []
                
                for model_name in all_predictions.keys():
                    pred_corr = np.corrcoef([all_predictions[model_name][c] for c in countries])[0,1]
                    model_corrs.append(pred_corr if not np.isnan(pred_corr) else 0)
                    model_names.append(model_name.replace('_', '\n'))
                
                # Bar plot of correlations
                bars = ax6.bar(range(len(model_names)), model_corrs, alpha=0.7)
                ax6.axhline(y=obs_corr, color='red', linestyle='--', linewidth=3, 
                           label=f'Observed: {obs_corr:.3f}')
                
                for i, bar in enumerate(bars):
                    model_name = model_names[i].replace('\n', '_')
                    color = self.quantum_framework.model_colors.get(model_name, f'C{i}')
                    bar.set_color(color)
                
                ax6.set_xticks(range(len(model_names)))
                ax6.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
                ax6.set_ylabel('Cross-Country Correlation', fontsize=10)
                ax6.legend(fontsize=9)
                ax6.grid(True, alpha=0.3)
            
            # 7. OVERALL MODEL RANKING (BOTTOM ROW)
            ax7 = fig2.add_subplot(gs2[3, :])
            ax7.set_title('🥇 Overall Model Ranking Across Both Countries', fontweight='bold', fontsize=14)
            
            # Calculate average performance across countries
            overall_performance = {}
            for model_name in all_predictions.keys():
                total_mse = 0
                for country in countries:
                    observed = observed_data[country]
                    pred = all_predictions[model_name][country]
                    mse = np.mean((observed - pred)**2)
                    total_mse += mse
                overall_performance[model_name] = total_mse / len(countries)
            
            sorted_overall = sorted(overall_performance.items(), key=lambda x: x[1])
            
            model_names = [name for name, _ in sorted_overall]
            avg_mses = [mse for _, mse in sorted_overall]
            
            bars = ax7.bar(range(len(model_names)), avg_mses, alpha=0.8)
            
            for i, (bar, model_name) in enumerate(zip(bars, model_names)):
                color = self.quantum_framework.model_colors.get(model_name, f'C{i}')
                bar.set_color(color)
                
                # Add rank numbers
                ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(avg_mses)*0.01,
                       f'#{i+1}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax7.set_xticks(range(len(model_names)))
            ax7.set_xticklabels([name.replace('_', '\n') for name in model_names], 
                               rotation=45, ha='right', fontsize=10)
            ax7.set_ylabel('Average MSE (Both Countries)', fontsize=12)
            ax7.grid(True, alpha=0.3)
            
            plt.suptitle('📊 Time Series Evaluation: Both Countries Analysis', 
                         fontsize=18, fontweight='bold', y=0.98)
            plt.tight_layout()
            
            # SAVE
            plot_path2 = f'{self.output_dir}/plots/essential/time_series_evaluation.png'
            fig2.savefig(plot_path2, dpi=200, bbox_inches='tight', facecolor='white',
                        edgecolor='none', pad_inches=0.2)
            plt.close(fig2)
            print(f"  ✅ Time series evaluation plots saved: {plot_path2}")
            
        except Exception as e:
            print(f"  ❌ Error creating time series evaluation plots: {e}")
            try:
                plt.close(fig2)
            except:
                pass
        
        # Clean up
        plt.clf()
        plt.cla()
        
        print("🎉 Enhanced plotting complete for BOTH countries!")
        print(f"📁 Quantum expert plots: {self.output_dir}/plots/detailed/quantum_expert_analysis.png")
        print(f"📊 Time series plots: {self.output_dir}/plots/essential/time_series_evaluation.png")
        print(f"🌍 Countries analyzed: {', '.join(countries)}")
    
    def _print_final_summary(self, all_metrics):
        """Print final analysis summary."""
        # Best performing model
        ranked_models = []
        for model_name, metrics in all_metrics.items():
            if model_name.startswith('_'):
                continue
            if 'overall' in metrics and 'composite_score' in metrics['overall']:
                score = metrics['overall']['composite_score']
                ranked_models.append((model_name, score))
        
        if ranked_models:
            ranked_models.sort(key=lambda x: x[1], reverse=True)
            best_model, best_score = ranked_models[0]
            print(f"🏆 Best Model: {best_model.replace('_', ' ').title()}")
            print(f"   Composite Score: {best_score:.3f}")
            
            # Quantum vs classical advantage
            quantum_models = [m for m, s in ranked_models if any(qm in m for qm in ['von_neumann', 'lindblad', 'floquet'])]
            classical_models = [m for m, s in ranked_models if 'schrodinger' in m and 'floquet' not in m]
            
            if quantum_models and classical_models:
                quantum_avg = np.mean([s for m, s in ranked_models if m in quantum_models])
                classical_avg = np.mean([s for m, s in ranked_models if m in classical_models])
                
                if quantum_avg > classical_avg:
                    print(f"✅ Quantum models outperform classical (Δ={quantum_avg-classical_avg:.3f})")
                else:
                    print(f"⚠️ Classical models competitive (Δ={classical_avg-quantum_avg:.3f})")

        # Model differentiation
        if '_differentiation_analysis' in all_metrics:
            diff_analysis = all_metrics['_differentiation_analysis']
            avg_diff = diff_analysis['average_differentiation']
            floquet_diff = diff_analysis['floquet_vs_standard_diff']
            
            print(f"\n🔄 Model Differentiation:")
            print(f"   Average difference: {avg_diff:.3f}")
            print(f"   Floquet advantage: {floquet_diff:.3f}")
            
            if avg_diff > 0.05:
                print("   ✅ Models are well-differentiated")
            else:
                print("   ⚠️ Consider increasing parameter differences")

        # Regime detection performance
        regime_scores = []
        for model_name, metrics in all_metrics.items():
            if not model_name.startswith('_') and 'overall' in metrics:
                f1 = metrics['overall'].get('avg_regime_f1', 0)
                regime_scores.append(f1)
        
        if regime_scores:
            avg_regime_f1 = np.mean(regime_scores)
            print(f"\n🎯 Regime Detection:")
            print(f"   Average F1 Score: {avg_regime_f1:.3f}")
            
            if avg_regime_f1 > 0.3:
                print("   ✅ Good regime change detection")
            elif avg_regime_f1 > 0.1:
                print("   ⚠️ Moderate regime detection - consider parameter tuning")
            else:
                print("   ❌ Poor regime detection - adjust synthetic data config")


# ===== CONFIGURATION HELPER FUNCTIONS =====

def create_custom_synthetic_config(
    coherence_country1: float = 2.5,
    coherence_country2: float = 1.8,
    entanglement_strength: float = 0.4,
    regime_times: List[float] = None,
    regime_amplitudes: List[float] = None,
    external_drive_frequency: float = 0.5,
    external_drive_amplitude: float = 0.15,
    quantum_noise: float = 0.08
) -> QuantumSyntheticConfig:
    """Create a custom synthetic data configuration."""
    if regime_times is None:
        regime_times = [3.0, 7.0]  # Default regime change times
    if regime_amplitudes is None:
        regime_amplitudes = [-0.25, 0.20]  # Default regime change amplitudes
    
    return QuantumSyntheticConfig(
        coherence_time_country1=coherence_country1,
        coherence_time_country2=coherence_country2,
        entanglement_strength=entanglement_strength,
        regime_change_times=regime_times,
        regime_change_amplitudes=regime_amplitudes,
        external_drive_frequency=external_drive_frequency,
        external_drive_amplitude=external_drive_amplitude,
        quantum_noise_level=quantum_noise,
        
        # Additional parameters with sensible defaults
        primary_frequency_1=0.8,
        primary_frequency_2=1.2,
        secondary_frequency_1=2.1,
        secondary_frequency_2=1.7,
        decoherence_rate_1=0.15,
        decoherence_rate_2=0.25,
        nonlinear_strength_1=0.3,
        nonlinear_strength_2=0.2
    )


def create_custom_quantum_config(
    coupling_strength: float = 0.4,
    floquet_frequency: float = 1.5,
    floquet_amplitude: float = 0.3,
    temperature: float = 0.1,
    dephasing_rate: float = 0.08,
    country_params: Dict[str, Dict[str, float]] = None
) -> QuantumModelConfig:
    """Create a custom quantum model configuration."""
    if country_params is None:
        # Default differentiated parameters
        country_params = {
            'Egypt': {
                'E0': 2.5,
                'Delta': 1.8,
                'delta': 0.8,
                'phi': 0.0,
                'gamma': 0.05,  # Low damping - coherent
                'coherence_time': 3.0
            },
            'Russian Federation': {
                'E0': 2.0,
                'Delta': 1.2,
                'delta': -0.6,
                'phi': np.pi/4,
                'gamma': 0.20,  # High damping - decoherent
                'coherence_time': 1.5
            }
        }
    
    return QuantumModelConfig(
        country_params=country_params,
        coupling_strength=coupling_strength,
        temperature=temperature,
        floquet_frequency=floquet_frequency,
        floquet_amplitude=floquet_amplitude,
        dephasing_rate=dephasing_rate,
        relaxation_rate=0.05
    )


# ===== QUICK START EXAMPLES =====

def run_basic_example():
    """Run a basic analysis with default settings."""
    print("🚀 BASIC QUANTUM TRADE ANALYSIS")
    print("=" * 40)
    
    try:
        analyzer = EnhancedQuantumTradeAnalysis(output_dir="basic_analysis")
        results = analyzer.run_enhanced_analysis()
        
        if results.get('status') != 'failed':
            print("✅ Basic analysis complete!")
            print(f"📁 Results: ./basic_analysis/")
            return results
        else:
            print(f"❌ Analysis failed: {results.get('error')}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def run_custom_example():
    """Run analysis with custom configuration."""
    print("🚀 CUSTOM QUANTUM TRADE ANALYSIS")
    print("=" * 40)
    
    try:
        # Create custom configurations
        custom_synthetic = create_custom_synthetic_config(
            coherence_country1=3.5,
            entanglement_strength=0.6,
            regime_times=[2.0, 5.0, 8.0],
            quantum_noise=0.05
        )
        
        custom_quantum = create_custom_quantum_config(
            coupling_strength=0.6,
            floquet_frequency=2.0,
            floquet_amplitude=0.4
        )
        
        analyzer = EnhancedQuantumTradeAnalysis(
            output_dir="custom_analysis",
            synthetic_config=custom_synthetic,
            quantum_config=custom_quantum
        )
        
        results = analyzer.run_enhanced_analysis()
        
        if results.get('status') != 'failed':
            print("✅ Custom analysis complete!")
            print(f"📁 Results: ./custom_analysis/")
            return results
        else:
            print(f"❌ Analysis failed: {results.get('error')}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None


def run_minimal_robust_example():
    """Run a minimal example that should work even with matplotlib issues."""
    print("🚀 MINIMAL ROBUST EXAMPLE")
    print("=" * 40)
    
    try:
        # Create minimal configuration
        minimal_config = create_custom_synthetic_config(
            coherence_country1=2.0,
            coherence_country2=1.5,
            entanglement_strength=0.3,
            quantum_noise=0.08
        )
        
        # Disable plotting to avoid matplotlib issues
        analyzer = EnhancedQuantumTradeAnalysis(
            output_dir="minimal_example",
            synthetic_config=minimal_config,
            enable_plotting=False
        )
        
        # Run basic analysis
        results = analyzer.run_enhanced_analysis()
        
        if results.get('status') != 'failed':
            print("\n✅ MINIMAL EXAMPLE SUCCESSFUL!")
            print("📄 Check ./minimal_example/ for data outputs")
            return results
        else:
            print(f"❌ Minimal example failed: {results.get('error')}")
            return None
            
    except Exception as e:
        print(f"❌ Critical error in minimal example: {e}")
        return None


# ===== MAIN FUNCTION =====

def main():
    """Main function demonstrating the enhanced framework capabilities."""
    print("🚀 ENHANCED QUANTUM TRADE POLICY ANALYSIS FRAMEWORK")
    print("=" * 60)
    print("🎯 Features:")
    print("  • Configurable synthetic data generation")
    print("  • Quantum-appropriate synthetic patterns") 
    print("  • Enhanced model differentiation")
    print("  • Comprehensive evaluation metrics")
    print("  • Robust error handling")
    print("")
    
    try:
        # Example 1: Basic analysis
        print("📊 EXAMPLE 1: Basic Analysis")
        print("-" * 30)
        results1 = run_basic_example()
        
        # Example 2: Custom configuration
        print("\n📊 EXAMPLE 2: Custom Configuration")
        print("-" * 30)
        results2 = run_custom_example()
        
        print("\n🎉 FRAMEWORK DEMONSTRATION COMPLETE!")
        print("\n🔧 TO CUSTOMIZE:")
        print("1. Use create_custom_synthetic_config() for data generation")
        print("2. Use create_custom_quantum_config() for model parameters")
        print("3. Pass configs to EnhancedQuantumTradeAnalysis()")
        
        return {'basic': results1, 'custom': results2}
        
    except Exception as e:
        print(f"\n❌ Main examples failed: {e}")
        print("Trying minimal example...")
        return run_minimal_robust_example()


if __name__ == "__main__":
    """
    Run the enhanced framework with examples.
    """
    print("Starting Enhanced Quantum Trade Analysis Framework...")
    print("Clean version with fixed indentation and syntax!\n")
    
    # Try main examples first
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Main examples failed: {e}")
        print("Trying minimal example...")
        results = run_minimal_robust_example()
    
    print("\n🎉 FRAMEWORK DEMONSTRATION COMPLETE!")
    print("\nIf you encountered any issues:")
    print("1. This version has fixed indentation and syntax errors")
    print("2. Matplotlib issues are handled gracefully") 
    print("3. Text-based alternatives are created automatically")
    
    print("\nFor successful runs, see the generated output directories.")
    print("CSV files contain all numerical results even when plotting fails.")
    print("This is the complete, clean framework ready to use!")