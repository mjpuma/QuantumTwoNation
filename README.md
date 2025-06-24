# Quantum Trade Policy Analysis Framework

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Quantum](https://img.shields.io/badge/quantum-physics-purple)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-validated-brightgreen)

A comprehensive framework for analyzing international trade policy dynamics using quantum mechanical models. This project implements 8 distinct quantum evolution models and provides rigorous validation against analytical solutions from quantum physics literature.

## 📖 Academic Documentation

For comprehensive mathematical foundations, complete derivations, and literature review, see:
- **[Quantum Framework Validation Documentation](docs/Quantum_Framework_Validation.pdf)** - Complete academic paper with:
  - Mathematical proofs of all validation tests
  - Literature review from foundational quantum mechanics papers
  - Economic interpretation guide for non-technical readers
  - Detailed error analysis and precision methodology

This PDF provides the rigorous academic backing for the practical framework described in this README.

## 🌟 Key Features

- **8 Quantum Models**: Schrödinger, von Neumann, Lindblad, Floquet variants with enhanced differentiation
- **Rigorous Validation**: Test suite using exact analytical solutions (Rabi oscillations, amplitude damping, stroboscopic periodicity)
- **Synthetic Data Generation**: Quantum-appropriate data with configurable parameters
- **Comprehensive Evaluation**: Probabilistic metrics, regime detection, model differentiation analysis
- **OECD Integration Ready**: Framework designed for easy adaptation to real trade policy data
- **Publication-Quality Plots**: Automated generation of analysis and validation figures

## 📚 Table of Contents

- [Installation](#installation)
- [Validation Suite](#validation-suite)
- [8-Model Framework](#8-model-framework)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [OECD Data Integration](#oecd-data-integration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🔧 Installation

### Requirements

```bash
# Core dependencies
pip install numpy scipy pandas matplotlib seaborn
pip install dataclasses typing-extensions

# Optional for enhanced plotting
pip install PyQt5  # For interactive plots
```

### Setup

```bash
git clone https://github.com/yourusername/quantum-trade-policy
cd quantum-trade-policy
```

**No additional setup required** - the framework is self-contained with robust error handling.

## ✅ Validation Suite

### Overview

The validation suite tests all quantum models against **exact analytical solutions** from quantum physics literature. This is not traditional unit testing, but validation against known physics benchmarks.

### Validation Tests

| Test | Physics Benchmark | Validates |
|------|------------------|-----------|
| **Schrödinger** | Rabi Oscillations: P₁(t) = sin²(Ωt/2) | Unitary evolution, energy conservation |
| **Lindblad** | Amplitude Damping: ρ₁₁(t) = ρ₁₁(0)e^(-γt) | Open systems, trace preservation |
| **Floquet** | Stroboscopic Periodicity: U(2T) = [U(T)]² | Time-periodic Hamiltonians |
| **Consistency** | von Neumann ↔ Schrödinger | Pure state limits |
| **Consistency** | Floquet ↔ Static | Zero drive limits |

### Running Validation

#### Quick Validation Check
```python
# Verify validation suite works with direct calculations
python validation_integration_guide.py
# Choose option 2: "Quick check with direct calculations"
```

#### Full Framework Validation
```python
# Test your quantum framework against analytical solutions
python validation_integration_guide.py
# Choose option 1: "Validate your quantum framework"
```

#### Create Validation Figures
```python
python validation_figures.py
```

**Expected Output**: All tests should **PASS** with errors < 1e-6

```
🎉 SUCCESS: All quantum models validated!
✅ Your framework correctly implements quantum physics
✅ Ready to proceed with trade policy analysis

Validation Results Summary:
  ✅ PASS: Schrödinger Rabi Oscillations
  ✅ PASS: Lindblad Amplitude Damping  
  ✅ PASS: Floquet Stroboscopic Periodicity
  ✅ PASS: von Neumann → Schrödinger Consistency
  ✅ PASS: Floquet → Static Consistency
```

### Validation Methodology

- **No Approximations**: Uses exact analytical solutions, not numerical approximations
- **Physics-Based**: Tests fundamental quantum mechanical properties
- **Rigorous Tolerances**: 1e-6 precision requirement
- **Cross-Model Consistency**: Ensures internal framework coherence
- **Publication Ready**: Generates detailed validation figures

## 🔬 8-Model Framework

### Model Overview

The framework implements 8 quantum evolution models with enhanced differentiation:

#### Standard Models (1-4)
1. **Schrödinger**: Linear unitary evolution
2. **Nonlinear Schrödinger**: Self-interaction effects  
3. **von Neumann**: Open system density matrix evolution
4. **Lindblad**: Markovian decoherence with jump operators

#### Floquet Models (5-8)
5. **Floquet Schrödinger**: Periodically driven unitary systems
6. **Floquet Nonlinear Schrödinger**: Driven nonlinear dynamics
7. **Floquet von Neumann**: Driven open systems
8. **Floquet Lindblad**: Driven Markovian decoherence

### Key Features

- **Enhanced Differentiation**: Models produce distinguishable outputs
- **Configurable Parameters**: Easy customization of quantum and synthetic parameters
- **Robust Error Handling**: Graceful fallbacks and detailed error reporting
- **Multiple Output Formats**: CSV data, JSON metrics, publication plots

## 🚀 Quick Start

### Basic Example

```python
from streamlined_quantum_trade import EnhancedQuantumTradeAnalysis

# Run basic analysis with default settings
analyzer = EnhancedQuantumTradeAnalysis(output_dir="basic_analysis")
results = analyzer.run_enhanced_analysis()

# Check results
if results.get('status') != 'failed':
    print("✅ Analysis complete! Check ./basic_analysis/")
else:
    print(f"❌ Analysis failed: {results.get('error')}")
```

### Custom Configuration Example

```python
from streamlined_quantum_trade import (
    EnhancedQuantumTradeAnalysis,
    create_custom_synthetic_config,
    create_custom_quantum_config
)

# Create custom configurations
custom_synthetic = create_custom_synthetic_config(
    coherence_country1=3.5,        # Longer coherence time
    entanglement_strength=0.6,     # Stronger country coupling
    regime_times=[2.0, 5.0, 8.0], # Three regime changes
    quantum_noise=0.05             # Lower noise level
)

custom_quantum = create_custom_quantum_config(
    coupling_strength=0.6,    # Stronger inter-country coupling
    floquet_frequency=2.0,    # Higher drive frequency
    floquet_amplitude=0.4     # Stronger external driving
)

# Run analysis with custom settings
analyzer = EnhancedQuantumTradeAnalysis(
    output_dir="custom_analysis",
    synthetic_config=custom_synthetic,
    quantum_config=custom_quantum
)

results = analyzer.run_enhanced_analysis()
```

### Minimal Robust Example

```python
from streamlined_quantum_trade import run_minimal_robust_example

# Guaranteed to work even with matplotlib issues
results = run_minimal_robust_example()
```

## ⚙️ Configuration

### Synthetic Data Configuration

Control quantum-appropriate synthetic data generation:

```python
from streamlined_quantum_trade import create_custom_synthetic_config

config = create_custom_synthetic_config(
    # Time parameters
    start_year=2008,
    end_year=2018,
    
    # Quantum coherence (how long quantum effects persist)
    coherence_country1=2.5,    # years
    coherence_country2=1.8,    # different for each country
    
    # Entanglement (cross-country correlations)
    entanglement_strength=0.4,  # 0-1 scale
    phase_difference=np.pi/3,   # phase relationship
    
    # Regime changes (policy shifts)
    regime_times=[3.0, 7.0],           # years when changes occur
    regime_amplitudes=[-0.25, 0.20],   # magnitude of changes
    
    # External driving (periodic influences)
    external_drive_frequency=0.5,      # 1/years
    external_drive_amplitude=0.15,     # strength
    
    # Noise levels
    quantum_noise=0.08,        # structured quantum noise
    classical_noise=0.03       # random noise
)
```

### Quantum Model Configuration

Control the 8 quantum models:

```python
from streamlined_quantum_trade import create_custom_quantum_config

config = create_custom_quantum_config(
    # System coupling
    coupling_strength=0.4,     # inter-country coupling
    temperature=0.1,           # effective temperature
    
    # Floquet driving (Models 5-8)
    floquet_frequency=1.5,     # drive frequency
    floquet_amplitude=0.3,     # drive strength
    floquet_phase=0.0,         # drive phase
    
    # Decoherence parameters
    dephasing_rate=0.08,       # pure dephasing
    relaxation_rate=0.05,      # energy relaxation
    
    # Country-specific parameters
    country_params={
        'Egypt': {
            'E0': 2.5,             # energy scale
            'Delta': 1.8,          # tunneling strength
            'delta': 0.8,          # detuning
            'gamma': 0.05,         # damping (low = coherent)
            'coherence_time': 3.0  # coherence time
        },
        'Russian Federation': {
            'E0': 2.0,
            'Delta': 1.2,
            'delta': -0.6,
            'gamma': 0.20,         # damping (high = decoherent)
            'coherence_time': 1.5
        }
    }
)
```

## 🌍 OECD Data Integration

### Current Setup (Synthetic Data)

The framework currently uses quantum-appropriate synthetic data that exhibits:
- Quantum coherence and decoherence
- Cross-country entanglement
- Regime changes (policy transitions)
- Periodic external driving
- Structured noise patterns

### Adapting for Real OECD Data

To integrate real OECD trade policy data:

#### 1. Data Preparation

```python
def load_oecd_data(countries, start_year, end_year):
    """
    Replace synthetic data generation with OECD data loading.
    
    Expected format:
    - DataFrame with 'Date', 'Year' columns
    - Country columns with policy restriction indices (0-1 scale)
    - Monthly or quarterly frequency recommended
    """
    # Your OECD data loading code here
    # Example structure:
    df = pd.read_csv('oecd_trade_policy_data.csv')
    
    # Ensure data is in [0,1] range
    for country in countries:
        df[country] = normalize_to_01_range(df[country])
    
    return df, {country: df[country].values for country in countries}
```

#### 2. Replace Synthetic Generator

In `streamlined_quantum_trade.py`, modify the `load_data` method:

```python
def load_data(self, countries: List[str]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Load real OECD data instead of synthetic."""
    print("🔄 Loading OECD trade policy data...")
    
    # Replace this line:
    # return self.synthetic_generator.generate_quantum_synthetic_data(countries)
    
    # With:
    return load_oecd_data(countries, 
                         self.synthetic_config.start_year, 
                         self.synthetic_config.end_year)
```

#### 3. Parameter Fitting

Replace hard-coded quantum parameters with fitted values:

```python
def fit_quantum_parameters(observed_data, countries):
    """
    Fit quantum model parameters to real data.
    
    This replaces the current synthetic parameter configuration
    with parameters fitted to actual OECD trade policy dynamics.
    """
    from scipy.optimize import minimize
    
    def objective(params):
        # Create quantum config from parameters
        quantum_config = create_quantum_config_from_params(params)
        
        # Run quantum models
        framework = EnhancedQuantumTradeEvolutionFramework(quantum_config)
        predictions = framework.run_all_models(H, t_points, countries)
        
        # Calculate fit quality
        mse = calculate_fit_quality(predictions, observed_data)
        return mse
    
    # Fit parameters
    fitted_params = minimize(objective, initial_guess, method='Nelder-Mead')
    return create_quantum_config_from_params(fitted_params.x)
```

#### 4. OECD-Specific Countries

Modify country parameters for your specific OECD analysis:

```python
# Example: G7 countries
countries = ['United States', 'Germany', 'Japan', 'United Kingdom', 
            'France', 'Italy', 'Canada']

# Or focus on specific regions
countries = ['China', 'European Union', 'NAFTA']
```

### OECD Data Requirements

- **Format**: CSV with Date, Year, and country columns
- **Range**: Policy restriction indices normalized to [0,1]
- **Frequency**: Monthly or quarterly (framework handles both)
- **Coverage**: Consistent time coverage across countries
- **Missing Data**: Framework includes robust handling

## 📊 Examples

### Output Structure

After running analysis, you'll find:

```
output_directory/
├── plots/
│   ├── essential/
│   │   ├── enhanced_evolution_comparison.png     # Main results
│   │   ├── performance_summary.png               # Model comparison
│   │   └── time_series_evaluation.png            # Detailed analysis
│   └── detailed/
│       ├── model_differentiation.png             # Technical analysis
│       └── quantum_expert_analysis.png           # Physics details
├── data/
│   ├── observed_data.csv                         # Input data
│   ├── [model]_predictions.csv                   # Model outputs
│   └── evaluation_metrics.json                   # Performance metrics
├── configs/
│   ├── synthetic_config.json                     # Reproducibility
│   └── quantum_config.json                       # Model parameters
└── enhanced_analysis_report.txt                  # Summary report
```

### Expected Results

#### Model Ranking Example
```
🏆 Best Model: Floquet Von Neumann
   Composite Score: 0.847

📊 Model Performance:
  1. Floquet Von Neumann: 0.847
  2. Lindblad: 0.823
  3. Floquet Lindblad: 0.801
  4. von Neumann: 0.789
  ...

🔄 Model Differentiation: 0.087 (Good)
🎯 Regime Detection: 0.654 (Good)
```

#### Validation Results
```
✅ Schrödinger Rabi Oscillations: Max error = 2.3e-08
✅ Lindblad Amplitude Damping: Max error = 1.8e-07  
✅ Floquet Stroboscopic Periodicity: Error = 4.1e-09
✅ von Neumann Consistency: Max error = 3.2e-08
✅ Floquet Consistency: Max error = 1.5e-07
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Matplotlib Backend Issues
```python
# Framework handles this automatically, but if you see warnings:
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Or disable plotting entirely:
analyzer = EnhancedQuantumTradeAnalysis(enable_plotting=False)
```

#### 2. Validation Failures
```bash
# If validation tests fail:
python validation_integration_guide.py
# Choose option 2 first to verify validation suite works
# Then choose option 1 to test your framework
```

#### 3. Memory Issues with Large Datasets
```python
# Reduce time resolution or number of models
custom_config = create_custom_synthetic_config(
    start_year=2010,  # Shorter time range
    end_year=2015
)
```

#### 4. Installation Issues
```bash
# Minimal installation
pip install numpy scipy pandas matplotlib

# If PyQt5 fails:
sudo apt-get install python3-tk  # Linux
# or use Agg backend (non-interactive)
```

### Getting Help

1. **Check validation first**: Run `python validation_integration_guide.py`
2. **Try minimal example**: Run `run_minimal_robust_example()`
3. **Check output directory**: Look for error messages in generated files
4. **Matplotlib issues**: Disable plotting with `enable_plotting=False`

## 📈 Performance Expectations

### Computational Complexity
- **Single Model**: O(N²T) where N = system size, T = time points
- **8 Models**: ~30 seconds for default parameters (11 years, monthly data)
- **Memory**: ~100MB for typical analysis

### Accuracy Benchmarks
- **Validation Tests**: All pass with errors < 1e-6
- **Model Differentiation**: RMS differences > 0.05 (good separation)
- **Physical Constraints**: Energy/trace conservation < 1e-10

## 🤝 Contributing

### Extending the Framework

#### Adding New Quantum Models
```python
def solve_new_quantum_model(self, H, t_points, model_params):
    """Add your quantum model here."""
    # Implement your quantum evolution
    # Return states or density matrices
    pass

# Add to model list in __init__
self.model_names.append('new_quantum_model')
```

#### Adding New Validation Tests
```python
def test_new_physics_benchmark(self):
    """Test against new analytical solution."""
    # Implement test using exact physics solution
    # Set self.test_results['new_test'] = result
    pass
```

### Code Style
- Follow existing patterns for error handling
- Include physics documentation for new models
- Add validation tests for new functionality
- Update configuration classes for new parameters

## 📄 License

MIT License - see LICENSE file for details.

## 🎯 Citation

If you use this framework in research, please cite:

```bibtex
@software{quantum_trade_policy,
  title={Quantum Trade Policy Analysis Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/quantum-trade-policy}
}
```

## 🔗 Related Work

- **Quantum Economics**: Application of quantum mechanics to economic systems
- **OECD Trade Policy**: International trade restriction databases
- **Floquet Theory**: Time-periodic quantum systems
- **Open Quantum Systems**: Decoherence and environmental effects

---

**🎉 Ready to analyze trade policy dynamics with quantum precision!**

For questions, issues, or contributions, please open a GitHub issue or contact the maintainer.
