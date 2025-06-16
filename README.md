# Quantum Trade Policy Analysis Framework

A comprehensive framework applying quantum mechanical evolution equations to model international trade policy dynamics between countries.

## What This Code Does

This framework models trade policy restrictions as quantum systems, where countries can exist in superposition states and exhibit quantum coupling. It compares three different quantum evolution approaches to understand how trade policies evolve and interact over time.

**Core concept**: Treat trade policy probabilities as quantum states that evolve according to quantum mechanical equations, with coupling terms representing how countries influence each other's policies.

## Key Features

- **Three quantum evolution models**: Schrödinger (pure states), von Neumann (mixed states), Lindblad (open systems)
- **Automated parameter extraction**: Converts time series fits to quantum physics parameters
- **Coupling strength optimization**: Finds optimal inter-country interaction strengths
- **Comprehensive analysis pipeline**: From data → fitting → quantum modeling → visualization
- **Modular design**: Easy to swap fitting methods or add new countries
- **Rich visualizations**: 6 detailed analysis figures covering all aspects

## Installation & Setup

### Dependencies
```bash
pip install numpy scipy matplotlib pandas
```

### Basic Usage
```python
from oecdpipelineRUEG import CompleteFixedQuantumAnalysis

# Run complete analysis
analyzer = CompleteFixedQuantumAnalysis(output_dir="results")
analyzer.run_complete_analysis()
```

## Code Structure

### Main Class: `CompleteFixedQuantumAnalysis`

**Data Generation & Fitting (Lines 57-300)**
- `create_enhanced_synthetic_data()` - Generates test data with realistic patterns
- `fit_enhanced_quantum_model()` - Fits individual country time series
- `extract_quantum_parameters()` - Maps fitting parameters to quantum variables

**Quantum Mechanics (Lines 300-500)**
- `construct_hamiltonian()` - Builds coupled quantum system
- `solve_schrodinger_pure()` - Pure state evolution
- `solve_von_neumann_unitary()` - Mixed state unitary evolution  
- `solve_lindblad_open_system()` - Open system with decoherence

**Analysis & Validation (Lines 500-700)**
- `validate_coupling_strength()` - Optimizes inter-country coupling
- `run_three_model_comparison()` - Compares all quantum approaches
- `calculate_observables_*()` - Extracts physical quantities

**Visualization (Lines 700-900)**
- `create_data_fitting_figure()` - Individual country fits and residuals
- `create_quantum_models_comparison_figure()` - Model performance comparison
- `create_coupling_analysis_figure()` - Coupling optimization results
- `create_phase_space_analysis_figure()` - Phase space trajectories
- `create_quantum_states_evolution_figure()` - Quantum state populations

## Current Implementation

### Data Source
**Synthetic data** (2008-2018) with realistic features:
- Egypt-Russia trade restriction probabilities
- Arab Spring disruption modeling
- Counter-cyclical coupling (ρ = -0.690)
- Realistic noise levels

### Fitting Method
**Enhanced multi-component quantum model:**
```
f(t) = [A₁sin(ω₁t + φ) + A₂sin(ω₂t + φ + π/3)] * exp(-γt) + C
```
- 7 parameters: amplitudes, frequencies, decoherence, phase, offset
- R² performance: 0.594-0.821 on synthetic data

### Quantum Parameter Mapping
- **Energy Gap (Δ)**: √(ω₁² + ω₂²) - fundamental quantum scale
- **Coupling Strength (E₀)**: |A₁| + |A₂| - interaction magnitude  
- **Rabi Frequency (Ω)**: |ω₂ - ω₁| - oscillation differences
- **Detuning (δ)**: 2(C - 0.5) - asymmetry from midpoint
- **Decoherence (γ)**: Direct from fitting - coherence loss rate

## Output Files

### Figures (saved to `output_dir/plots/`)
1. **data_fitting_analysis.png** - Individual country fits, phase diagrams, residuals
2. **quantum_models_comparison.png** - Three-model performance and phase space
3. **coupling_strength_analysis.png** - Coupling optimization and entanglement
4. **model_hierarchy_explanation.png** - Quantum theory overview
5. **phase_space_analysis.png** - Trajectories, velocity fields, density
6. **quantum_states_evolution.png** - |00⟩, |01⟩, |10⟩, |11⟩ populations

### Data Files (saved to `output_dir/data/`)
- **Model predictions**: Time series from all three quantum models
- **Performance metrics**: R², MAE, RMSE for each model and country
- **Quantum parameters**: Extracted Δ, E₀, Ω, δ, γ values
- **Coupling validation**: Optimization results across coupling strengths
- **Analysis summary**: Comprehensive text report

## Extending the Framework

### Adding New Fitting Methods
Replace `fit_enhanced_quantum_model()` method (lines 150-200):
```python
def your_fitting_method(self, time_points, data, country_name):
    # Your fitting approach here
    return {
        'params': fitted_parameters,
        'r2': r_squared_score,
        'prediction': fitted_curve,
        'residuals': data - fitted_curve
    }
```

### Using Real Data
Replace `create_enhanced_synthetic_data()` (lines 57-78):
```python
def load_real_data(self, file_path):
    df = pd.read_csv(file_path)
    processed_data = {
        'Egypt': df['Egypt'].values,
        'Russian Federation': df['Russian Federation'].values
    }
    return df, processed_data
```

### Adding New Countries
Extend the data dictionary and parameter extraction:
```python
processed_data = {
    'Egypt': egypt_data,
    'Russian Federation': russia_data,
    'India': india_data,  # Add new countries
    'China': china_data
}
```

### Custom Quantum Models
Add new evolution methods following the pattern:
```python
def solve_your_quantum_model(self, H, t_points, initial_state):
    # Your quantum evolution approach
    return evolved_states
```

## Performance & Validation

### Current Results
- **Egypt**: R² = 0.594, successfully captures policy oscillations
- **Russia**: R² = 0.821, reproduces counter-cyclical relationship
- **Correlation matching**: Achieves target ρ = -0.690
- **Model comparison**: von Neumann shows best overall performance

### Validation Approach
1. **Synthetic data testing** - Controlled validation with known relationships
2. **Parameter stability** - Consistent quantum variable extraction
3. **Coupling optimization** - Systematic validation across interaction strengths
4. **Cross-model comparison** - Multiple quantum approaches for robustness

## Research Applications

### Methodology Development
- Establish quantum frameworks for economic modeling
- Compare different quantum evolution approaches
- Develop parameter extraction techniques

### Policy Analysis
- Model trade policy interdependencies
- Analyze coupling strengths between countries
- Study phase space dynamics and stability

### Future Extensions
- **Floquet theory**: Time-periodic Hamiltonians for electoral cycles
- **Multi-country networks**: Scale beyond bilateral analysis
- **Machine learning**: Optimize parameter extraction
- **Real data validation**: OECD Export Restrictions Database

## Technical Notes

### Quantum Mechanical Framework
The framework treats trade policies as quantum observables with:
- **States**: |0⟩ (low restrictions), |1⟩ (high restrictions)
- **Superposition**: Countries can be in mixed policy states
- **Entanglement**: Genuine policy interdependence beyond correlation
- **Decoherence**: External noise degrades policy coordination

### Mathematical Foundation
- **Hamiltonians**: H = H₁ + H₂ + H_coupling for two-country systems
- **Evolution**: Three approaches (Schrödinger, von Neumann, Lindblad)
- **Observables**: σz measurements give policy restriction probabilities
- **Coupling**: σx ⊗ σx interaction terms model policy coordination

## Citation

[Add citation information when published]

## License

[Add license information]

## Contact

[Add contact information for questions/collaboration]
