# Quantum Trade Policy Analysis Framework

A comprehensive framework for modeling international trade policy dynamics using quantum mechanical evolution equations. Applies three quantum models (Schrödinger, von Neumann, Lindblad) to analyze trade policy coupling between countries.

## Overview

This framework fits individual country trade policies using enhanced multi-component models, then analyzes their interactions using quantum coupling dynamics. Currently demonstrates synthetic Egypt-Russia trade restrictions (2008-2018) with plans for real OECD data validation.

## Key Features

- **Modular fitting interface**: Easy to swap different time series fitting methods
- **Three quantum evolution models**: Schrödinger (pure states), von Neumann (mixed unitary), Lindblad (open system)
- **Automated coupling optimization**: Finds optimal inter-country interaction strengths
- **Comprehensive visualization**: 6 detailed analysis figures covering all aspects
- **Quantum parameter extraction**: Maps fitted parameters to physically meaningful quantum variables

## Quick Start

```python
# Run complete analysis
analyzer = CompleteFixedQuantumAnalysis(output_dir="quantum_analysis_results")
analyzer.run_complete_analysis()
```

## Framework Architecture

### Data Pipeline (Lines 57-258)
- **Data source** (lines 57-78): Currently synthetic data, easily replaceable with real OECD data
- **Individual fitting** (lines 150-200): Fits each country separately using enhanced quantum model
- **Parameter extraction** (lines 240-258): Converts fitted parameters to quantum variables

### Quantum Coupling Analysis (Lines 300+)
- **Hamiltonian construction**: Builds coupled quantum system from individual parameters
- **Three evolution models**: Compares different quantum mechanical approaches
- **Performance validation**: Comprehensive metrics and coupling optimization

## Current Fitting Method

**Enhanced Multi-Component Model:**
```
f(t) = [A₁sin(ω₁t + φ) + A₂sin(ω₂t + φ + π/3)] * exp(-γt) + C
```

**Parameters**: [A₁, ω₁, A₂, ω₂, γ, φ, offset] (7 total)
**Performance**: R² = 0.594-0.821 on synthetic data

## Easy Integration Guide

### Replacing the Fitting Method

To integrate your own time series fitting (e.g., chirped sine):

1. **Keep the interface** in `fit_enhanced_quantum_model()` (lines 150-200):
   - **Input**: `(time_points, data_values, country_name)`
   - **Output**: Dictionary with `{'params': [...], 'r2': float, 'prediction': array, 'residuals': array}`

2. **Replace the model function** (lines 180-195):
   ```python
   # Replace this function with your fitting approach
   def enhanced_quantum_model(self, t, *params):
       # Your chirped sine or other model here
       return fitted_values
   ```

3. **Adjust parameter mapping** (lines 240-258) if needed:
   - Current mapping expects 7 parameters → quantum variables (Δ, E₀, Ω, δ, γ)
   - Modify `extract_quantum_parameters()` for different parameter sets

### Replacing with Real Data

1. **Replace data source** (lines 57-78):
   ```python
   def load_real_oecd_data(self, file_path):
       df = pd.read_csv(file_path)  # or pd.read_excel()
       processed_data = {
           'Egypt': df['Egypt'].values,
           'Russian Federation': df['Russian Federation'].values
       }
       return df, processed_data
   ```

2. **Update main analysis** (line ~580):
   ```python
   # Replace this line:
   df, processed_data = self.create_enhanced_synthetic_data()
   # With:
   df, processed_data = self.load_real_oecd_data("data/oecd_restrictions.csv")
   ```

## Output Files

### Generated Figures
- `data_fitting_analysis.png` - Individual country fits and residual analysis
- `quantum_models_comparison.png` - Three-model performance comparison
- `coupling_strength_analysis.png` - Optimal coupling validation
- `model_hierarchy_explanation.png` - Quantum model theory overview
- `phase_space_analysis.png` - Phase space dynamics and trajectories
- `quantum_states_evolution.png` - Quantum state population evolution

### Data Files
- `synthetic_trade_data.csv/xlsx` - Generated trade policy data
- `quantum_parameters.csv/xlsx` - Extracted quantum parameters (Δ, E₀, Ω, δ, γ)
- `model_performance_metrics.csv/xlsx` - R², MAE, RMSE for all models
- `model_predictions.csv/xlsx` - Time series predictions from all quantum models
- `coupling_validation.csv/xlsx` - Coupling strength optimization results

## Dependencies

```python
numpy
scipy
matplotlib
pandas
```

## Current Performance

**Synthetic Data Results:**
- Egypt: R² = 0.594, Russia: R² = 0.821
- Successfully reproduces designed correlation (ρ = -0.690)
- Three quantum models show distinct evolution characteristics

## Research Pipeline

- **Stage 0** (Current): Synthetic data methodology validation
- **Stage 1** (Planned): Real OECD Export Restrictions Database
- **Stage 2** (Future): Floquet theory, multi-country networks, ML integration

## Usage Examples

### Basic Analysis
```python
analyzer = CompleteFixedQuantumAnalysis()
analyzer.run_complete_analysis()
```

### Custom Data Source
```python
analyzer = CompleteFixedQuantumAnalysis()
df, data = analyzer.load_real_oecd_data("my_data.csv")
# Continue with standard analysis...
```

### Individual Components
```python
# Just fitting
fit_result = analyzer.fit_enhanced_quantum_model(years, data, "Egypt")

# Just quantum analysis
quantum_params = analyzer.extract_quantum_parameters(fitted_params)
H, countries = analyzer.construct_hamiltonian(quantum_params)
```

## Contributing

This framework is designed for easy extension and collaboration:

1. **Better fitting methods**: Replace the multi-component sine model with advanced approaches
2. **Real data integration**: Add OECD, WTO, or other trade policy databases
3. **Extended quantum models**: Add Floquet theory, stochastic processes
4. **Multi-country networks**: Scale beyond bilateral analysis

## License

[Add your license here]

## Citation

[Add citation format when published]
