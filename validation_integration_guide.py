"""
Integration Guide: How to Validate Your Quantum Framework

This shows exactly how to use the validation suite with YOUR existing quantum framework.
Updated to work with your streamlined_quantum_trade.py file.

STEP 1: This integration script
STEP 2: Run validation against your framework  
STEP 3: Only proceed with trade policy analysis if ALL tests pass
"""

import sys
import os

def validate_your_framework():
    """
    Complete validation of your quantum framework.
    Run this BEFORE applying to trade policy data.
    """
    print("🔬 VALIDATING YOUR QUANTUM FRAMEWORK")
    print("=" * 60)
    print("Testing against analytical solutions from quantum physics literature...")
    print()
    
    # Step 1: Try to import your framework
    print("📋 Step 1: Importing your quantum framework...")
    try:
        # Try to import from your streamlined_quantum_trade.py file
        from streamlined_quantum_trade import QuantumTradeEvolutionFramework
        print("✅ Successfully imported from streamlined_quantum_trade.py")
        quantum_framework = QuantumTradeEvolutionFramework()
        print("✅ Framework instance created")
    except ImportError as e:
        print(f"❌ Could not import your framework: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Make sure 'streamlined_quantum_trade.py' is in the same directory")
        print("  2. Check that the file contains 'QuantumTradeEvolutionFramework' class")
        print("  3. Verify there are no syntax errors in your framework file")
        
        # List files in current directory for debugging
        print(f"\n📁 Files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.py'):
                print(f"    {file}")
        
        return False, {}
    except Exception as e:
        print(f"❌ Error creating framework instance: {e}")
        print("💡 Check your QuantumTradeEvolutionFramework class for errors")
        return False, {}
    
    # Step 2: Import validation suite
    print("\n📋 Step 2: Setting up validation suite...")
    try:
        # Try different possible validation suite names
        validation_files = [
            'quantum_validation_suite_FINAL',
            'quantum_validation_suite', 
            'final_quantum_validation_suite'
        ]
        
        validator = None
        for filename in validation_files:
            try:
                module = __import__(filename)
                QuantumModelValidator = getattr(module, 'QuantumModelValidator')
                validator = QuantumModelValidator(tolerance=1e-6)
                validator.set_quantum_framework(quantum_framework)
                print(f"✅ Validation suite imported from {filename}.py")
                break
            except (ImportError, AttributeError):
                continue
        
        if validator is None:
            print("❌ Could not import validation suite")
            print("💡 Make sure one of these files exists in the current directory:")
            for filename in validation_files:
                print(f"    {filename}.py")
            return False, {}
            
        print("✅ Validator configured with your framework")
        
    except Exception as e:
        print(f"❌ Error setting up validation: {e}")
        return False, {}
    
    # Step 3: Run comprehensive validation
    print("\n📋 Step 3: Running validation tests...")
    print("Testing YOUR quantum framework against known analytical solutions:")
    print("  • Schrödinger: Rabi oscillations")
    print("  • Lindblad: Amplitude damping") 
    print("  • Floquet: Stroboscopic periodicity")
    print("  • Cross-model consistency checks")
    print()
    
    try:
        all_passed, results = validator.run_all_validation_tests()
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        print("💡 Check your quantum framework methods for compatibility")
        return False, {}
    
    # Step 4: Create validation plots
    print("\n📋 Step 4: Creating validation plots...")
    try:
        validator.create_validation_plots("your_framework_validation.png")
        print("✅ Validation plots saved")
    except Exception as e:
        print(f"⚠️ Warning: Could not create plots: {e}")
    
    # Step 5: Final decision
    print("\n📋 Step 5: Validation Decision")
    print("=" * 40)
    
    if all_passed:
        print("🎉 SUCCESS: All quantum models validated!")
        print("✅ Your framework correctly implements quantum physics")
        print("✅ Ready to proceed with trade policy analysis")
        print("\nValidation Results Summary:")
        
        for test_name, result in results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"  {status}: {result['test_name']}")
        
        print("\nNext steps:")
        print("  1. Run your enhanced trade policy analysis")
        print("  2. Replace synthetic parameters with real fitting methods")
        print("  3. Apply to actual OECD data")
        
    else:
        print("⚠️  WARNING: Some validation tests failed!")
        print("❌ Your quantum models have implementation errors")
        print("❌ DO NOT proceed with trade policy analysis yet")
        print("\nFailed tests:")
        
        for test_name, result in results.items():
            if not result['passed']:
                print(f"  ❌ {result['test_name']}")
                # Try to provide specific error information
                if 'max_error' in result:
                    print(f"     Max error: {result['max_error']:.2e}")
                if 'error' in result:
                    print(f"     Error: {result['error']:.2e}")
        
        print("\nRequired fixes:")
        print("  1. Check the failed test methods in your framework")
        print("  2. Verify numerical solver parameters")
        print("  3. Ensure proper state normalization")
        print("  4. Re-run this validation")
        print("  5. Only proceed after all tests pass")
    
    return all_passed, results

def quick_validation_check():
    """
    Quick validation using direct calculations (no framework needed).
    Use this to verify the validation suite works.
    """
    print("🔧 QUICK VALIDATION CHECK (Direct Calculations)")
    print("=" * 50)
    print("Testing validation suite with direct quantum calculations...")
    print("(This verifies the validation benchmarks are correct)")
    print()
    
    try:
        # Try to import validation suite
        validation_files = [
            'quantum_validation_suite_FINAL',
            'quantum_validation_suite',
            'final_quantum_validation_suite'
        ]
        
        validator = None
        for filename in validation_files:
            try:
                module = __import__(filename)
                QuantumModelValidator = getattr(module, 'QuantumModelValidator')
                validator = QuantumModelValidator(tolerance=1e-6)
                print(f"✅ Using validation suite from {filename}.py")
                break
            except (ImportError, AttributeError):
                continue
        
        if validator is None:
            print("❌ Could not find validation suite")
            return False, {}
        
        # Don't set framework - uses direct calculations
        all_passed, results = validator.run_all_validation_tests()
        validator.create_validation_plots("direct_validation_check.png")
        
        return all_passed, results
        
    except Exception as e:
        print(f"❌ Error in quick validation: {e}")
        return False, {}

def main():
    """Main function with user interaction."""
    print("🔬 QUANTUM FRAMEWORK VALIDATION")
    print("=" * 50)
    print("Choose validation mode:")
    print("1. Validate your quantum framework (streamlined_quantum_trade.py)")
    print("2. Quick check with direct calculations")
    print("3. Exit")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        success, results = validate_your_framework()
        if success:
            print("\n🚀 FRAMEWORK VALIDATED! Ready for trade policy analysis.")
        else:
            print("\n⚠️  FRAMEWORK NEEDS FIXES before proceeding.")
    
    elif choice == "2":
        success, results = quick_validation_check()
        if success:
            print("\n✅ Validation suite working correctly!")
            print("💡 Now try option 1 to test your actual framework.")
        else:
            print("\n❌ Issues with validation suite itself.")
    
    elif choice == "3":
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice. Please run again and enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
