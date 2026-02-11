#!/bin/bash

# Sweep Results Analyzer Runner
# Analyzes best model YAML files from hyperparameter sweeps

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/analyze_sweep_results.py"
DEFAULT_SWEEPS_DIR="$PROJECT_ROOT/logs/sweeps"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/sweep_analysis_results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Sweep Results Analyzer

Analyzes best model YAML files from hyperparameter sweeps to extract
performance statistics and identify the best models for each dataset.

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -s, --sweeps-dir DIR    Directory containing sweep results (default: logs/sweeps)
    -o, --output-dir DIR    Directory to save analysis results (default: sweep_analysis_results)
    --dry-run              Show what would be analyzed without executing
    --list-configs         List all config files that would be analyzed

Examples:
    $0                                          # Analyze with defaults
    $0 -s custom/sweeps -o custom/output       # Custom directories
    $0 --list-configs                          # See what configs will be analyzed

The analyzer will:
    - Find all best model configuration files
    - Extract performance metrics and hyperparameters
    - Identify best models for each dataset
    - Generate comprehensive statistics
    - Create visualizations
    - Save results in multiple formats
EOF
}

# Check if sweeps directory exists and has configs
check_sweeps_directory() {
    local sweeps_dir="$1"
    
    if [[ ! -d "$sweeps_dir" ]]; then
        print_error "Sweeps directory not found: $sweeps_dir"
        print_status "Make sure you've run hyperparameter sweeps first"
        return 1
    fi
    
    # Check for config files
    local config_count=$(find "$sweeps_dir" -name "*config*.yaml" -o -name "*best*.yaml" | wc -l)
    
    if [[ $config_count -eq 0 ]]; then
        print_warning "No configuration files found in $sweeps_dir"
        print_status "Expected files like: best_models/*config.yaml or *best*config*.yaml"
        return 1
    fi
    
    print_status "Found $config_count configuration files to analyze"
    return 0
}

# List all config files that will be analyzed
list_config_files() {
    local sweeps_dir="$1"
    
    print_status "Configuration files that will be analyzed:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Use same search patterns as Python script
    find "$sweeps_dir" -type f \( -name "*config*.yaml" -o -name "*best*.yaml" \) | while read -r file; do
        echo "📄 $file"
    done
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Run the analysis
run_analysis() {
    local sweeps_dir="$1"
    local output_dir="$2"
    
    print_status "Starting sweep results analysis..."
    print_status "Sweeps directory: $sweeps_dir"
    print_status "Output directory: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Set PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Run the analysis
    if python3 "$PYTHON_SCRIPT" --sweeps-dir "$sweeps_dir" --output-dir "$output_dir"; then
        print_success "Analysis completed successfully!"
        return 0
    else
        print_error "Analysis failed!"
        return 1
    fi
}

# Show analysis results summary
show_results_summary() {
    local output_dir="$1"
    
    print_status "Analysis Results Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [[ -d "$output_dir" ]]; then
        echo "📁 Results Directory: $output_dir"
        echo "📊 Generated Files:"
        
        # Check for each expected output file
        local files=(
            "sweep_analysis_raw_data.csv:Raw experiment data"
            "sweep_statistics.yaml:Performance statistics"
            "best_models_per_dataset.yaml:Best model configurations"
            "analysis_summary.txt:Human-readable summary"
            "val_loss_by_model_dataset.png:Validation loss visualization"
            "hyperparameter_distributions.png:Hyperparameter distributions"
            "best_models_comparison.png:Best models comparison"
        )
        
        for file_desc in "${files[@]}"; do
            IFS=':' read -r filename description <<< "$file_desc"
            if [[ -f "$output_dir/$filename" ]]; then
                echo "  ✅ $filename - $description"
            else
                echo "  ❌ $filename - $description (not found)"
            fi
        done
        
        # Show summary from analysis_summary.txt if available
        if [[ -f "$output_dir/analysis_summary.txt" ]]; then
            echo ""
            echo "📋 Quick Summary:"
            head -15 "$output_dir/analysis_summary.txt" | tail -n +3
        fi
        
    else
        print_warning "Results directory not found: $output_dir"
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Parse arguments
SWEEPS_DIR="$DEFAULT_SWEEPS_DIR"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
DRY_RUN=false
LIST_CONFIGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--sweeps-dir)
            SWEEPS_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --list-configs)
            LIST_CONFIGS=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "🔍 Sweep Results Analyzer"
    echo "=========================="
    
    # Check sweeps directory
    if ! check_sweeps_directory "$SWEEPS_DIR"; then
        exit 1
    fi
    
    # List configs if requested
    if [[ "$LIST_CONFIGS" == true ]]; then
        list_config_files "$SWEEPS_DIR"
        exit 0
    fi
    
    # Dry run mode
    if [[ "$DRY_RUN" == true ]]; then
        print_status "DRY RUN MODE - What would be analyzed:"
        print_status "Sweeps directory: $SWEEPS_DIR"
        print_status "Output directory: $OUTPUT_DIR"
        echo ""
        list_config_files "$SWEEPS_DIR"
        print_status "Use without --dry-run to execute the analysis"
        exit 0
    fi
    
    # Run the analysis
    if run_analysis "$SWEEPS_DIR" "$OUTPUT_DIR"; then
        show_results_summary "$OUTPUT_DIR"
        print_success "All done! 🎉"
        echo ""
        print_status "To view results:"
        echo "  📄 Summary: cat $OUTPUT_DIR/analysis_summary.txt"
        echo "  📊 Data: $OUTPUT_DIR/sweep_analysis_raw_data.csv"
        echo "  📈 Plots: $OUTPUT_DIR/*.png"
    else
        exit 1
    fi
}

# Execute main function
main "$@"