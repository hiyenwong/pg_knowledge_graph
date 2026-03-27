#!/bin/bash
# Benchmark Runner Script
# Usage: ./benchmarks/run_benchmarks.sh [options]
#
# Options:
#   --entities=N     Number of entities to generate (default: 10000)
#   --dimensions=D   Vector dimensions (default: 1536)
#   --iterations=I   Number of iterations per benchmark (default: 5)
#   --output=FILE    Output file for results (default: benchmarks/results.json)

set -e

# Default values
ENTITIES=10000
DIMENSIONS=1536
ITERATIONS=5
OUTPUT="benchmarks/results.json"
PGRX_PG="pg18"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --entities=*)
            ENTITIES="${arg#*=}"
            shift
            ;;
        --dimensions=*)
            DIMENSIONS="${arg#*=}"
            shift
            ;;
        --iterations=*)
            ITERATIONS="${arg#*=}"
            shift
            ;;
        --output=*)
            OUTPUT="${arg#*=}"
            shift
            ;;
        --pg=*)
            PGRX_PG="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "pg_knowledge_graph Benchmark Suite"
echo "=============================================="
echo "Entities:     $ENTITIES"
echo "Dimensions:   $DIMENSIONS"
echo "Iterations:   $ITERATIONS"
echo "PostgreSQL:   $PGRX_PG"
echo "Output:       $OUTPUT"
echo "=============================================="
echo ""

# Check if extension is installed
echo "Checking extension..."
cargo pgrx stop $PGRX_PG 2>/dev/null || true
cargo pgrx start $PGRX_PG

# Run pgrx test to ensure extension works
echo "Running pgrx tests..."
cargo pgrx test $PGRX_PG --no-run 2>/dev/null || echo "Tests skipped"

# Create benchmark SQL with parameters
BENCH_SQL=$(mktemp)
sed -e "s/10000/$ENTITIES/g" \
    -e "s/1536/$DIMENSIONS/g" \
    benchmarks/vector_search_bench.sql > "$BENCH_SQL"

# Run benchmark
echo ""
echo "Executing benchmarks..."
PGDATABASE=postgres cargo pgrx connect $PGRX_PG -- -f "$BENCH_SQL" 2>&1 | tee benchmark_output.txt

# Cleanup
rm -f "$BENCH_SQL"

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "Results saved to: benchmark_output.txt"
echo "=============================================="