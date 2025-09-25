# TechArena 2025 Phase 1 - EMS Algorithm Solution

## Overview
This solution implements an Energy Management System (EMS) algorithm for optimizing Battery Energy Storage System (BESS) operations across multiple European electricity markets. The algorithm maximizes revenue through strategic participation in day-ahead energy markets and ancillary services (FCR and aFRR).

## Approach

### 1. Operation Optimization
The algorithm optimizes BESS charge/discharge strategies using:
- **Energy Arbitrage**: Identifies optimal times to charge (low prices) and discharge (high prices)
- **Ancillary Services**: Allocates capacity to FCR and aFRR markets when profitable
- **Constraint Management**: Respects battery physical limits (C-rate, SOC bounds, daily cycles)

Key features:
- 15-minute resolution optimization matching German/Austrian market structure
- Greedy algorithm with daily optimization windows
- Dynamic capacity allocation between energy and ancillary services

### 2. Investment Optimization
Evaluates ROI across 5 countries (DE, AT, CH, HU, CZ) considering:
- Country-specific WACC and inflation rates
- 10-year project horizon with NPV calculations
- Levelized ROI accounting for time value of money

### 3. Configuration Optimization
Tests multiple battery configurations:
- **C-rates**: 0.25, 0.33, 0.50 (determines maximum charge/discharge power)
- **Daily cycles**: 1.0, 1.5, 2.0 (limits daily energy throughput)
- Identifies optimal configuration for each market

## Technical Implementation

### Algorithm Components
1. **Data Processing**: Loads and aligns multi-resolution market data (15-min day-ahead, 4-hour ancillary)
2. **Optimization Engine**: Daily optimization with greedy price-based strategy
3. **Revenue Calculation**: Aggregates revenue from all market participation
4. **ROI Analysis**: Multi-year financial analysis with discounting

### Key Assumptions
- Battery efficiency: 95% round-trip
- SOC limits: 10% - 90%
- Investment cost: 200 EUR/kWh
- 1 MWh nominal capacity for normalized calculations

## Results Structure

### Output Files
1. **TechArena_Phase1_Configuration.csv**: Comparative analysis of all C-rate and cycle combinations
2. **TechArena_Phase1_Investment.csv**: Detailed ROI analysis with year-by-year projections
3. **TechArena_Phase1_Operation.csv**: Time-series operational data (charge/discharge/SOC/market participation)

## Installation & Execution

### Requirements
- Python 3.8+
- Dependencies listed in requirements.txt

### Running the Solution
```bash
# Install dependencies
pip install -r requirements.txt

# Run optimization
python main.py
```

## Performance Optimization
The algorithm balances computational efficiency with solution quality through:
- Daily optimization windows to reduce problem size
- Greedy heuristics for rapid decision-making
- Vectorized operations using NumPy/Pandas

## Future Enhancements
Potential improvements for subsequent phases:
- Mixed-integer linear programming for global optimization
- Stochastic optimization for price uncertainty
- Machine learning for price forecasting
- Battery degradation modeling

## Authors
TechArena 2025 Competition Submission - Phase 1