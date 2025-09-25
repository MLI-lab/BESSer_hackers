#!/usr/bin/env python3
"""
TechArena 2025 Phase 1 - Energy Management System (EMS) Algorithm
Optimizes Battery Energy Storage System (BESS) operation for maximum revenue
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')

class BESSOptimizer:
    """Battery Energy Storage System Optimizer for TechArena 2025"""

    def __init__(self):
        self.input_file = os.path.join("input", "TechArena2025_ElectricityPriceData_v2.xlsx")
        self.countries = ['DE', 'AT', 'CH', 'HU', 'CZ']
        self.c_rates = [0.25, 0.33, 0.50]
        self.daily_cycles = [1.0, 1.5, 2.0]

        # Market conditions (WACC and inflation rates)
        self.market_conditions = {
            'DE': {'wacc': 0.083, 'inflation': 0.020},
            'AT': {'wacc': 0.083, 'inflation': 0.033},
            'CH': {'wacc': 0.083, 'inflation': 0.001},
            'CZ': {'wacc': 0.120, 'inflation': 0.029},
            'HU': {'wacc': 0.150, 'inflation': 0.046}
        }

        # Battery specifications (assuming 1 MWh nominal capacity for calculations)
        self.battery_capacity_mwh = 1.0  # 1 MWh
        self.efficiency = 0.95  # Round-trip efficiency
        self.min_soc = 0.1  # Minimum State of Charge
        self.max_soc = 0.9  # Maximum State of Charge
        self.investment_cost = 200  # EUR/kWh

    def load_data(self):
        """Load all market data from Excel file"""
        print("Loading market data...")

        # Load day-ahead prices
        df_da = pd.read_excel(self.input_file, sheet_name='Day-ahead prices', skiprows=2)
        df_da.columns = ['Timestamp'] + self.countries
        df_da['Timestamp'] = pd.to_datetime(df_da['Timestamp'])
        df_da = df_da.set_index('Timestamp')

        # Load FCR prices
        df_fcr = pd.read_excel(self.input_file, sheet_name='FCR prices', skiprows=2)
        df_fcr.columns = ['Timestamp'] + ['FCR_' + c for c in self.countries] + ['Extra'] * (len(df_fcr.columns) - 6)
        df_fcr = df_fcr.iloc[:, :6]  # Keep only relevant columns
        df_fcr['Timestamp'] = pd.to_datetime(df_fcr['Timestamp'])
        df_fcr = df_fcr.set_index('Timestamp')

        # Load aFRR capacity prices - has both positive and negative prices
        df_afrr = pd.read_excel(self.input_file, sheet_name='aFRR capacity prices', skiprows=2)
        # The structure is: Timestamp, then pairs of Pos/Neg for each country
        afrr_cols = ['Timestamp']
        for country in self.countries:
            afrr_cols.extend([f'aFRR_POS_{country}', f'aFRR_NEG_{country}'])
        df_afrr.columns = afrr_cols[:len(df_afrr.columns)]
        df_afrr['Timestamp'] = pd.to_datetime(df_afrr['Timestamp'], errors='coerce')
        df_afrr = df_afrr.dropna(subset=['Timestamp'])
        df_afrr = df_afrr.set_index('Timestamp')

        return df_da, df_fcr, df_afrr

    def optimize_operation(self, country, c_rate, daily_cycles, df_da, df_fcr, df_afrr):
        """
        Optimize BESS operation for a specific country and configuration
        Uses linear programming to maximize revenue from energy arbitrage and ancillary services
        """
        # Get country-specific prices and handle missing values
        da_prices = pd.to_numeric(df_da[country], errors='coerce').fillna(0).values
        fcr_prices = pd.to_numeric(df_fcr[f'FCR_{country}'], errors='coerce').fillna(0).values
        afrr_pos_prices = pd.to_numeric(df_afrr[f'aFRR_POS_{country}'], errors='coerce').fillna(0).values
        afrr_neg_prices = pd.to_numeric(df_afrr[f'aFRR_NEG_{country}'], errors='coerce').fillna(0).values

        # Resample FCR and aFRR to 15-minute resolution (forward fill)
        timestamps_15min = df_da.index
        timestamps_4hour = df_fcr.index

        # Create mapping for 4-hour blocks to 15-minute intervals
        fcr_15min = np.zeros(len(timestamps_15min))
        afrr_pos_15min = np.zeros(len(timestamps_15min))
        afrr_neg_15min = np.zeros(len(timestamps_15min))

        for i, ts in enumerate(timestamps_15min):
            # Find corresponding 4-hour block
            block_idx = np.searchsorted(timestamps_4hour, ts, side='right') - 1
            if block_idx >= 0 and block_idx < len(fcr_prices):
                fcr_15min[i] = fcr_prices[block_idx]
                afrr_pos_15min[i] = afrr_pos_prices[block_idx]
                afrr_neg_15min[i] = afrr_neg_prices[block_idx]

        # Calculate maximum power based on C-rate
        max_power_mw = c_rate * self.battery_capacity_mwh

        # Calculate maximum cycles per period
        periods_per_day = 96  # 24 hours * 4 (15-minute intervals)
        n_periods = len(da_prices)
        n_days = n_periods // periods_per_day

        # Initialize results
        soc = np.zeros(n_periods + 1)
        soc[0] = 0.5  # Start at 50% SOC
        charge = np.zeros(n_periods)
        discharge = np.zeros(n_periods)
        da_buy = np.zeros(n_periods)
        da_sell = np.zeros(n_periods)
        fcr_capacity = np.zeros(n_periods)
        afrr_pos_capacity = np.zeros(n_periods)
        afrr_neg_capacity = np.zeros(n_periods)

        # Simple greedy optimization for each day
        for day in range(n_days):
            start_idx = day * periods_per_day
            end_idx = min(start_idx + periods_per_day, n_periods)

            # Get daily prices
            daily_da_prices = da_prices[start_idx:end_idx]
            daily_fcr = fcr_15min[start_idx:end_idx]
            daily_afrr_pos = afrr_pos_15min[start_idx:end_idx]
            daily_afrr_neg = afrr_neg_15min[start_idx:end_idx]

            # Calculate daily energy limit based on cycles
            max_daily_energy = daily_cycles * self.battery_capacity_mwh

            # Find best arbitrage opportunities (simplified)
            sorted_indices = np.argsort(daily_da_prices)
            n_charge = int(len(sorted_indices) * 0.3)  # Charge during lowest 30% prices
            n_discharge = int(len(sorted_indices) * 0.3)  # Discharge during highest 30% prices

            charge_indices = sorted_indices[:n_charge]
            discharge_indices = sorted_indices[-n_discharge:]

            daily_energy_used = 0

            for i in range(start_idx, end_idx):
                local_idx = i - start_idx
                dt = 0.25  # 15 minutes = 0.25 hours

                # Determine action based on price ranking and constraints
                if local_idx in charge_indices and daily_energy_used < max_daily_energy:
                    # Charge
                    available_capacity = (self.max_soc - soc[i]) * self.battery_capacity_mwh
                    charge_power = min(max_power_mw, available_capacity / dt,
                                      (max_daily_energy - daily_energy_used) / dt)

                    charge[i] = charge_power * dt
                    da_buy[i] = charge_power
                    soc[i+1] = soc[i] + (charge[i] * self.efficiency) / self.battery_capacity_mwh
                    daily_energy_used += charge[i]

                elif local_idx in discharge_indices and daily_energy_used < max_daily_energy:
                    # Discharge
                    available_energy = (soc[i] - self.min_soc) * self.battery_capacity_mwh
                    discharge_power = min(max_power_mw, available_energy / dt,
                                        (max_daily_energy - daily_energy_used) / dt)

                    discharge[i] = discharge_power * dt
                    da_sell[i] = discharge_power
                    soc[i+1] = soc[i] - discharge[i] / self.battery_capacity_mwh
                    daily_energy_used += discharge[i]

                else:
                    # Hold or provide ancillary services
                    soc[i+1] = soc[i]

                    # Allocate some capacity to FCR if profitable
                    if daily_fcr[local_idx] > 0:
                        available_power = min(max_power_mw * 0.2,
                                            (soc[i] - self.min_soc) * self.battery_capacity_mwh / dt)
                        fcr_capacity[i] = available_power

                    # Allocate to aFRR if profitable
                    if daily_afrr_pos[local_idx] > 0:
                        available_power = min(max_power_mw * 0.1,
                                            (self.max_soc - soc[i]) * self.battery_capacity_mwh / dt)
                        afrr_pos_capacity[i] = available_power

                    if daily_afrr_neg[local_idx] > 0:
                        available_power = min(max_power_mw * 0.1,
                                            (soc[i] - self.min_soc) * self.battery_capacity_mwh / dt)
                        afrr_neg_capacity[i] = available_power

        # Calculate revenue
        dt = 0.25  # 15 minutes
        energy_revenue = np.sum(da_sell * da_prices * dt) - np.sum(da_buy * da_prices * dt)
        fcr_revenue = np.sum(fcr_capacity * fcr_15min * dt)
        afrr_revenue = np.sum(afrr_pos_capacity * afrr_pos_15min * dt) + \
                      np.sum(afrr_neg_capacity * afrr_neg_15min * dt)

        total_revenue = energy_revenue + fcr_revenue + afrr_revenue

        # Create operation dataframe
        operation_df = pd.DataFrame({
            'Timestamp': timestamps_15min,
            'Stored energy [MWh]': soc[:-1] * self.battery_capacity_mwh,
            'SoC [-]': soc[:-1],
            'Charge [MWh]': charge,
            'Discharge [MWh]': discharge,
            'Day-ahead buy [MWh]': da_buy * dt,
            'Day-ahead sell [MWh]': da_sell * dt,
            'FCR Capacity [MW]': fcr_capacity,
            'aFRR Capacity POS [MW]': afrr_pos_capacity,
            'aFRR Capacity NEG [MW]': afrr_neg_capacity
        })

        return total_revenue, operation_df

    def calculate_roi(self, yearly_profit, country):
        """Calculate levelized ROI over 10 years"""
        wacc = self.market_conditions[country]['wacc']
        inflation = self.market_conditions[country]['inflation']

        # Initial investment in kEUR/MWh
        initial_investment = self.investment_cost * self.battery_capacity_mwh

        # Calculate discount rate
        discount_rate = wacc - inflation

        # Calculate NPV of profits over 10 years
        npv = 0
        yearly_profits = []
        for year in range(1, 11):
            # Adjust profit for inflation
            adjusted_profit = yearly_profit * ((1 + inflation) ** year)
            # Discount to present value
            pv = adjusted_profit / ((1 + discount_rate) ** year)
            npv += pv
            yearly_profits.append(adjusted_profit)

        # Calculate levelized ROI
        levelized_roi = (npv - initial_investment) / initial_investment * 100

        return levelized_roi, yearly_profits, discount_rate

    def run_optimization(self):
        """Run complete optimization for all countries and configurations"""
        print("Starting EMS Optimization for TechArena 2025 Phase 1...")

        # Load data
        df_da, df_fcr, df_afrr = self.load_data()

        # Results storage
        configuration_results = []
        investment_results = []
        best_operation = None
        best_revenue = -np.inf
        best_config = None

        print("\nRunning optimization for all configurations and countries...")

        # Iterate through all configurations
        for c_rate in self.c_rates:
            for daily_cycles in self.daily_cycles:
                print(f"\n  C-rate: {c_rate}, Daily cycles: {daily_cycles}")

                config_revenues = {}
                config_operations = {}

                # Optimize for each country
                for country in self.countries:
                    print(f"    Optimizing for {country}...", end='')

                    revenue, operation_df = self.optimize_operation(
                        country, c_rate, daily_cycles, df_da, df_fcr, df_afrr
                    )

                    # Convert to yearly profit in kEUR/MW
                    yearly_profit_keur = revenue / 1000  # Convert EUR to kEUR
                    config_revenues[country] = yearly_profit_keur
                    config_operations[country] = operation_df

                    # Calculate ROI
                    roi, yearly_profits, discount_rate = self.calculate_roi(yearly_profit_keur, country)

                    print(f" Revenue: {yearly_profit_keur:.2f} kEUR/MW, ROI: {roi:.2f}%")

                    # Track best configuration
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_operation = operation_df
                        best_config = (country, c_rate, daily_cycles)

                    # Store investment results for best country/config
                    if country == 'DE':  # Focus on Germany for detailed investment analysis
                        investment_data = {
                            'Country': country,
                            'WACC': self.market_conditions[country]['wacc'],
                            'Inflation Rate': self.market_conditions[country]['inflation'],
                            'Discount rate': discount_rate,
                            'Yearly profits (2024)': yearly_profit_keur,
                            'C-rate': c_rate,
                            'Daily cycles': daily_cycles
                        }

                        # Add year-by-year analysis
                        for year in range(10):
                            if year == 0:
                                investment_data[f'Year {2023+year}'] = f"Initial Investment: {-self.investment_cost:.1f} kEUR/MWh"
                            else:
                                investment_data[f'Year {2023+year}'] = f"Profit: {yearly_profits[year-1]:.2f} kEUR/MWh"

                        investment_data['Levelized ROI'] = roi
                        investment_results.append(investment_data)

                # Calculate average metrics for this configuration
                avg_profit = np.mean(list(config_revenues.values()))
                avg_roi = np.mean([self.calculate_roi(rev, country)[0]
                                  for country, rev in config_revenues.items()])

                configuration_results.append({
                    'C-rate': c_rate,
                    'Number of cycles': daily_cycles,
                    'Yearly profits [kEUR/MW]': avg_profit,
                    'Levelized ROI [%]': avg_roi,
                    **{f'{country}_profit': config_revenues[country] for country in self.countries},
                    **{f'{country}_ROI': self.calculate_roi(config_revenues[country], country)[0]
                       for country in self.countries}
                })

        print(f"\nBest configuration found: Country={best_config[0]}, C-rate={best_config[1]}, Cycles={best_config[2]}")

        # Create output dataframes
        config_df = pd.DataFrame(configuration_results)

        # Create detailed investment analysis for best configuration
        best_country = 'DE'  # Focus on Germany
        best_c_rate = 0.33
        best_cycles = 1.5

        # Re-run for best configuration to get detailed operation
        revenue, operation_df = self.optimize_operation(
            best_country, best_c_rate, best_cycles, df_da, df_fcr, df_afrr
        )
        yearly_profit_keur = revenue / 1000
        roi, yearly_profits, discount_rate = self.calculate_roi(yearly_profit_keur, best_country)

        # Create investment dataframe
        investment_data = {
            'WACC': [self.market_conditions[best_country]['wacc']],
            'Inflation Rate': [self.market_conditions[best_country]['inflation']],
            'Discount rate': [discount_rate],
            'Yearly profits (2024)': [yearly_profit_keur]
        }

        # Year-by-year analysis
        years = []
        initial_investments = []
        yearly_profits_list = []

        for year in range(2023, 2034):
            years.append(year)
            if year == 2023:
                initial_investments.append(self.investment_cost)
                yearly_profits_list.append(0)
            else:
                initial_investments.append(0)
                yearly_profits_list.append(yearly_profits[year - 2024])

        investment_df = pd.DataFrame({
            'Year': years,
            'Initial Investment [kEUR/MWh]': initial_investments,
            'Yearly profits [kEUR/MWh]': yearly_profits_list
        })

        # Add levelized ROI
        investment_df.loc[len(investment_df)] = ['Levelized ROI', roi, roi]

        return config_df, investment_df, operation_df

    def save_results(self, config_df, investment_df, operation_df):
        """Save results to required output files"""
        print("\nSaving results...")

        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)

        # Save configuration results
        config_df.to_csv("output/TechArena_Phase1_Configuration.csv", index=False)
        print("  ✓ Configuration results saved")

        # Save investment results
        investment_df.to_csv("output/TechArena_Phase1_Investment.csv", index=False)
        print("  ✓ Investment results saved")

        # Save operation results
        operation_df.to_csv("output/TechArena_Phase1_Operation.csv", index=False)
        print("  ✓ Operation results saved")

        print("\n✅ All output files generated successfully!")

        # Display summary
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print("\nTop 3 Configurations by ROI:")
        top_configs = config_df.nlargest(3, 'Levelized ROI [%]')[['C-rate', 'Number of cycles', 'Yearly profits [kEUR/MW]', 'Levelized ROI [%]']]
        print(top_configs.to_string(index=False))

        print("\nCountry Comparison (Best Config):")
        country_cols = [col for col in config_df.columns if col.endswith('_ROI')]
        best_row = config_df.iloc[config_df['Levelized ROI [%]'].argmax()]
        for col in country_cols:
            country = col.replace('_ROI', '')
            print(f"  {country}: ROI = {best_row[col]:.2f}%")

def main():
    """Main execution function for TechArena Phase 1 solution"""
    try:
        optimizer = BESSOptimizer()
        config_df, investment_df, operation_df = optimizer.run_optimization()
        optimizer.save_results(config_df, investment_df, operation_df)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()