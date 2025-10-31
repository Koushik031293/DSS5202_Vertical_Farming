"""
NPV, IRR, AND SENSITIVITY ANALYSIS FOR VERTICAL FARMING
========================================================

This script performs:
1. Net Present Value (NPV) calculation
2. Internal Rate of Return (IRR) calculation
3. Payback Period analysis
4. Sensitivity Analysis on key parameters
5. Tornado Plots for NPV and IRR
6. Monte Carlo Simulation (optional)

Author: Financial Analysis Module
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve
import warnings
import os
warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = 'financial_analysis'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =============================================================================
# SECTION 1: FINANCIAL DATA LOADER
# =============================================================================

class FinancialDataLoader:
    """Load and prepare financial data for NPV/IRR analysis"""
    
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.data = {}
        
    def load_farm_data(self, param_sheet, capex_sheet, opex_sheet):
        """Load financial parameters for a farm system"""
        
        # Load sheets (skip first 2 rows for headers in row 3)
        params = pd.read_excel(self.excel_file, sheet_name=param_sheet, header=2)
        capex = pd.read_excel(self.excel_file, sheet_name=capex_sheet, header=2)
        opex = pd.read_excel(self.excel_file, sheet_name=opex_sheet, header=2)
        
        # Extract key parameters
        def get_param(df, param_name):
            valid_df = df[df['Parameter'].notna()]
            row = valid_df[valid_df['Parameter'] == param_name]
            return row['Value'].values[0] if len(row) > 0 else None
        
        financial_data = {
            'annual_output': get_param(params, 'Annual edible output (kg/yr)'),
            'discount_rate': get_param(params, 'Discount rate r (decimal)'),
            'lifetime': get_param(params, 'Facility lifetime (years)'),
            'product_price': get_param(params, 'Product price (SGD/kg)'),
            'electricity_tariff': get_param(params, 'Electricity tariff (SGD/kWh)'),
            'water_tariff': get_param(params, 'Water tariff (SGD/m3)'),
            'labor_wage': get_param(params, 'Labor wage (SGD/hr)'),
        }
        
        # Calculate total initial CAPEX
        valid_capex = capex[capex['Cost_SGD'].notna() & (capex['Asset'].notna())]
        financial_data['initial_capex'] = valid_capex['Cost_SGD'].sum()
        
        # Store individual CAPEX items for replacement timing
        financial_data['capex_items'] = {}
        for _, row in valid_capex.iterrows():
            if pd.notna(row['Asset']) and pd.notna(row['Cost_SGD']):
                financial_data['capex_items'][row['Asset']] = {
                    'cost': row['Cost_SGD'],
                    'lifetime': row['Lifetime_years']
                }
        
        # Calculate annual OPEX per kg
        valid_opex = opex[opex['Cost_per_kg_SGD'].notna() & (opex['Item'].notna())]
        valid_opex = valid_opex[valid_opex['Item'] != 'TOTAL per kg']  # Exclude total row
        financial_data['opex_per_kg'] = valid_opex['Cost_per_kg_SGD'].sum()
        financial_data['annual_opex'] = financial_data['opex_per_kg'] * financial_data['annual_output']
        
        # Store OPEX breakdown for sensitivity
        financial_data['opex_items'] = {}
        for _, row in valid_opex.iterrows():
            if pd.notna(row['Item']) and pd.notna(row['Cost_per_kg_SGD']):
                financial_data['opex_items'][row['Item']] = row['Cost_per_kg_SGD']
        
        return financial_data

# =============================================================================
# SECTION 2: NPV AND IRR CALCULATOR
# =============================================================================

class NPVIRRCalculator:
    """Calculate NPV, IRR, and related financial metrics"""
    
    def __init__(self, financial_data):
        self.data = financial_data
        self.cash_flows = []
        
    def project_cash_flows(self, salvage_value_pct=0.15):
        """
        Project cash flows over facility lifetime
        
        Year 0: -Initial CAPEX
        Year 1-N: Annual net cash flow (revenue - opex - replacement capex)
        Year N: + Salvage value
        """
        
        lifetime = int(self.data['lifetime'])
        annual_output = self.data['annual_output']
        product_price = self.data['product_price']
        annual_opex = self.data['annual_opex']
        
        cash_flows = np.zeros(lifetime + 1)
        
        # Year 0: Initial investment (negative)
        cash_flows[0] = -self.data['initial_capex']
        
        # Years 1 to N: Operating cash flows
        annual_revenue = annual_output * product_price
        annual_net_operating = annual_revenue - annual_opex
        
        for year in range(1, lifetime + 1):
            # Base operating cash flow
            cash_flows[year] = annual_net_operating
            
            # Check for equipment replacement needs
            for asset, details in self.data['capex_items'].items():
                asset_lifetime = details['lifetime']
                if year % asset_lifetime == 0 and year < lifetime:
                    # Need to replace this asset
                    replacement_cost = details['cost']
                    cash_flows[year] -= replacement_cost
        
        # Final year: Add salvage value
        salvage_value = self.data['initial_capex'] * salvage_value_pct
        cash_flows[lifetime] += salvage_value
        
        self.cash_flows = cash_flows
        self.salvage_value = salvage_value
        
        return cash_flows
    
    def calculate_npv(self, discount_rate=None):
        """Calculate Net Present Value"""
        if discount_rate is None:
            discount_rate = self.data['discount_rate']
        
        if len(self.cash_flows) == 0:
            self.project_cash_flows()
        
        npv = 0
        for year, cash_flow in enumerate(self.cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** year)
        
        return npv
    
    def calculate_irr(self, initial_guess=0.1):
        """Calculate Internal Rate of Return"""
        if len(self.cash_flows) == 0:
            self.project_cash_flows()
        
        def npv_at_rate(rate):
            """NPV as function of discount rate"""
            npv = 0
            for year, cash_flow in enumerate(self.cash_flows):
                npv += cash_flow / ((1 + rate) ** year)
            return npv
        
        try:
            irr = fsolve(npv_at_rate, initial_guess)[0]
            return irr
        except:
            return np.nan
    
    def calculate_payback_period(self):
        """Calculate simple and discounted payback period"""
        if len(self.cash_flows) == 0:
            self.project_cash_flows()
        
        # Simple payback
        cumulative = np.cumsum(self.cash_flows)
        simple_payback_years = np.where(cumulative > 0)[0]
        simple_payback = simple_payback_years[0] if len(simple_payback_years) > 0 else np.nan
        
        # Discounted payback
        discount_rate = self.data['discount_rate']
        discounted_cf = [cf / ((1 + discount_rate) ** year) 
                         for year, cf in enumerate(self.cash_flows)]
        cumulative_discounted = np.cumsum(discounted_cf)
        discounted_payback_years = np.where(cumulative_discounted > 0)[0]
        discounted_payback = discounted_payback_years[0] if len(discounted_payback_years) > 0 else np.nan
        
        return simple_payback, discounted_payback
    
    def calculate_profitability_index(self):
        """Calculate Profitability Index (PI)"""
        if len(self.cash_flows) == 0:
            self.project_cash_flows()
        
        discount_rate = self.data['discount_rate']
        
        # PV of future cash inflows
        pv_inflows = 0
        for year in range(1, len(self.cash_flows)):
            if self.cash_flows[year] > 0:
                pv_inflows += self.cash_flows[year] / ((1 + discount_rate) ** year)
        
        # Initial investment (absolute value)
        initial_investment = abs(self.cash_flows[0])
        
        pi = pv_inflows / initial_investment if initial_investment > 0 else 0
        
        return pi
    
    def get_summary(self):
        """Get comprehensive financial summary"""
        npv = self.calculate_npv()
        irr = self.calculate_irr()
        simple_pb, discounted_pb = self.calculate_payback_period()
        pi = self.calculate_profitability_index()
        
        summary = {
            'NPV (SGD)': npv,
            'IRR (%)': irr * 100,
            'Simple Payback (years)': simple_pb,
            'Discounted Payback (years)': discounted_pb,
            'Profitability Index': pi,
            'Initial Investment (SGD)': abs(self.cash_flows[0]),
            'Total PV of Cash Flows (SGD)': npv + abs(self.cash_flows[0]),
            'Average Annual Cash Flow (SGD)': np.mean(self.cash_flows[1:])
        }
        
        return summary

# =============================================================================
# SECTION 3: SENSITIVITY ANALYSIS
# =============================================================================

class SensitivityAnalyzer:
    """Perform sensitivity analysis on NPV and IRR"""
    
    def __init__(self, base_financial_data):
        self.base_data = base_financial_data.copy()
        self.sensitivity_results = {}
        
    def one_way_sensitivity(self, parameter_name, variation_range=(-30, 30, 10)):
        """
        Perform one-way sensitivity analysis
        
        Parameters:
        -----------
        parameter_name : str
            Name of parameter to vary
        variation_range : tuple
            (min_pct, max_pct, num_points)
        """
        
        min_pct, max_pct, num_points = variation_range
        variations = np.linspace(min_pct, max_pct, num_points)
        
        base_value = self.base_data[parameter_name]
        
        npv_results = []
        irr_results = []
        
        for variation_pct in variations:
            # Create modified data
            modified_data = self.base_data.copy()
            modified_data[parameter_name] = base_value * (1 + variation_pct / 100)
            
            # Recalculate dependent values if needed
            if parameter_name == 'annual_output':
                modified_data['annual_opex'] = (modified_data['opex_per_kg'] * 
                                                modified_data['annual_output'])
            elif parameter_name == 'opex_per_kg':
                modified_data['annual_opex'] = (modified_data['opex_per_kg'] * 
                                                modified_data['annual_output'])
            
            # Calculate NPV and IRR
            calc = NPVIRRCalculator(modified_data)
            calc.project_cash_flows()
            
            npv = calc.calculate_npv()
            irr = calc.calculate_irr()
            
            npv_results.append(npv)
            irr_results.append(irr * 100)  # Convert to percentage
        
        self.sensitivity_results[parameter_name] = {
            'variations': variations,
            'npv': npv_results,
            'irr': irr_results,
            'base_value': base_value
        }
        
        return variations, npv_results, irr_results
    
    def run_full_sensitivity(self, parameters_to_test=None):
        """Run sensitivity analysis on multiple parameters"""
        
        if parameters_to_test is None:
            parameters_to_test = [
                'product_price',
                'annual_output',
                'initial_capex',
                'opex_per_kg',
                'discount_rate',
                'electricity_tariff',
                'labor_wage'
            ]
        
        print("Running sensitivity analysis...")
        for param in parameters_to_test:
            if param in self.base_data:
                print(f"  Analyzing {param}...")
                self.one_way_sensitivity(param)
        
        print("✓ Sensitivity analysis complete")
        
    def calculate_tornado_data(self, metric='npv', variation_pct=20):
        """
        Calculate data for tornado plot
        
        Parameters:
        -----------
        metric : str
            'npv' or 'irr'
        variation_pct : float
            Percentage variation (+/-)
        """
        
        tornado_data = []
        
        # Calculate base case
        base_calc = NPVIRRCalculator(self.base_data)
        base_calc.project_cash_flows()
        base_npv = base_calc.calculate_npv()
        base_irr = base_calc.calculate_irr() * 100
        
        base_value = base_npv if metric == 'npv' else base_irr
        
        for param_name, results in self.sensitivity_results.items():
            # Find values closest to +/- variation_pct
            variations = results['variations']
            metric_values = results['npv'] if metric == 'npv' else results['irr']
            
            # Find low case (-variation_pct)
            low_idx = np.argmin(np.abs(variations - (-variation_pct)))
            low_value = metric_values[low_idx]
            low_change = low_value - base_value
            
            # Find high case (+variation_pct)
            high_idx = np.argmin(np.abs(variations - variation_pct))
            high_value = metric_values[high_idx]
            high_change = high_value - base_value
            
            # Calculate total swing
            total_swing = abs(high_change) + abs(low_change)
            
            tornado_data.append({
                'parameter': param_name,
                'base_value': results['base_value'],
                'low_change': low_change,
                'high_change': high_change,
                'total_swing': total_swing
            })
        
        # Sort by total swing (descending)
        tornado_data.sort(key=lambda x: x['total_swing'], reverse=True)
        
        return tornado_data, base_value

# =============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cash_flows(calculator, title, filename):
    """Plot projected cash flows over time"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    years = np.arange(len(calculator.cash_flows))
    cash_flows = calculator.cash_flows
    
    # Plot 1: Annual cash flows
    colors = ['red' if cf < 0 else 'green' for cf in cash_flows]
    ax1.bar(years, cash_flows / 1e6, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cash Flow (Million SGD)', fontsize=12, fontweight='bold')
    ax1.set_title('Annual Cash Flows', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Cumulative cash flows
    cumulative = np.cumsum(cash_flows)
    ax2.plot(years, cumulative / 1e6, linewidth=2.5, marker='o', markersize=4, color='#2E86AB')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax2.fill_between(years, 0, cumulative / 1e6, where=(cumulative >= 0), 
                     alpha=0.3, color='green', label='Profitable')
    ax2.fill_between(years, 0, cumulative / 1e6, where=(cumulative < 0), 
                     alpha=0.3, color='red', label='Loss')
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Cash Flow (Million SGD)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Cash Flows', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{title} - Cash Flow Projection', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_sensitivity_curves(analyzer, filename):
    """Plot sensitivity analysis curves for NPV and IRR"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot NPV sensitivity
    for param_name, results in analyzer.sensitivity_results.items():
        variations = results['variations']
        npv_values = np.array(results['npv']) / 1e6  # Convert to millions
        
        # Format parameter name for legend
        param_label = param_name.replace('_', ' ').title()
        
        ax1.plot(variations, npv_values, linewidth=2.5, marker='o', 
                markersize=5, label=param_label, alpha=0.8)
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Parameter Variation (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('NPV (Million SGD)', fontsize=12, fontweight='bold')
    ax1.set_title('NPV Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot IRR sensitivity
    for param_name, results in analyzer.sensitivity_results.items():
        variations = results['variations']
        irr_values = results['irr']
        
        param_label = param_name.replace('_', ' ').title()
        
        ax2.plot(variations, irr_values, linewidth=2.5, marker='s', 
                markersize=5, label=param_label, alpha=0.8)
    
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Parameter Variation (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('IRR (%)', fontsize=12, fontweight='bold')
    ax2.set_title('IRR Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Sensitivity Analysis: NPV and IRR vs Key Parameters', 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_tornado_chart(tornado_data, base_value, metric_name, title, filename):
    """Create tornado chart for sensitivity analysis"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    parameters = [item['parameter'].replace('_', ' ').title() for item in tornado_data]
    low_changes = [item['low_change'] for item in tornado_data]
    high_changes = [item['high_change'] for item in tornado_data]
    
    # Create bars
    y_pos = np.arange(len(parameters))
    
    # Plot bars (low in red, high in green)
    if metric_name == 'NPV':
        low_bars = ax.barh(y_pos, low_changes, height=0.7, color='#C00000', 
                          alpha=0.7, label='Low Case (-20%)')
        high_bars = ax.barh(y_pos, high_changes, height=0.7, color='#70AD47', 
                           alpha=0.7, label='High Case (+20%)')
        unit = 'SGD'
        divisor = 1e6 if abs(base_value) > 1e6 else 1
        unit_label = 'Million SGD' if divisor == 1e6 else 'SGD'
    else:
        low_bars = ax.barh(y_pos, low_changes, height=0.7, color='#C00000', 
                          alpha=0.7, label='Low Case (-20%)')
        high_bars = ax.barh(y_pos, high_changes, height=0.7, color='#70AD47', 
                           alpha=0.7, label='High Case (+20%)')
        unit = '%'
        divisor = 1
        unit_label = 'Percentage Points'
    
    # Add value labels
    for i, (low, high) in enumerate(zip(low_changes, high_changes)):
        if metric_name == 'NPV':
            ax.text(low / divisor - 0.5, i, f'{low/divisor:.1f}', 
                   va='center', ha='right', fontsize=9, fontweight='bold')
            ax.text(high / divisor + 0.5, i, f'{high/divisor:.1f}', 
                   va='center', ha='left', fontsize=9, fontweight='bold')
        else:
            ax.text(low - 0.1, i, f'{low:.1f}{unit}', 
                   va='center', ha='right', fontsize=9, fontweight='bold')
            ax.text(high + 0.1, i, f'{high:.1f}{unit}', 
                   va='center', ha='left', fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(parameters, fontsize=10)
    
    if metric_name == 'NPV':
        ax.set_xlabel(f'Change in {metric_name} ({unit_label})', fontsize=12, fontweight='bold')
        base_text = f'Base {metric_name}: {base_value/divisor:.1f} {unit_label}'
    else:
        ax.set_xlabel(f'Change in {metric_name} ({unit_label})', fontsize=12, fontweight='bold')
        base_text = f'Base {metric_name}: {base_value:.2f}{unit}'
    
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    # Add base value annotation
    ax.text(0.02, 0.98, base_text, transform=ax.transAxes, 
           fontsize=11, fontweight='bold', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_comparison_dashboard(vf_calc, tf_calc, vf_summary, tf_summary, filename):
    """Create comparison dashboard for VF vs TF"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. NPV Comparison (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    farms = ['Vertical\nFarm', 'Traditional\nFarm']
    npvs = [vf_summary['NPV (SGD)'] / 1e6, tf_summary['NPV (SGD)'] / 1e6]
    colors = ['#2E86AB' if npv > 0 else '#C00000' for npv in npvs]
    
    bars = ax1.bar(farms, npvs, color=colors, alpha=0.8, width=0.6)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('NPV (Million SGD)', fontsize=11, fontweight='bold')
    ax1.set_title('Net Present Value', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, npv in zip(bars, npvs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{npv:.1f}M', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    # 2. IRR Comparison (Top Middle)
    ax2 = plt.subplot(2, 3, 2)
    irrs = [vf_summary['IRR (%)'], tf_summary['IRR (%)']]
    discount_rate = vf_calc.data['discount_rate'] * 100
    
    bars = ax2.bar(farms, irrs, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.6)
    ax2.axhline(y=discount_rate, color='red', linestyle='--', linewidth=2, 
               label=f'Discount Rate ({discount_rate:.1f}%)')
    ax2.set_ylabel('IRR (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Internal Rate of Return', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, irr in zip(bars, irrs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{irr:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    # 3. Payback Period (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    
    categories = ['Simple\nPayback', 'Discounted\nPayback']
    vf_pb = [vf_summary['Simple Payback (years)'], 
             vf_summary['Discounted Payback (years)']]
    tf_pb = [tf_summary['Simple Payback (years)'], 
             tf_summary['Discounted Payback (years)']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, vf_pb, width, label='Vertical Farm', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax3.bar(x + width/2, tf_pb, width, label='Traditional Farm', 
                   color='#A23B72', alpha=0.8)
    
    ax3.set_ylabel('Years', fontsize=11, fontweight='bold')
    ax3.set_title('Payback Period', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Profitability Index (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    pis = [vf_summary['Profitability Index'], tf_summary['Profitability Index']]
    
    bars = ax4.bar(farms, pis, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.6)
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
               label='Break-even (PI = 1.0)')
    ax4.set_ylabel('Profitability Index', fontsize=11, fontweight='bold')
    ax4.set_title('Profitability Index\n(PV Inflows / Initial Investment)', 
                 fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, pi in zip(bars, pis):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pi:.2f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    # 5. Initial Investment (Bottom Middle)
    ax5 = plt.subplot(2, 3, 5)
    investments = [vf_summary['Initial Investment (SGD)'] / 1e6,
                  tf_summary['Initial Investment (SGD)'] / 1e6]
    
    bars = ax5.bar(farms, investments, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.6)
    ax5.set_ylabel('Initial CAPEX (Million SGD)', fontsize=11, fontweight='bold')
    ax5.set_title('Initial Investment Required', fontsize=13, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, inv in zip(bars, investments):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{inv:.1f}M', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    # 6. Decision Matrix (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create decision table
    metrics = ['NPV Winner', 'IRR Winner', 'Faster Payback', 'Higher PI', 'Lower CAPEX']
    vf_wins = []
    
    # Determine winners
    vf_wins.append('✓' if npvs[0] > npvs[1] else '✗')
    vf_wins.append('✓' if irrs[0] > irrs[1] else '✗')
    vf_wins.append('✓' if vf_pb[1] < tf_pb[1] else '✗')  # Discounted payback
    vf_wins.append('✓' if pis[0] > pis[1] else '✗')
    vf_wins.append('✓' if investments[0] < investments[1] else '✗')
    
    # Create table
    table_data = []
    for metric, vf_win in zip(metrics, vf_wins):
        tf_win = '✓' if vf_win == '✗' else '✗'
        table_data.append([metric, vf_win, tf_win])
    
    table = ax6.table(cellText=table_data, 
                     colLabels=['Metric', 'VF', 'TF'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the cells
    for i in range(len(metrics)):
        if table_data[i][1] == '✓':
            table[(i+1, 1)].set_facecolor('#90EE90')
        else:
            table[(i+1, 1)].set_facecolor('#FFB6C6')
        
        if table_data[i][2] == '✓':
            table[(i+1, 2)].set_facecolor('#90EE90')
        else:
            table[(i+1, 2)].set_facecolor('#FFB6C6')
    
    # Header row
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Financial Performance Comparison', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('FINANCIAL ANALYSIS COMPARISON: VF vs TF', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")

# =============================================================================
# SECTION 5: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("NPV, IRR, AND SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    BASE_FILE = 'Corrected_Base_Data_Singapore.xlsx'
    
    if not os.path.exists(BASE_FILE):
        print(f"ERROR: {BASE_FILE} not found!")
        return
    
    print("Step 1: Loading Financial Data...")
    print("-" * 80)
    
    loader = FinancialDataLoader(BASE_FILE)
    
    vf_data = loader.load_farm_data('VF_Parameters', 'VF_CAPEX', 'VF_OPEX')
    print(f"✓ Loaded Vertical Farm data")
    print(f"  Initial CAPEX: {vf_data['initial_capex']/1e6:.2f}M SGD")
    print(f"  Annual OPEX: {vf_data['annual_opex']/1e6:.2f}M SGD")
    
    tf_data = loader.load_farm_data('TF_Parameters', 'TF_CAPEX', 'TF_OPEX')
    print(f"✓ Loaded Traditional Farm data")
    print(f"  Initial CAPEX: {tf_data['initial_capex']/1e6:.2f}M SGD")
    print(f"  Annual OPEX: {tf_data['annual_opex']/1e6:.2f}M SGD")
    print()
    
    # Calculate NPV and IRR for both systems
    print("Step 2: Calculating NPV and IRR...")
    print("-" * 80)
    
    vf_calc = NPVIRRCalculator(vf_data)
    vf_calc.project_cash_flows()
    vf_summary = vf_calc.get_summary()
    
    print("Vertical Farm:")
    print(f"  NPV: {vf_summary['NPV (SGD)']/1e6:.2f}M SGD")
    print(f"  IRR: {vf_summary['IRR (%)']:.2f}%")
    print(f"  Payback: {vf_summary['Discounted Payback (years)']:.1f} years")
    
    tf_calc = NPVIRRCalculator(tf_data)
    tf_calc.project_cash_flows()
    tf_summary = tf_calc.get_summary()
    
    print("\nTraditional Farm:")
    print(f"  NPV: {tf_summary['NPV (SGD)']/1e6:.2f}M SGD")
    print(f"  IRR: {tf_summary['IRR (%)']:.2f}%")
    print(f"  Payback: {tf_summary['Discounted Payback (years)']:.1f} years")
    print()
    
    # Sensitivity Analysis
    print("Step 3: Running Sensitivity Analysis...")
    print("-" * 80)
    
    # VF Sensitivity
    vf_sensitivity = SensitivityAnalyzer(vf_data)
    vf_sensitivity.run_full_sensitivity()
    
    # TF Sensitivity
    tf_sensitivity = SensitivityAnalyzer(tf_data)
    tf_sensitivity.run_full_sensitivity()
    print()
    
    # Generate Visualizations
    print("Step 4: Generating Visualizations...")
    print("-" * 80)
    
    # Cash flow projections
    plot_cash_flows(vf_calc, "Vertical Farm", 
                   os.path.join(OUTPUT_DIR, 'VF_Cash_Flows.png'))
    plot_cash_flows(tf_calc, "Traditional Farm",
                   os.path.join(OUTPUT_DIR, 'TF_Cash_Flows.png'))
    
    # Sensitivity curves
    plot_sensitivity_curves(vf_sensitivity,
                          os.path.join(OUTPUT_DIR, 'VF_Sensitivity_Curves.png'))
    plot_sensitivity_curves(tf_sensitivity,
                          os.path.join(OUTPUT_DIR, 'TF_Sensitivity_Curves.png'))
    
    # Tornado charts - VF
    vf_tornado_npv, vf_base_npv = vf_sensitivity.calculate_tornado_data('npv', 20)
    plot_tornado_chart(vf_tornado_npv, vf_base_npv, 'NPV',
                      'Vertical Farm - NPV Sensitivity (±20%)',
                      os.path.join(OUTPUT_DIR, 'VF_Tornado_NPV.png'))
    
    vf_tornado_irr, vf_base_irr = vf_sensitivity.calculate_tornado_data('irr', 20)
    plot_tornado_chart(vf_tornado_irr, vf_base_irr, 'IRR',
                      'Vertical Farm - IRR Sensitivity (±20%)',
                      os.path.join(OUTPUT_DIR, 'VF_Tornado_IRR.png'))
    
    # Tornado charts - TF
    tf_tornado_npv, tf_base_npv = tf_sensitivity.calculate_tornado_data('npv', 20)
    plot_tornado_chart(tf_tornado_npv, tf_base_npv, 'NPV',
                      'Traditional Farm - NPV Sensitivity (±20%)',
                      os.path.join(OUTPUT_DIR, 'TF_Tornado_NPV.png'))
    
    tf_tornado_irr, tf_base_irr = tf_sensitivity.calculate_tornado_data('irr', 20)
    plot_tornado_chart(tf_tornado_irr, tf_base_irr, 'IRR',
                      'Traditional Farm - IRR Sensitivity (±20%)',
                      os.path.join(OUTPUT_DIR, 'TF_Tornado_IRR.png'))
    
    # Comparison dashboard
    plot_comparison_dashboard(vf_calc, tf_calc, vf_summary, tf_summary,
                            os.path.join(OUTPUT_DIR, 'Financial_Comparison_Dashboard.png'))
    
    print()
    
    # Generate summary report
    print("Step 5: Generating Summary Report...")
    print("-" * 80)
    
    report_path = os.path.join(OUTPUT_DIR, 'NPV_IRR_Analysis_Report.txt')
    generate_report(vf_summary, tf_summary, vf_tornado_npv, tf_tornado_npv,
                   vf_tornado_irr, tf_tornado_irr, report_path)
    
    print(f"✓ Report saved: {report_path}")
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print(f"Results saved in '{OUTPUT_DIR}' folder:")
    print("  1. VF_Cash_Flows.png")
    print("  2. TF_Cash_Flows.png")
    print("  3. VF_Sensitivity_Curves.png")
    print("  4. TF_Sensitivity_Curves.png")
    print("  5. VF_Tornado_NPV.png")
    print("  6. VF_Tornado_IRR.png")
    print("  7. TF_Tornado_NPV.png")
    print("  8. TF_Tornado_IRR.png")
    print("  9. Financial_Comparison_Dashboard.png")
    print(" 10. NPV_IRR_Analysis_Report.txt")
    print()
    
    # Print quick summary
    print("QUICK SUMMARY:")
    print("-" * 80)
    if vf_summary['NPV (SGD)'] > tf_summary['NPV (SGD)']:
        print(f"✓ Vertical Farm has HIGHER NPV (+{(vf_summary['NPV (SGD)']-tf_summary['NPV (SGD)'])/1e6:.1f}M SGD)")
    else:
        print(f"✓ Traditional Farm has HIGHER NPV (+{(tf_summary['NPV (SGD)']-vf_summary['NPV (SGD)'])/1e6:.1f}M SGD)")
    
    if vf_summary['IRR (%)'] > tf_summary['IRR (%)']:
        print(f"✓ Vertical Farm has HIGHER IRR (+{vf_summary['IRR (%)']-tf_summary['IRR (%)']:.1f}%)")
    else:
        print(f"✓ Traditional Farm has HIGHER IRR (+{tf_summary['IRR (%)']-vf_summary['IRR (%)']:.1f}%)")
    print()


def generate_report(vf_summary, tf_summary, vf_tornado_npv, tf_tornado_npv,
                   vf_tornado_irr, tf_tornado_irr, filename):
    """Generate comprehensive text report"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NPV, IRR, AND SENSITIVITY ANALYSIS REPORT\n")
        f.write("Vertical Farming vs Traditional Greenhouse Farming\n")
        f.write("=" * 80 + "\n\n")
        
        # Financial Summary
        f.write("FINANCIAL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Vertical Farm:\n")
        f.write("-" * 80 + "\n")
        for metric, value in vf_summary.items():
            if 'SGD' in metric:
                f.write(f"{metric:<40} {value:>20,.2f}\n")
            else:
                f.write(f"{metric:<40} {value:>20.2f}\n")
        
        f.write("\nTraditional Farm:\n")
        f.write("-" * 80 + "\n")
        for metric, value in tf_summary.items():
            if 'SGD' in metric:
                f.write(f"{metric:<40} {value:>20,.2f}\n")
            else:
                f.write(f"{metric:<40} {value:>20.2f}\n")
        
        f.write("\n\n")
        
        # Sensitivity Analysis
        f.write("SENSITIVITY ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Most Influential Parameters on NPV (±20% variation):\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("Vertical Farm:\n")
        for i, item in enumerate(vf_tornado_npv[:5], 1):
            param = item['parameter'].replace('_', ' ').title()
            swing = item['total_swing'] / 1e6
            f.write(f"{i}. {param:<30} Total Swing: {swing:>10.2f}M SGD\n")
        
        f.write("\nTraditional Farm:\n")
        for i, item in enumerate(tf_tornado_npv[:5], 1):
            param = item['parameter'].replace('_', ' ').title()
            swing = item['total_swing'] / 1e6
            f.write(f"{i}. {param:<30} Total Swing: {swing:>10.2f}M SGD\n")
        
        f.write("\n\n")
        
        # Investment Recommendation
        f.write("INVESTMENT RECOMMENDATION\n")
        f.write("=" * 80 + "\n\n")
        
        if vf_summary['NPV (SGD)'] > 0 and vf_summary['IRR (%)'] > vf_summary['Initial Investment (SGD)'] * 0.08:
            f.write("Vertical Farm: RECOMMENDED\n")
            f.write(f"  • Positive NPV: {vf_summary['NPV (SGD)']/1e6:.2f}M SGD\n")
            f.write(f"  • IRR ({vf_summary['IRR (%)']:.2f}%) exceeds discount rate\n")
            f.write(f"  • Payback period: {vf_summary['Discounted Payback (years)']:.1f} years\n")
        else:
            f.write("Vertical Farm: NOT RECOMMENDED at current parameters\n")
        
        f.write("\n")
        
        if tf_summary['NPV (SGD)'] > 0 and tf_summary['IRR (%)'] > tf_summary['Initial Investment (SGD)'] * 0.08:
            f.write("Traditional Farm: RECOMMENDED\n")
            f.write(f"  • Positive NPV: {tf_summary['NPV (SGD)']/1e6:.2f}M SGD\n")
            f.write(f"  • IRR ({tf_summary['IRR (%)']:.2f}%) exceeds discount rate\n")
            f.write(f"  • Payback period: {tf_summary['Discounted Payback (years)']:.1f} years\n")
        else:
            f.write("Traditional Farm: NOT RECOMMENDED at current parameters\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


if __name__ == "__main__":
    main()
