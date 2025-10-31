"""
COMPREHENSIVE LIFE CYCLE COSTING (LCC) ANALYSIS
================================================

Standalone economic analysis for Vertical Farming vs Traditional Farming
with detailed cost breakdowns, visualizations, and scenario analysis.

This script performs:
1. Complete CAPEX analysis (initial investment breakdown)
2. Complete OPEX analysis (operating cost breakdown)
3. Levelized Cost calculation (LCOv - cost per kg)
4. Cost structure comparison
5. Breakeven analysis
6. Scenario analysis (optimistic, baseline, pessimistic)
7. Cost sensitivity analysis
8. Annual cost projections

Author: LCC Module
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = 'lcc_analysis'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =============================================================================
# SECTION 1: LCC DATA LOADER
# =============================================================================

class LCCDataLoader:
    """Load and prepare Life Cycle Costing data"""
    
    def __init__(self, excel_file='Corrected_Base_Data_Singapore.xlsx'):
        self.excel_file = excel_file
        self.vf_data = {}
        self.tf_data = {}
        
    def load_system_data(self, param_sheet, capex_sheet, opex_sheet, system_name):
        """Load complete LCC data for a farming system"""
        
        # Load parameters
        params = pd.read_excel(self.excel_file, sheet_name=param_sheet, header=2)
        
        def get_param(param_name):
            valid_df = params[params['Parameter'].notna()]
            row = valid_df[valid_df['Parameter'] == param_name]
            if len(row) > 0:
                return row['Value'].values[0]
            return None
        
        # Extract key parameters
        system_data = {
            'name': system_name,
            'annual_output': get_param('Annual edible output (kg/yr)'),
            'discount_rate': get_param('Discount rate r (decimal)'),
            'lifetime': get_param('Facility lifetime (years)'),
            'product_price': get_param('Product price (SGD/kg)'),
        }
        
        # Load CAPEX data
        capex = pd.read_excel(self.excel_file, sheet_name=capex_sheet, header=2)
        valid_capex = capex[capex['Cost_SGD'].notna() & (capex['Asset'].notna())]
        
        system_data['capex_items'] = {}
        system_data['capex_total'] = 0
        
        # Get discount rate for annualization
        discount_rate = system_data['discount_rate']
        
        for _, row in valid_capex.iterrows():
            if pd.notna(row['Asset']) and pd.notna(row['Cost_SGD']):
                asset_name = row['Asset']
                cost = row['Cost_SGD']
                lifetime = row['Lifetime_years']
                
                # Calculate annualized cost using capital recovery factor
                # Formula: CAPEX * [r(1+r)^n] / [(1+r)^n - 1]
                if lifetime > 0 and discount_rate > 0:
                    crf = (discount_rate * (1 + discount_rate)**lifetime) / \
                          ((1 + discount_rate)**lifetime - 1)
                    annualized = cost * crf
                else:
                    # If lifetime or discount rate is 0, spread evenly over facility lifetime
                    annualized = cost / system_data['lifetime'] if system_data['lifetime'] > 0 else 0
                
                system_data['capex_items'][asset_name] = {
                    'cost': cost,
                    'lifetime': lifetime,
                    'annualized': annualized
                }
                system_data['capex_total'] += cost
        
        system_data['capex_annualized_total'] = sum(
            item['annualized'] for item in system_data['capex_items'].values()
        )
        
        # Load OPEX data
        opex = pd.read_excel(self.excel_file, sheet_name=opex_sheet, header=2)
        valid_opex = opex[opex['Cost_per_kg_SGD'].notna() & (opex['Item'].notna())]
        valid_opex = valid_opex[valid_opex['Item'] != 'TOTAL per kg']
        
        system_data['opex_items'] = {}
        system_data['opex_per_kg'] = 0
        
        for _, row in valid_opex.iterrows():
            if pd.notna(row['Item']) and pd.notna(row['Cost_per_kg_SGD']):
                item_name = row['Item']
                cost_per_kg = row['Cost_per_kg_SGD']
                
                system_data['opex_items'][item_name] = cost_per_kg
                system_data['opex_per_kg'] += cost_per_kg
        
        system_data['opex_annual'] = system_data['opex_per_kg'] * system_data['annual_output']
        
        # Calculate CAPEX per kg (annualized)
        system_data['capex_per_kg'] = (system_data['capex_annualized_total'] / 
                                       system_data['annual_output'])
        
        # Calculate Levelized Cost of Vegetables (LCOv)
        system_data['lcov'] = system_data['capex_per_kg'] + system_data['opex_per_kg']
        
        # Calculate revenues and profits
        system_data['revenue_annual'] = (system_data['product_price'] * 
                                        system_data['annual_output'])
        system_data['cost_annual'] = (system_data['capex_annualized_total'] + 
                                     system_data['opex_annual'])
        system_data['profit_annual'] = (system_data['revenue_annual'] - 
                                       system_data['cost_annual'])
        system_data['profit_margin'] = (system_data['profit_annual'] / 
                                       system_data['revenue_annual'] * 100 
                                       if system_data['revenue_annual'] > 0 else 0)
        
        return system_data
    
    def load_all_data(self):
        """Load data for both VF and TF systems"""
        
        self.vf_data = self.load_system_data(
            'VF_Parameters', 'VF_CAPEX', 'VF_OPEX', 'Vertical Farm'
        )
        
        self.tf_data = self.load_system_data(
            'TF_Parameters', 'TF_CAPEX', 'TF_OPEX', 'Traditional Farm'
        )
        
        return self.vf_data, self.tf_data

# =============================================================================
# SECTION 2: LCC ANALYZER
# =============================================================================

class LCCAnalyzer:
    """Perform comprehensive LCC analysis"""
    
    def __init__(self, vf_data, tf_data):
        self.vf = vf_data
        self.tf = tf_data
        
    def calculate_cost_ratios(self):
        """Calculate comparative cost ratios"""
        
        ratios = {
            'capex_ratio': self.vf['capex_total'] / self.tf['capex_total'],
            'opex_per_kg_ratio': self.vf['opex_per_kg'] / self.tf['opex_per_kg'],
            'lcov_ratio': self.vf['lcov'] / self.tf['lcov'],
            'output_ratio': self.vf['annual_output'] / self.tf['annual_output'],
            'profit_ratio': (self.vf['profit_annual'] / self.tf['profit_annual'] 
                           if self.tf['profit_annual'] > 0 else 0)
        }
        
        return ratios
    
    def breakeven_analysis(self):
        """Calculate breakeven points"""
        
        # Breakeven production (at current prices)
        vf_breakeven_kg = (self.vf['capex_annualized_total'] / 
                          (self.vf['product_price'] - self.vf['opex_per_kg'])
                          if (self.vf['product_price'] - self.vf['opex_per_kg']) > 0 
                          else np.inf)
        
        tf_breakeven_kg = (self.tf['capex_annualized_total'] / 
                          (self.tf['product_price'] - self.tf['opex_per_kg'])
                          if (self.tf['product_price'] - self.tf['opex_per_kg']) > 0 
                          else np.inf)
        
        # Breakeven price (at current production)
        vf_breakeven_price = self.vf['lcov']
        tf_breakeven_price = self.tf['lcov']
        
        # Capacity utilization at breakeven
        vf_capacity_util = (vf_breakeven_kg / self.vf['annual_output'] * 100 
                           if self.vf['annual_output'] > 0 else 0)
        tf_capacity_util = (tf_breakeven_kg / self.tf['annual_output'] * 100 
                           if self.tf['annual_output'] > 0 else 0)
        
        return {
            'VF': {
                'breakeven_production_kg': vf_breakeven_kg,
                'breakeven_price_sgd': vf_breakeven_price,
                'capacity_utilization_pct': vf_capacity_util
            },
            'TF': {
                'breakeven_production_kg': tf_breakeven_kg,
                'breakeven_price_sgd': tf_breakeven_price,
                'capacity_utilization_pct': tf_capacity_util
            }
        }
    
    def scenario_analysis(self):
        """Perform scenario analysis (optimistic, baseline, pessimistic)"""
        
        scenarios = {}
        
        # Baseline (current)
        scenarios['baseline'] = {
            'VF': {'lcov': self.vf['lcov'], 'profit_annual': self.vf['profit_annual']},
            'TF': {'lcov': self.tf['lcov'], 'profit_annual': self.tf['profit_annual']}
        }
        
        # Optimistic: -15% OPEX, +10% output, +5% price
        scenarios['optimistic'] = {
            'VF': {
                'opex_per_kg': self.vf['opex_per_kg'] * 0.85,
                'output': self.vf['annual_output'] * 1.10,
                'price': self.vf['product_price'] * 1.05,
            },
            'TF': {
                'opex_per_kg': self.tf['opex_per_kg'] * 0.85,
                'output': self.tf['annual_output'] * 1.10,
                'price': self.tf['product_price'] * 1.05,
            }
        }
        
        # Calculate optimistic LCOv
        scenarios['optimistic']['VF']['lcov'] = (
            self.vf['capex_annualized_total'] / scenarios['optimistic']['VF']['output'] +
            scenarios['optimistic']['VF']['opex_per_kg']
        )
        scenarios['optimistic']['TF']['lcov'] = (
            self.tf['capex_annualized_total'] / scenarios['optimistic']['TF']['output'] +
            scenarios['optimistic']['TF']['opex_per_kg']
        )
        
        # Calculate optimistic profit
        scenarios['optimistic']['VF']['profit_annual'] = (
            scenarios['optimistic']['VF']['price'] * scenarios['optimistic']['VF']['output'] -
            (self.vf['capex_annualized_total'] + 
             scenarios['optimistic']['VF']['opex_per_kg'] * scenarios['optimistic']['VF']['output'])
        )
        scenarios['optimistic']['TF']['profit_annual'] = (
            scenarios['optimistic']['TF']['price'] * scenarios['optimistic']['TF']['output'] -
            (self.tf['capex_annualized_total'] + 
             scenarios['optimistic']['TF']['opex_per_kg'] * scenarios['optimistic']['TF']['output'])
        )
        
        # Pessimistic: +15% OPEX, -10% output, -5% price
        scenarios['pessimistic'] = {
            'VF': {
                'opex_per_kg': self.vf['opex_per_kg'] * 1.15,
                'output': self.vf['annual_output'] * 0.90,
                'price': self.vf['product_price'] * 0.95,
            },
            'TF': {
                'opex_per_kg': self.tf['opex_per_kg'] * 1.15,
                'output': self.tf['annual_output'] * 0.90,
                'price': self.tf['product_price'] * 0.95,
            }
        }
        
        # Calculate pessimistic LCOv
        scenarios['pessimistic']['VF']['lcov'] = (
            self.vf['capex_annualized_total'] / scenarios['pessimistic']['VF']['output'] +
            scenarios['pessimistic']['VF']['opex_per_kg']
        )
        scenarios['pessimistic']['TF']['lcov'] = (
            self.tf['capex_annualized_total'] / scenarios['pessimistic']['TF']['output'] +
            scenarios['pessimistic']['TF']['opex_per_kg']
        )
        
        # Calculate pessimistic profit
        scenarios['pessimistic']['VF']['profit_annual'] = (
            scenarios['pessimistic']['VF']['price'] * scenarios['pessimistic']['VF']['output'] -
            (self.vf['capex_annualized_total'] + 
             scenarios['pessimistic']['VF']['opex_per_kg'] * scenarios['pessimistic']['VF']['output'])
        )
        scenarios['pessimistic']['TF']['profit_annual'] = (
            scenarios['pessimistic']['TF']['price'] * scenarios['pessimistic']['TF']['output'] -
            (self.tf['capex_annualized_total'] + 
             scenarios['pessimistic']['TF']['opex_per_kg'] * scenarios['pessimistic']['TF']['output'])
        )
        
        return scenarios

# =============================================================================
# SECTION 3: VISUALIZATION FUNCTIONS
# =============================================================================

def plot_capex_breakdown(vf_data, tf_data, filename):
    """Plot detailed CAPEX breakdown comparison"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # VF CAPEX breakdown (pie chart)
    vf_capex_items = vf_data['capex_items']
    vf_labels = list(vf_capex_items.keys())
    vf_values = [item['cost'] / 1e6 for item in vf_capex_items.values()]
    
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(vf_labels)))
    wedges1, texts1, autotexts1 = ax1.pie(vf_values, labels=vf_labels, autopct='%1.1f%%',
                                           startangle=90, colors=colors1, textprops={'fontsize': 9})
    ax1.set_title(f'Vertical Farm CAPEX Breakdown\nTotal: {vf_data["capex_total"]/1e6:.1f}M SGD',
                 fontsize=13, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts1:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # TF CAPEX breakdown (pie chart)
    tf_capex_items = tf_data['capex_items']
    tf_labels = list(tf_capex_items.keys())
    tf_values = [item['cost'] / 1e6 for item in tf_capex_items.values()]
    
    colors2 = plt.cm.Purples(np.linspace(0.4, 0.9, len(tf_labels)))
    wedges2, texts2, autotexts2 = ax2.pie(tf_values, labels=tf_labels, autopct='%1.1f%%',
                                           startangle=90, colors=colors2, textprops={'fontsize': 9})
    ax2.set_title(f'Traditional Farm CAPEX Breakdown\nTotal: {tf_data["capex_total"]/1e6:.1f}M SGD',
                 fontsize=13, fontweight='bold')
    
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Side-by-side comparison (bar chart)
    all_assets = set(vf_labels + tf_labels)
    
    vf_costs = []
    tf_costs = []
    asset_names = []
    
    for asset in all_assets:
        asset_names.append(asset)
        vf_costs.append(vf_capex_items.get(asset, {'cost': 0})['cost'] / 1e6)
        tf_costs.append(tf_capex_items.get(asset, {'cost': 0})['cost'] / 1e6)
    
    # Sort by VF cost
    sorted_data = sorted(zip(asset_names, vf_costs, tf_costs), key=lambda x: x[1], reverse=True)
    asset_names, vf_costs, tf_costs = zip(*sorted_data)
    
    y = np.arange(len(asset_names))
    width = 0.35
    
    bars1 = ax3.barh(y + width/2, vf_costs, width, label='Vertical Farm', 
                    color='#2E86AB', alpha=0.8)
    bars2 = ax3.barh(y - width/2, tf_costs, width, label='Traditional Farm',
                    color='#A23B72', alpha=0.8)
    
    ax3.set_yticks(y)
    ax3.set_yticklabels(asset_names, fontsize=9)
    ax3.set_xlabel('Cost (Million SGD)', fontsize=11, fontweight='bold')
    ax3.set_title('CAPEX Comparison by Asset', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Capital Expenditure (CAPEX) Analysis', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_opex_breakdown(vf_data, tf_data, filename):
    """Plot detailed OPEX breakdown comparison"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # VF OPEX waterfall
    ax1 = plt.subplot(2, 2, 1)
    
    vf_opex_items = vf_data['opex_items']
    vf_labels = list(vf_opex_items.keys())
    vf_values = list(vf_opex_items.values())
    
    # Sort by value
    sorted_vf = sorted(zip(vf_labels, vf_values), key=lambda x: x[1], reverse=True)
    vf_labels, vf_values = zip(*sorted_vf)
    
    colors = ['#2E86AB' if v > 0 else '#C00000' for v in vf_values]
    bars = ax1.barh(vf_labels, vf_values, color=colors, alpha=0.8)
    
    ax1.set_xlabel('Cost per kg (SGD/kg)', fontsize=11, fontweight='bold')
    ax1.set_title(f'VF OPEX Breakdown\nTotal: {vf_data["opex_per_kg"]:.3f} SGD/kg',
                 fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, vf_values):
        width = bar.get_width()
        ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # TF OPEX waterfall
    ax2 = plt.subplot(2, 2, 2)
    
    tf_opex_items = tf_data['opex_items']
    tf_labels = list(tf_opex_items.keys())
    tf_values = list(tf_opex_items.values())
    
    sorted_tf = sorted(zip(tf_labels, tf_values), key=lambda x: x[1], reverse=True)
    tf_labels, tf_values = zip(*sorted_tf)
    
    colors = ['#A23B72' if v > 0 else '#C00000' for v in tf_values]
    bars = ax2.barh(tf_labels, tf_values, color=colors, alpha=0.8)
    
    ax2.set_xlabel('Cost per kg (SGD/kg)', fontsize=11, fontweight='bold')
    ax2.set_title(f'TF OPEX Breakdown\nTotal: {tf_data["opex_per_kg"]:.3f} SGD/kg',
                 fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, value in zip(bars, tf_values):
        width = bar.get_width()
        ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # OPEX comparison (stacked bar)
    ax3 = plt.subplot(2, 2, 3)
    
    # Get all unique OPEX items
    all_items = set(list(vf_opex_items.keys()) + list(tf_opex_items.keys()))
    
    vf_opex_dict = {k: v for k, v in zip(vf_opex_items.keys(), vf_opex_items.values())}
    tf_opex_dict = {k: v for k, v in zip(tf_opex_items.keys(), tf_opex_items.values())}
    
    # Create stacked bar data
    items_for_stack = []
    vf_stack = []
    tf_stack = []
    
    for item in all_items:
        items_for_stack.append(item)
        vf_stack.append(vf_opex_dict.get(item, 0))
        tf_stack.append(tf_opex_dict.get(item, 0))
    
    # Sort by average value
    avg_values = [(vf + tf) / 2 for vf, tf in zip(vf_stack, tf_stack)]
    sorted_data = sorted(zip(items_for_stack, vf_stack, tf_stack, avg_values),
                        key=lambda x: x[3], reverse=True)
    items_for_stack, vf_stack, tf_stack, _ = zip(*sorted_data)
    
    x = np.arange(2)
    width = 0.6
    
    bottom_vf = 0
    bottom_tf = 0
    colors_stack = plt.cm.Set3(np.linspace(0, 1, len(items_for_stack)))
    
    for i, (item, vf_val, tf_val) in enumerate(zip(items_for_stack, vf_stack, tf_stack)):
        ax3.bar(0, vf_val, width, bottom=bottom_vf, label=item if i < 10 else '',
               color=colors_stack[i], alpha=0.9)
        ax3.bar(1, tf_val, width, bottom=bottom_tf, color=colors_stack[i], alpha=0.9)
        bottom_vf += vf_val
        bottom_tf += tf_val
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Vertical Farm', 'Traditional Farm'], fontsize=11)
    ax3.set_ylabel('Cost per kg (SGD/kg)', fontsize=11, fontweight='bold')
    ax3.set_title('OPEX Structure Comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    ax3.grid(axis='y', alpha=0.3)
    
    # Add total labels
    ax3.text(0, bottom_vf + 0.1, f'Total:\n{vf_data["opex_per_kg"]:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.text(1, bottom_tf + 0.1, f'Total:\n{tf_data["opex_per_kg"]:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Annual OPEX comparison
    ax4 = plt.subplot(2, 2, 4)
    
    systems = ['Vertical\nFarm', 'Traditional\nFarm']
    annual_opex = [vf_data['opex_annual'] / 1e6, tf_data['opex_annual'] / 1e6]
    
    bars = ax4.bar(systems, annual_opex, color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.5)
    ax4.set_ylabel('Annual OPEX (Million SGD)', fontsize=11, fontweight='bold')
    ax4.set_title('Total Annual Operating Costs', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, annual_opex):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Operating Expenditure (OPEX) Analysis', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_lcov_comparison(vf_data, tf_data, filename):
    """Plot Levelized Cost of Vegetables (LCOv) comparison"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # LCOv breakdown (stacked bar)
    systems = ['Vertical\nFarm', 'Traditional\nFarm']
    capex_per_kg = [vf_data['capex_per_kg'], tf_data['capex_per_kg']]
    opex_per_kg = [vf_data['opex_per_kg'], tf_data['opex_per_kg']]
    
    x = np.arange(len(systems))
    width = 0.5
    
    bars1 = ax1.bar(x, capex_per_kg, width, label='CAPEX (annualized)',
                   color='#5B9BD5', alpha=0.8)
    bars2 = ax1.bar(x, opex_per_kg, width, bottom=capex_per_kg,
                   label='OPEX', color='#70AD47', alpha=0.8)
    
    ax1.set_ylabel('Cost per kg (SGD/kg)', fontsize=11, fontweight='bold')
    ax1.set_title('LCOv: Cost Structure per kg', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (capex, opex) in enumerate(zip(capex_per_kg, opex_per_kg)):
        total = capex + opex
        ax1.text(i, total + 0.1, f'LCOv:\n{total:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.text(i, capex/2, f'{capex:.2f}',
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax1.text(i, capex + opex/2, f'{opex:.2f}',
                ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    # LCOv vs Product Price
    lcov = [vf_data['lcov'], tf_data['lcov']]
    prices = [vf_data['product_price'], tf_data['product_price']]
    
    x = np.arange(len(systems))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, lcov, width, label='LCOv (Cost)',
                   color='#C00000', alpha=0.8)
    bars2 = ax2.bar(x + width/2, prices, width, label='Product Price',
                   color='#70AD47', alpha=0.8)
    
    ax2.set_ylabel('SGD per kg', fontsize=11, fontweight='bold')
    ax2.set_title('LCOv vs Selling Price', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels and profit margin
    for i, (cost, price) in enumerate(zip(lcov, prices)):
        margin = ((price - cost) / price * 100) if price > 0 else 0
        
        ax2.text(i - width/2, cost + 0.1, f'{cost:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(i + width/2, price + 0.1, f'{price:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Profit margin annotation
        color = 'green' if margin > 0 else 'red'
        ax2.annotate(f'Margin:\n{margin:.1f}%',
                    xy=(i, max(cost, price) + 0.3),
                    ha='center', fontsize=9, fontweight='bold', color=color)
    
    # Annual costs comparison
    cost_categories = ['CAPEX\n(annualized)', 'OPEX', 'Total Cost']
    vf_costs = [vf_data['capex_annualized_total'] / 1e6,
                vf_data['opex_annual'] / 1e6,
                vf_data['cost_annual'] / 1e6]
    tf_costs = [tf_data['capex_annualized_total'] / 1e6,
                tf_data['opex_annual'] / 1e6,
                tf_data['cost_annual'] / 1e6]
    
    x = np.arange(len(cost_categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, vf_costs, width, label='Vertical Farm',
                   color='#2E86AB', alpha=0.8)
    bars2 = ax3.bar(x + width/2, tf_costs, width, label='Traditional Farm',
                   color='#A23B72', alpha=0.8)
    
    ax3.set_ylabel('Annual Cost (Million SGD)', fontsize=11, fontweight='bold')
    ax3.set_title('Annual Cost Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cost_categories)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Profitability comparison
    metrics = ['Revenue', 'Cost', 'Profit']
    vf_metrics = [vf_data['revenue_annual'] / 1e6,
                  vf_data['cost_annual'] / 1e6,
                  vf_data['profit_annual'] / 1e6]
    tf_metrics = [tf_data['revenue_annual'] / 1e6,
                  tf_data['cost_annual'] / 1e6,
                  tf_data['profit_annual'] / 1e6]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, vf_metrics, width, label='Vertical Farm',
                   color='#2E86AB', alpha=0.8)
    bars2 = ax4.bar(x + width/2, tf_metrics, width, label='Traditional Farm',
                   color='#A23B72', alpha=0.8)
    
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_ylabel('Amount (Million SGD)', fontsize=11, fontweight='bold')
    ax4.set_title('Annual Financial Performance', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M', ha='center', va=va, fontsize=9, fontweight='bold')
    
    plt.suptitle('Levelized Cost of Vegetables (LCOv) Analysis',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_breakeven_analysis(analyzer, filename):
    """Plot breakeven analysis"""
    
    breakeven = analyzer.breakeven_analysis()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Breakeven production
    systems = ['Vertical\nFarm', 'Traditional\nFarm']
    breakeven_prod = [breakeven['VF']['breakeven_production_kg'] / 1e6,
                     breakeven['TF']['breakeven_production_kg'] / 1e6]
    actual_prod = [analyzer.vf['annual_output'] / 1e6,
                  analyzer.tf['annual_output'] / 1e6]
    
    x = np.arange(len(systems))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, breakeven_prod, width, label='Breakeven Production',
                   color='#C00000', alpha=0.8)
    bars2 = ax1.bar(x + width/2, actual_prod, width, label='Actual Production',
                   color='#70AD47', alpha=0.8)
    
    ax1.set_ylabel('Production (Million kg/year)', fontsize=11, fontweight='bold')
    ax1.set_title('Breakeven vs Actual Production', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (be_prod, act_prod) in enumerate(zip(breakeven_prod, actual_prod)):
        margin = ((act_prod - be_prod) / act_prod * 100) if act_prod > 0 else 0
        
        ax1.text(i - width/2, be_prod + 0.5, f'{be_prod:.1f}M',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + width/2, act_prod + 0.5, f'{act_prod:.1f}M',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        color = 'green' if margin > 0 else 'red'
        ax1.annotate(f'Margin:\n{margin:.0f}%',
                    xy=(i, max(be_prod, act_prod) + 1),
                    ha='center', fontsize=9, fontweight='bold', color=color)
    
    # Capacity utilization at breakeven
    capacity_util = [breakeven['VF']['capacity_utilization_pct'],
                    breakeven['TF']['capacity_utilization_pct']]
    
    bars = ax2.barh(systems, capacity_util, color=['#2E86AB', '#A23B72'], alpha=0.8)
    ax2.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Full Capacity')
    ax2.set_xlabel('Capacity Utilization at Breakeven (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Breakeven Capacity Utilization', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, value in zip(bars, capacity_util):
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # Breakeven price
    breakeven_price = [breakeven['VF']['breakeven_price_sgd'],
                      breakeven['TF']['breakeven_price_sgd']]
    actual_price = [analyzer.vf['product_price'],
                   analyzer.tf['product_price']]
    
    x = np.arange(len(systems))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, breakeven_price, width, label='Breakeven Price',
                   color='#C00000', alpha=0.8)
    bars2 = ax3.bar(x + width/2, actual_price, width, label='Actual Price',
                   color='#70AD47', alpha=0.8)
    
    ax3.set_ylabel('Price (SGD/kg)', fontsize=11, fontweight='bold')
    ax3.set_title('Breakeven vs Actual Selling Price', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(systems)
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (be_price, act_price) in enumerate(zip(breakeven_price, actual_price)):
        margin = ((act_price - be_price) / act_price * 100) if act_price > 0 else 0
        
        ax3.text(i - width/2, be_price + 0.1, f'{be_price:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(i + width/2, act_price + 0.1, f'{act_price:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        color = 'green' if margin > 0 else 'red'
        ax3.annotate(f'Margin:\n{margin:.0f}%',
                    xy=(i, max(be_price, act_price) + 0.3),
                    ha='center', fontsize=9, fontweight='bold', color=color)
    
    # Profit margin sensitivity
    price_range = np.linspace(3, 8, 50)
    
    vf_profits = []
    tf_profits = []
    
    for price in price_range:
        vf_profit = (price - analyzer.vf['lcov']) / price * 100 if price > 0 else 0
        tf_profit = (price - analyzer.tf['lcov']) / price * 100 if price > 0 else 0
        vf_profits.append(vf_profit)
        tf_profits.append(tf_profit)
    
    ax4.plot(price_range, vf_profits, linewidth=2.5, marker='o', markersize=4,
            label='Vertical Farm', color='#2E86AB')
    ax4.plot(price_range, tf_profits, linewidth=2.5, marker='s', markersize=4,
            label='Traditional Farm', color='#A23B72')
    
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
    ax4.axvline(x=analyzer.vf['product_price'], color='blue', linestyle=':', 
               linewidth=2, alpha=0.5, label='VF Current Price')
    ax4.axvline(x=analyzer.tf['product_price'], color='purple', linestyle=':',
               linewidth=2, alpha=0.5, label='TF Current Price')
    
    ax4.set_xlabel('Product Price (SGD/kg)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Profit Margin (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Profit Margin Sensitivity to Price', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Breakeven Analysis', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_scenario_analysis(analyzer, filename):
    """Plot scenario analysis (optimistic, baseline, pessimistic)"""
    
    scenarios = analyzer.scenario_analysis()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # LCOv across scenarios
    scenario_names = ['Optimistic', 'Baseline', 'Pessimistic']
    vf_lcov = [scenarios['optimistic']['VF']['lcov'],
               scenarios['baseline']['VF']['lcov'],
               scenarios['pessimistic']['VF']['lcov']]
    tf_lcov = [scenarios['optimistic']['TF']['lcov'],
               scenarios['baseline']['TF']['lcov'],
               scenarios['pessimistic']['TF']['lcov']]
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, vf_lcov, width, label='Vertical Farm',
                   color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, tf_lcov, width, label='Traditional Farm',
                   color='#A23B72', alpha=0.8)
    
    ax1.set_ylabel('LCOv (SGD/kg)', fontsize=11, fontweight='bold')
    ax1.set_title('LCOv Under Different Scenarios', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Profit across scenarios
    vf_profit = [scenarios['optimistic']['VF']['profit_annual'] / 1e6,
                 scenarios['baseline']['VF']['profit_annual'] / 1e6,
                 scenarios['pessimistic']['VF']['profit_annual'] / 1e6]
    tf_profit = [scenarios['optimistic']['TF']['profit_annual'] / 1e6,
                 scenarios['baseline']['TF']['profit_annual'] / 1e6,
                 scenarios['pessimistic']['TF']['profit_annual'] / 1e6]
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, vf_profit, width, label='Vertical Farm',
                   color='#2E86AB', alpha=0.8)
    bars2 = ax2.bar(x + width/2, tf_profit, width, label='Traditional Farm',
                   color='#A23B72', alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Annual Profit (Million SGD)', fontsize=11, fontweight='bold')
    ax2.set_title('Profitability Under Different Scenarios', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M', ha='center', va=va, fontsize=9, fontweight='bold')
    
    # Scenario assumptions table
    ax3.axis('off')
    
    table_data = [
        ['Optimistic', '-15%', '+10%', '+5%'],
        ['Baseline', '0%', '0%', '0%'],
        ['Pessimistic', '+15%', '-10%', '-5%']
    ]
    
    table = ax3.table(cellText=table_data,
                     colLabels=['Scenario', 'OPEX Change', 'Output Change', 'Price Change'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.2, 1, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color code
    for i in range(3):
        if i == 0:  # Optimistic
            for j in range(1, 4):
                table[(i+1, j)].set_facecolor('#90EE90')
        elif i == 2:  # Pessimistic
            for j in range(1, 4):
                table[(i+1, j)].set_facecolor('#FFB6C6')
    
    # Header
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Scenario Assumptions', fontsize=13, fontweight='bold', pad=50)
    
    # Range of outcomes
    ax4.axis('off')
    
    # Create box plot style visualization
    metrics = ['LCOv (SGD/kg)', 'Profit (M SGD/yr)']
    
    vf_ranges = [
        [min(vf_lcov), scenarios['baseline']['VF']['lcov'], max(vf_lcov)],
        [min(vf_profit), scenarios['baseline']['VF']['profit_annual'] / 1e6, max(vf_profit)]
    ]
    
    tf_ranges = [
        [min(tf_lcov), scenarios['baseline']['TF']['lcov'], max(tf_lcov)],
        [min(tf_profit), scenarios['baseline']['TF']['profit_annual'] / 1e6, max(tf_profit)]
    ]
    
    y_pos = [0.7, 0.3]
    
    for i, (metric, vf_range, tf_range) in enumerate(zip(metrics, vf_ranges, tf_ranges)):
        # VF range
        ax4.plot([vf_range[0], vf_range[2]], [y_pos[i] + 0.05, y_pos[i] + 0.05],
                'o-', color='#2E86AB', linewidth=3, markersize=8, label='VF' if i == 0 else '')
        ax4.plot([vf_range[1]], [y_pos[i] + 0.05], 'D',
                color='#2E86AB', markersize=10)
        
        # TF range
        ax4.plot([tf_range[0], tf_range[2]], [y_pos[i] - 0.05, y_pos[i] - 0.05],
                'o-', color='#A23B72', linewidth=3, markersize=8, label='TF' if i == 0 else '')
        ax4.plot([tf_range[1]], [y_pos[i] - 0.05], 'D',
                color='#A23B72', markersize=10)
        
        # Labels
        ax4.text(-0.1, y_pos[i], metric, ha='right', va='center',
                fontsize=10, fontweight='bold')
        
        # Value labels
        ax4.text(vf_range[0], y_pos[i] + 0.08, f'{vf_range[0]:.2f}',
                ha='center', fontsize=8, color='#2E86AB')
        ax4.text(vf_range[2], y_pos[i] + 0.08, f'{vf_range[2]:.2f}',
                ha='center', fontsize=8, color='#2E86AB')
        
        ax4.text(tf_range[0], y_pos[i] - 0.08, f'{tf_range[0]:.2f}',
                ha='center', fontsize=8, color='#A23B72')
        ax4.text(tf_range[2], y_pos[i] - 0.08, f'{tf_range[2]:.2f}',
                ha='center', fontsize=8, color='#A23B72')
    
    ax4.set_xlim(-0.2, max(max(vf_lcov), max(tf_lcov)) * 1.1)
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=10, loc='upper right')
    ax4.set_title('Range of Outcomes Across Scenarios', fontsize=13, fontweight='bold')
    
    plt.suptitle('Scenario Analysis: Optimistic, Baseline, Pessimistic',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")


def plot_cost_comparison_dashboard(vf_data, tf_data, filename):
    """Create comprehensive cost comparison dashboard"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Total cost comparison (top left)
    ax1 = plt.subplot(2, 3, 1)
    
    categories = ['Initial\nCAPEX', 'Annual\nOPEX', 'Cost\nper kg']
    vf_values = [vf_data['capex_total'] / 1e6,
                 vf_data['opex_annual'] / 1e6,
                 vf_data['lcov']]
    tf_values = [tf_data['capex_total'] / 1e6,
                 tf_data['opex_annual'] / 1e6,
                 tf_data['lcov']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Normalize for visualization (different scales)
    vf_norm = [v / max(vf_values[i], tf_values[i]) for i, v in enumerate(vf_values)]
    tf_norm = [v / max(vf_values[i], tf_values[i]) for i, v in enumerate(tf_values)]
    
    bars1 = ax1.bar(x - width/2, vf_norm, width, label='Vertical Farm',
                   color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, tf_norm, width, label='Traditional Farm',
                   color='#A23B72', alpha=0.8)
    
    ax1.set_ylabel('Normalized Value', fontsize=10, fontweight='bold')
    ax1.set_title('Cost Metrics Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add actual values as text
    labels = [f'VF: {vf_values[i]:.1f}M\nTF: {tf_values[i]:.1f}M' if i < 2
              else f'VF: {vf_values[i]:.2f}\nTF: {tf_values[i]:.2f}'
              for i in range(len(categories))]
    
    for i, label in enumerate(labels):
        ax1.text(i, 1.1, label, ha='center', fontsize=7)
    
    # 2. CAPEX breakdown pie charts (top middle and right)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    
    # VF CAPEX
    vf_capex_top = sorted(vf_data['capex_items'].items(),
                         key=lambda x: x[1]['cost'], reverse=True)[:5]
    vf_labels = [item[0][:15] for item in vf_capex_top]
    vf_values = [item[1]['cost'] / 1e6 for item in vf_capex_top]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(vf_labels)))
    ax2.pie(vf_values, labels=vf_labels, autopct='%1.0f%%', startangle=90,
           colors=colors, textprops={'fontsize': 8})
    ax2.set_title(f'VF Top CAPEX Items\n({vf_data["capex_total"]/1e6:.1f}M total)',
                 fontsize=11, fontweight='bold')
    
    # TF CAPEX
    tf_capex_top = sorted(tf_data['capex_items'].items(),
                         key=lambda x: x[1]['cost'], reverse=True)[:5]
    tf_labels = [item[0][:15] for item in tf_capex_top]
    tf_values = [item[1]['cost'] / 1e6 for item in tf_capex_top]
    
    colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(tf_labels)))
    ax3.pie(tf_values, labels=tf_labels, autopct='%1.0f%%', startangle=90,
           colors=colors, textprops={'fontsize': 8})
    ax3.set_title(f'TF Top CAPEX Items\n({tf_data["capex_total"]/1e6:.1f}M total)',
                 fontsize=11, fontweight='bold')
    
    # 3. OPEX breakdown comparison (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    
    # Get top 6 OPEX items by combined value
    all_opex_items = set(list(vf_data['opex_items'].keys()) + list(tf_data['opex_items'].keys()))
    combined_opex = [(item, 
                     vf_data['opex_items'].get(item, 0) + tf_data['opex_items'].get(item, 0))
                    for item in all_opex_items]
    top_opex = sorted(combined_opex, key=lambda x: x[1], reverse=True)[:6]
    
    opex_labels = [item[0][:20] for item in top_opex]
    vf_opex_vals = [vf_data['opex_items'].get(item[0], 0) for item in top_opex]
    tf_opex_vals = [tf_data['opex_items'].get(item[0], 0) for item in top_opex]
    
    x = np.arange(len(opex_labels))
    width = 0.35
    
    bars1 = ax4.barh(x + width/2, vf_opex_vals, width, label='VF',
                    color='#2E86AB', alpha=0.8)
    bars2 = ax4.barh(x - width/2, tf_opex_vals, width, label='TF',
                    color='#A23B72', alpha=0.8)
    
    ax4.set_yticks(x)
    ax4.set_yticklabels(opex_labels, fontsize=9)
    ax4.set_xlabel('Cost per kg (SGD/kg)', fontsize=10, fontweight='bold')
    ax4.set_title('Top OPEX Items Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='x', alpha=0.3)
    
    # 4. Profitability metrics (bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    
    metrics = ['Revenue', 'Cost', 'Profit', 'Margin %']
    vf_metrics = [vf_data['revenue_annual'] / 1e6,
                  vf_data['cost_annual'] / 1e6,
                  vf_data['profit_annual'] / 1e6,
                  vf_data['profit_margin']]
    tf_metrics = [tf_data['revenue_annual'] / 1e6,
                  tf_data['cost_annual'] / 1e6,
                  tf_data['profit_annual'] / 1e6,
                  tf_data['profit_margin']]
    
    # Normalize first 3 (monetary), keep margin as %
    max_val = max(max(vf_metrics[:3]), max(tf_metrics[:3]))
    vf_norm = vf_metrics[:3] + [vf_metrics[3] / 100 * max_val]
    tf_norm = tf_metrics[:3] + [tf_metrics[3] / 100 * max_val]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, vf_norm, width, label='VF',
                   color='#2E86AB', alpha=0.8)
    bars2 = ax5.bar(x + width/2, tf_norm, width, label='TF',
                   color='#A23B72', alpha=0.8)
    
    ax5.set_ylabel('Value (M SGD or normalized)', fontsize=10, fontweight='bold')
    ax5.set_title('Financial Performance Metrics', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, fontsize=9)
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    
    # Add actual values
    for i in range(len(metrics)):
        if i < 3:
            label = f'VF:{vf_metrics[i]:.1f}M\nTF:{tf_metrics[i]:.1f}M'
        else:
            label = f'VF:{vf_metrics[i]:.1f}%\nTF:{tf_metrics[i]:.1f}%'
        ax5.text(i, max(vf_norm[i], tf_norm[i]) + max_val * 0.05,
                label, ha='center', fontsize=7)
    
    # 5. Summary table (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    table_data = [
        ['Initial CAPEX', f'{vf_data["capex_total"]/1e6:.1f}M', f'{tf_data["capex_total"]/1e6:.1f}M'],
        ['Annual OPEX', f'{vf_data["opex_annual"]/1e6:.1f}M', f'{tf_data["opex_annual"]/1e6:.1f}M'],
        ['LCOv (per kg)', f'{vf_data["lcov"]:.2f}', f'{tf_data["lcov"]:.2f}'],
        ['Product Price', f'{vf_data["product_price"]:.2f}', f'{tf_data["product_price"]:.2f}'],
        ['Annual Output', f'{vf_data["annual_output"]/1e6:.1f}M kg', f'{tf_data["annual_output"]/1e6:.1f}M kg'],
        ['Annual Profit', f'{vf_data["profit_annual"]/1e6:.1f}M', f'{tf_data["profit_annual"]/1e6:.1f}M'],
        ['Profit Margin', f'{vf_data["profit_margin"]:.1f}%', f'{tf_data["profit_margin"]:.1f}%'],
    ]
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Metric', 'Vertical Farm', 'Traditional Farm'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code better values
    for i in range(len(table_data)):
        vf_val = float(table_data[i][1].replace('M', '').replace('%', '').replace(' kg', ''))
        tf_val = float(table_data[i][2].replace('M', '').replace('%', '').replace(' kg', ''))
        
        # Lower is better for CAPEX, OPEX, LCOv
        if i < 3:
            if vf_val < tf_val:
                table[(i+1, 1)].set_facecolor('#90EE90')
                table[(i+1, 2)].set_facecolor('#FFB6C6')
            else:
                table[(i+1, 1)].set_facecolor('#FFB6C6')
                table[(i+1, 2)].set_facecolor('#90EE90')
        # Higher is better for output, profit, margin
        elif i >= 4:
            if vf_val > tf_val:
                table[(i+1, 1)].set_facecolor('#90EE90')
                table[(i+1, 2)].set_facecolor('#FFB6C6')
            else:
                table[(i+1, 1)].set_facecolor('#FFB6C6')
                table[(i+1, 2)].set_facecolor('#90EE90')
    
    # Header
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('LCC Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('LIFE CYCLE COSTING (LCC) COMPARISON DASHBOARD',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {filename}")

# =============================================================================
# SECTION 4: REPORT GENERATOR
# =============================================================================

def generate_lcc_report(vf_data, tf_data, analyzer, filename):
    """Generate comprehensive LCC text report"""
    
    ratios = analyzer.calculate_cost_ratios()
    breakeven = analyzer.breakeven_analysis()
    scenarios = analyzer.scenario_analysis()
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LIFE CYCLE COSTING (LCC) ANALYSIS REPORT\n")
        f.write("Vertical Farming vs Traditional Greenhouse Farming\n")
        f.write("=" * 80 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        lower_lcov = "Vertical Farm" if vf_data['lcov'] < tf_data['lcov'] else "Traditional Farm"
        lcov_diff = abs(vf_data['lcov'] - tf_data['lcov'])
        lcov_pct = (lcov_diff / min(vf_data['lcov'], tf_data['lcov']) * 100)
        
        f.write(f"Lower Cost System: {lower_lcov}\n")
        f.write(f"LCOv Difference: {lcov_diff:.2f} SGD/kg ({lcov_pct:.1f}% advantage)\n\n")
        
        more_profitable = "Vertical Farm" if vf_data['profit_annual'] > tf_data['profit_annual'] else "Traditional Farm"
        profit_diff = abs(vf_data['profit_annual'] - tf_data['profit_annual']) / 1e6
        
        f.write(f"More Profitable System: {more_profitable}\n")
        f.write(f"Annual Profit Difference: {profit_diff:.1f}M SGD\n\n")
        
        # Detailed Cost Analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED COST ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for system_name, data in [("VERTICAL FARM", vf_data), ("TRADITIONAL FARM", tf_data)]:
            f.write(f"\n{system_name}:\n")
            f.write("-" * 80 + "\n\n")
            
            f.write(f"Initial CAPEX:               {data['capex_total']/1e6:>15.2f} Million SGD\n")
            f.write(f"Annual CAPEX (annualized):   {data['capex_annualized_total']/1e6:>15.2f} Million SGD\n")
            f.write(f"Annual OPEX:                 {data['opex_annual']/1e6:>15.2f} Million SGD\n")
            f.write(f"Total Annual Cost:           {data['cost_annual']/1e6:>15.2f} Million SGD\n\n")
            
            f.write(f"CAPEX per kg:                {data['capex_per_kg']:>15.3f} SGD/kg\n")
            f.write(f"OPEX per kg:                 {data['opex_per_kg']:>15.3f} SGD/kg\n")
            f.write(f"LCOv (Total):                {data['lcov']:>15.3f} SGD/kg\n\n")
            
            f.write(f"Product Price:               {data['product_price']:>15.2f} SGD/kg\n")
            f.write(f"Annual Output:               {data['annual_output']/1e6:>15.2f} Million kg\n")
            f.write(f"Annual Revenue:              {data['revenue_annual']/1e6:>15.2f} Million SGD\n")
            f.write(f"Annual Profit:               {data['profit_annual']/1e6:>15.2f} Million SGD\n")
            f.write(f"Profit Margin:               {data['profit_margin']:>15.1f} %\n\n")
        
        # CAPEX Breakdown
        f.write("\n" + "=" * 80 + "\n")
        f.write("CAPEX BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        
        for system_name, data in [("Vertical Farm", vf_data), ("Traditional Farm", tf_data)]:
            f.write(f"\n{system_name}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Asset':<40} {'Cost (M SGD)':>15} {'%':>10} {'Lifetime':>10}\n")
            f.write("-" * 80 + "\n")
            
            sorted_items = sorted(data['capex_items'].items(),
                                key=lambda x: x[1]['cost'], reverse=True)
            
            for asset, details in sorted_items:
                cost_m = details['cost'] / 1e6
                pct = (details['cost'] / data['capex_total'] * 100)
                lifetime = details['lifetime']
                f.write(f"{asset:<40} {cost_m:>15.2f} {pct:>10.1f} {lifetime:>10.0f}\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"{'TOTAL':<40} {data['capex_total']/1e6:>15.2f} {100:>10.1f}\n\n")
        
        # OPEX Breakdown
        f.write("\n" + "=" * 80 + "\n")
        f.write("OPEX BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        
        for system_name, data in [("Vertical Farm", vf_data), ("Traditional Farm", tf_data)]:
            f.write(f"\n{system_name}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Item':<40} {'SGD/kg':>15} {'%':>10} {'M SGD/yr':>15}\n")
            f.write("-" * 80 + "\n")
            
            sorted_items = sorted(data['opex_items'].items(),
                                key=lambda x: x[1], reverse=True)
            
            for item, cost_per_kg in sorted_items:
                pct = (cost_per_kg / data['opex_per_kg'] * 100)
                annual_m = (cost_per_kg * data['annual_output']) / 1e6
                f.write(f"{item:<40} {cost_per_kg:>15.3f} {pct:>10.1f} {annual_m:>15.2f}\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"{'TOTAL':<40} {data['opex_per_kg']:>15.3f} {100:>10.1f} "
                   f"{data['opex_annual']/1e6:>15.2f}\n\n")
        
        # Breakeven Analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("BREAKEVEN ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for system_name, be_data in [("Vertical Farm", breakeven['VF']), 
                                     ("Traditional Farm", breakeven['TF'])]:
            f.write(f"\n{system_name}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Breakeven Production:        {be_data['breakeven_production_kg']/1e6:>15.2f} Million kg/yr\n")
            f.write(f"Breakeven Price:             {be_data['breakeven_price_sgd']:>15.2f} SGD/kg\n")
            f.write(f"Capacity Utilization @BE:    {be_data['capacity_utilization_pct']:>15.1f} %\n\n")
        
        # Scenario Analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("SCENARIO ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Scenario':<20} {'System':<20} {'LCOv':>12} {'Profit (M)':>15}\n")
        f.write("-" * 80 + "\n")
        
        for scenario in ['optimistic', 'baseline', 'pessimistic']:
            scenario_name = scenario.capitalize()
            for system in ['VF', 'TF']:
                system_name = "Vertical Farm" if system == 'VF' else "Traditional Farm"
                lcov = scenarios[scenario][system]['lcov']
                profit = scenarios[scenario][system]['profit_annual'] / 1e6
                f.write(f"{scenario_name:<20} {system_name:<20} {lcov:>12.2f} {profit:>15.1f}\n")
            f.write("\n")
        
        # Cost Ratios
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMPARATIVE RATIOS (VF / TF)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"CAPEX Ratio:                 {ratios['capex_ratio']:>15.2f}x\n")
        f.write(f"OPEX per kg Ratio:           {ratios['opex_per_kg_ratio']:>15.2f}x\n")
        f.write(f"LCOv Ratio:                  {ratios['lcov_ratio']:>15.2f}x\n")
        f.write(f"Output Ratio:                {ratios['output_ratio']:>15.2f}x\n")
        f.write(f"Profit Ratio:                {ratios['profit_ratio']:>15.2f}x\n\n")
        
        # Recommendations
        f.write("\n" + "=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        if vf_data['lcov'] < tf_data['lcov']:
            f.write("Based on LCOv: VERTICAL FARM is more cost-effective per kg\n\n")
        else:
            f.write("Based on LCOv: TRADITIONAL FARM is more cost-effective per kg\n\n")
        
        if vf_data['profit_annual'] > tf_data['profit_annual']:
            f.write("Based on Annual Profit: VERTICAL FARM is more profitable overall\n\n")
        else:
            f.write("Based on Annual Profit: TRADITIONAL FARM is more profitable overall\n\n")
        
        f.write("Key Considerations:\n")
        f.write(f"• VF requires {ratios['capex_ratio']:.1f}x higher initial investment\n")
        f.write(f"• VF produces {ratios['output_ratio']:.1f}x more output annually\n")
        f.write(f"• VF operating costs are {ratios['opex_per_kg_ratio']:.1f}x TF costs per kg\n")
        f.write(f"• Under pessimistic scenario, VF profit: {scenarios['pessimistic']['VF']['profit_annual']/1e6:.1f}M\n")
        f.write(f"• Under pessimistic scenario, TF profit: {scenarios['pessimistic']['TF']['profit_annual']/1e6:.1f}M\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

# =============================================================================
# SECTION 5: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("COMPREHENSIVE LIFE CYCLE COSTING (LCC) ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    print("Step 1: Loading Data...")
    print("-" * 80)
    
    loader = LCCDataLoader()
    vf_data, tf_data = loader.load_all_data()
    
    print(f"✓ Loaded {vf_data['name']} data")
    print(f"  Initial CAPEX: {vf_data['capex_total']/1e6:.2f}M SGD")
    print(f"  Annual OPEX: {vf_data['opex_annual']/1e6:.2f}M SGD")
    print(f"  LCOv: {vf_data['lcov']:.3f} SGD/kg")
    
    print(f"\n✓ Loaded {tf_data['name']} data")
    print(f"  Initial CAPEX: {tf_data['capex_total']/1e6:.2f}M SGD")
    print(f"  Annual OPEX: {tf_data['opex_annual']/1e6:.2f}M SGD")
    print(f"  LCOv: {tf_data['lcov']:.3f} SGD/kg")
    print()
    
    # Perform analysis
    print("Step 2: Performing LCC Analysis...")
    print("-" * 80)
    
    analyzer = LCCAnalyzer(vf_data, tf_data)
    
    ratios = analyzer.calculate_cost_ratios()
    print(f"✓ Calculated cost ratios")
    print(f"  LCOv ratio (VF/TF): {ratios['lcov_ratio']:.2f}x")
    
    breakeven = analyzer.breakeven_analysis()
    print(f"\n✓ Performed breakeven analysis")
    print(f"  VF breakeven production: {breakeven['VF']['breakeven_production_kg']/1e6:.2f}M kg")
    print(f"  TF breakeven production: {breakeven['TF']['breakeven_production_kg']/1e6:.2f}M kg")
    
    scenarios = analyzer.scenario_analysis()
    print(f"\n✓ Completed scenario analysis")
    print(f"  Optimistic VF LCOv: {scenarios['optimistic']['VF']['lcov']:.3f} SGD/kg")
    print(f"  Pessimistic VF LCOv: {scenarios['pessimistic']['VF']['lcov']:.3f} SGD/kg")
    print()
    
    # Generate visualizations
    print("Step 3: Generating Visualizations...")
    print("-" * 80)
    
    plot_capex_breakdown(vf_data, tf_data,
                        os.path.join(OUTPUT_DIR, 'LCC_CAPEX_Breakdown.png'))
    
    plot_opex_breakdown(vf_data, tf_data,
                       os.path.join(OUTPUT_DIR, 'LCC_OPEX_Breakdown.png'))
    
    plot_lcov_comparison(vf_data, tf_data,
                        os.path.join(OUTPUT_DIR, 'LCC_LCOv_Comparison.png'))
    
    plot_breakeven_analysis(analyzer,
                          os.path.join(OUTPUT_DIR, 'LCC_Breakeven_Analysis.png'))
    
    plot_scenario_analysis(analyzer,
                         os.path.join(OUTPUT_DIR, 'LCC_Scenario_Analysis.png'))
    
    plot_cost_comparison_dashboard(vf_data, tf_data,
                                  os.path.join(OUTPUT_DIR, 'LCC_Comparison_Dashboard.png'))
    
    print()
    
    # Generate report
    print("Step 4: Generating Report...")
    print("-" * 80)
    
    report_path = os.path.join(OUTPUT_DIR, 'LCC_Analysis_Report.txt')
    generate_lcc_report(vf_data, tf_data, analyzer, report_path)
    print(f"✓ Report saved: {report_path}")
    print()
    
    print("=" * 80)
    print("LCC ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print(f"Results saved in '{OUTPUT_DIR}' folder:")
    print("  1. LCC_CAPEX_Breakdown.png")
    print("  2. LCC_OPEX_Breakdown.png")
    print("  3. LCC_LCOv_Comparison.png")
    print("  4. LCC_Breakeven_Analysis.png")
    print("  5. LCC_Scenario_Analysis.png")
    print("  6. LCC_Comparison_Dashboard.png")
    print("  7. LCC_Analysis_Report.txt")
    print()
    
    # Print summary
    print("SUMMARY:")
    print("-" * 80)
    if vf_data['lcov'] < tf_data['lcov']:
        print(f"✓ Vertical Farm has LOWER LCOv ({vf_data['lcov']:.3f} vs {tf_data['lcov']:.3f})")
    else:
        print(f"✓ Traditional Farm has LOWER LCOv ({tf_data['lcov']:.3f} vs {vf_data['lcov']:.3f})")
    
    if vf_data['profit_annual'] > tf_data['profit_annual']:
        print(f"✓ Vertical Farm has HIGHER profit ({vf_data['profit_annual']/1e6:.1f}M vs {tf_data['profit_annual']/1e6:.1f}M)")
    else:
        print(f"✓ Traditional Farm has HIGHER profit ({tf_data['profit_annual']/1e6:.1f}M vs {vf_data['profit_annual']/1e6:.1f}M)")
    
    print()

if __name__ == "__main__":
    main()