"""
COMPLETE ANALYSIS: Vertical Farming Sustainability Assessment for Singapore
=============================================================================

This script performs:
1. Life Cycle Costing (LCC) analysis
2. Life Cycle Impact Assessment (LCIA)
3. Analytic Hierarchy Process (AHP) multi-criteria decision analysis
4. Final report generation

Research Question: Is VF a sustainable and applicable food strategy for Singapore?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Create outputs directory
OUTPUT_DIR = 'analysis_outputs'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# SECTION 1: LIFE CYCLE COSTING (LCC) CALCULATOR
# =============================================================================

class LCCAnalysis:
    """Life Cycle Costing Analysis for farming systems"""
    
    def __init__(self, excel_file, param_sheet, capex_sheet, opex_sheet):
        """Load data from Excel"""
        # Skip first 2 rows (title and blank), read from row 3 where headers are
        self.params = pd.read_excel(excel_file, sheet_name=param_sheet, header=2)
        self.capex = pd.read_excel(excel_file, sheet_name=capex_sheet, header=2)
        self.opex = pd.read_excel(excel_file, sheet_name=opex_sheet, header=2)
        
    def get_param(self, param_name):
        """Get parameter value by name"""
        # Filter out NaN rows
        valid_params = self.params[self.params['Parameter'].notna()]
        row = valid_params[valid_params['Parameter'] == param_name]
        if len(row) > 0:
            return row['Value'].values[0]
        print(f"Warning: Parameter '{param_name}' not found")
        return None
    
    def calculate_annualized_capex(self):
        """Calculate annualized CAPEX using capital recovery factor"""
        discount_rate = self.get_param('Discount rate r (decimal)')
        
        annualized_costs = []
        for _, row in self.capex.iterrows():
            if pd.notna(row['Cost_SGD']) and row['Cost_SGD'] > 0:
                cost = row['Cost_SGD']
                lifetime = row['Lifetime_years']
                # Capital Recovery Factor: r(1+r)^n / ((1+r)^n - 1)
                crf = (discount_rate * (1 + discount_rate)**lifetime) / \
                      ((1 + discount_rate)**lifetime - 1)
                annualized = cost * crf
                annualized_costs.append(annualized)
        
        return sum(annualized_costs)
    
    def calculate_opex_per_kg(self):
        """Calculate total OPEX per kg"""
        # Check if there's a TOTAL row
        total_row = self.opex[self.opex['Item'] == 'TOTAL per kg']
        if len(total_row) > 0 and pd.notna(total_row['Cost_per_kg_SGD'].values[0]):
            return total_row['Cost_per_kg_SGD'].values[0]
        # If no total row, sum all valid costs (excluding NaN and TOTAL row)
        valid_costs = self.opex[
            (self.opex['Item'].notna()) & 
            (self.opex['Item'] != 'TOTAL per kg') &
            (self.opex['Cost_per_kg_SGD'].notna())
        ]
        return valid_costs['Cost_per_kg_SGD'].sum()
    
    def calculate_lcc(self):
        """Calculate complete LCC metrics"""
        annual_output = self.get_param('Annual edible output (kg/yr)')
        product_price = self.get_param('Product price (SGD/kg)')
        
        # CAPEX
        annual_capex = self.calculate_annualized_capex()
        capex_per_kg = annual_capex / annual_output
        
        # OPEX
        opex_per_kg = self.calculate_opex_per_kg()
        annual_opex = opex_per_kg * annual_output
        
        # LCOv (Levelized Cost of Vegetables)
        lcov = capex_per_kg + opex_per_kg
        
        # Financial metrics
        annual_revenue = product_price * annual_output
        annual_cost = annual_capex + annual_opex
        net_cashflow = annual_revenue - annual_cost
        profit_margin = (net_cashflow / annual_revenue * 100) if annual_revenue > 0 else 0
        
        return {
            'annual_output': annual_output,
            'capex_per_kg': capex_per_kg,
            'opex_per_kg': opex_per_kg,
            'lcov': lcov,
            'annual_capex': annual_capex,
            'annual_opex': annual_opex,
            'annual_revenue': annual_revenue,
            'annual_cost': annual_cost,
            'net_cashflow': net_cashflow,
            'profit_margin': profit_margin,
            'product_price': product_price
        }

# =============================================================================
# SECTION 2: LIFE CYCLE IMPACT ASSESSMENT (LCIA) CALCULATOR
# =============================================================================

class LCIAAnalysis:
    """Life Cycle Impact Assessment using environmental factors"""
    
    # Environmental impact factors (per unit)
    IMPACT_FACTORS = {
        # Category: {impact_type: factor}
        'Electricity (kWh)': {
            'GWP100': 0.408,      # kg CO2-eq/kWh (Singapore grid 2024)
            'HOFP': 0.00028,      # kg NOx-eq/kWh
            'PMFP': 0.00018,      # kg PM2.5-eq/kWh
            'AP': 0.00028,        # kg SO2-eq/kWh
            'EOFP': 0.0001,       # kg NOx-eq/kWh
            'FFP': 7.6            # MJ/kWh
        },
        'Water (m3)': {
            'GWP100': 0.344,      # kg CO2-eq/m3
            'HOFP': 0.00005,
            'PMFP': 0.00001,
            'AP': 0.00005,
            'EOFP': 0.00001,
            'FFP': 2.5
        },
        'Wastewater (m3)': {
            'GWP100': 0.472,      # kg CO2-eq/m3
            'HOFP': 0.0001,
            'PMFP': 0.00005,
            'AP': 0.0001,
            'EOFP': 0.0001,
            'FFP': 1.5
        },
        'Fertilizer (kg)': {
            'GWP100': 4.0,        # kg CO2-eq/kg NPK
            'HOFP': 0.001,
            'PMFP': 0.0015,
            'AP': 0.009,
            'EOFP': 0.003,
            'FFP': 60.0
        },
        'Pesticide (kg)': {
            'GWP100': 25.0,       # kg CO2-eq/kg active ingredient
            'HOFP': 0.02,
            'PMFP': 0.01,
            'AP': 0.03,
            'EOFP': 0.01,
            'FFP': 100.0
        },
        'Packaging (kg)': {
            'GWP100': 2.7,        # kg CO2-eq/kg plastic
            'HOFP': 0.001,
            'PMFP': 0.001,
            'AP': 0.004,
            'EOFP': 0.001,
            'FFP': 70.0
        },
        'Transport (km)': {
            'GWP100': 0.18,       # kg CO2-eq/km (refrigerated van)
            'HOFP': 0.0006,
            'PMFP': 0.0005,
            'AP': 0.0009,
            'EOFP': 0.0001,
            'FFP': 3.0
        }
    }
    
    def __init__(self, excel_file, param_sheet):
        """Initialize with parameter data"""
        # Skip first 2 rows (title and blank), read from row 3 where headers are
        self.params = pd.read_excel(excel_file, sheet_name=param_sheet, header=2)
        
    def get_param(self, param_name):
        """Get parameter value"""
        # Filter out NaN rows
        valid_params = self.params[self.params['Parameter'].notna()]
        row = valid_params[valid_params['Parameter'] == param_name]
        if len(row) > 0:
            return row['Value'].values[0]
        return 0  # Return 0 if not found (for optional parameters)
    
    def calculate_impacts_per_kg(self, system_type='VF'):
        """Calculate environmental impacts per kg produce"""
        
        if system_type == 'VF':
            # Vertical Farm
            electricity = self.get_param('Lighting kWh per kg') + \
                         self.get_param('HVAC & other kWh per kg')
            water = self.get_param('Water m3 per kg')
            wastewater = water  # Assume same as water input
            fertilizer = self.get_param('Fertilizer intensity (kg/kg produce)')
            pesticide = self.get_param('Pesticide intensity (kg/kg produce)')
            packaging = 0.08  # Assume 80g plastic per kg
            transport_km = self.get_param('Distribution distance (km)')
        else:
            # Traditional Farm
            electricity = self.get_param('Electricity use (kWh/kg)')
            water = self.get_param('Water m3 per kg')
            wastewater = water * 0.75  # Not all water becomes wastewater
            fertilizer = self.get_param('Fertilizer use (kg/kg)')
            pesticide = self.get_param('Pesticide use (kg/kg)')
            packaging = 0.05  # Lighter packaging
            transport_km = self.get_param('Distribution distance (km)')
        
        # Calculate impacts
        impacts = {}
        for impact_type in ['GWP100', 'HOFP', 'PMFP', 'AP', 'EOFP', 'FFP']:
            total = 0
            total += electricity * self.IMPACT_FACTORS['Electricity (kWh)'][impact_type]
            total += water * self.IMPACT_FACTORS['Water (m3)'][impact_type]
            total += wastewater * self.IMPACT_FACTORS['Wastewater (m3)'][impact_type]
            total += fertilizer * self.IMPACT_FACTORS['Fertilizer (kg)'][impact_type]
            total += pesticide * self.IMPACT_FACTORS['Pesticide (kg)'][impact_type]
            total += packaging * self.IMPACT_FACTORS['Packaging (kg)'][impact_type]
            total += transport_km * self.IMPACT_FACTORS['Transport (km)'][impact_type]
            impacts[impact_type] = total
        
        # Add inventory data
        impacts['inventory'] = {
            'electricity_kwh': electricity,
            'water_m3': water,
            'wastewater_m3': wastewater,
            'fertilizer_kg': fertilizer,
            'pesticide_kg': pesticide,
            'packaging_kg': packaging,
            'transport_km': transport_km
        }
        
        return impacts

# =============================================================================
# SECTION 3: AHP ANALYSIS (using previous code)
# =============================================================================

class AHPAnalysis:
    """Analytic Hierarchy Process for multi-criteria decision making"""
    
    def __init__(self, alternatives):
        self.alternatives = alternatives
        self.n_alternatives = len(alternatives)
        self.criteria_weights = {}
        self.subcriteria_weights = {}
        self.alternative_scores = {}
        self.consistency_ratios = {}
        
    @staticmethod
    def calculate_weights(pairwise_matrix):
        """Calculate weights using eigenvalue method"""
        n = pairwise_matrix.shape[0]
        eigenvalues, eigenvectors = linalg.eig(pairwise_matrix)
        max_idx = np.argmax(eigenvalues.real)
        max_eigenvalue = eigenvalues[max_idx].real
        principal_eigenvector = eigenvectors[:, max_idx].real
        weights = principal_eigenvector / principal_eigenvector.sum()
        
        CI = (max_eigenvalue - n) / (n - 1) if n > 1 else 0
        RI_values = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        RI = RI_values.get(n, 1.49)
        CR = CI / RI if RI > 0 else 0
        
        return weights, CR, max_eigenvalue
    
    @staticmethod
    def create_pairwise_matrix_from_dict(comparisons, items):
        """Create pairwise matrix from comparison dictionary"""
        n = len(items)
        matrix = np.ones((n, n))
        for (i, j), value in comparisons.items():
            matrix[i, j] = value
            matrix[j, i] = 1 / value
        return matrix
    
    def add_criteria_comparisons(self, criteria_names, pairwise_comparisons):
        """Add main criteria comparisons"""
        matrix = self.create_pairwise_matrix_from_dict(pairwise_comparisons, criteria_names)
        weights, cr, eigenval = self.calculate_weights(matrix)
        self.criteria_names = criteria_names
        self.criteria_weights = dict(zip(criteria_names, weights))
        self.consistency_ratios['Criteria'] = cr
        
    def add_subcriteria_comparisons(self, criterion, subcriteria_names, pairwise_comparisons):
        """Add sub-criteria comparisons"""
        matrix = self.create_pairwise_matrix_from_dict(pairwise_comparisons, subcriteria_names)
        weights, cr, eigenval = self.calculate_weights(matrix)
        if criterion not in self.subcriteria_weights:
            self.subcriteria_weights[criterion] = {}
        self.subcriteria_weights[criterion] = dict(zip(subcriteria_names, weights))
        self.consistency_ratios[f'Subcriteria_{criterion}'] = cr
        
    def add_alternative_scores_for_subcriterion(self, criterion, subcriterion, pairwise_comparisons):
        """Add alternative comparisons"""
        matrix = self.create_pairwise_matrix_from_dict(pairwise_comparisons, self.alternatives)
        weights, cr, eigenval = self.calculate_weights(matrix)
        self.alternative_scores[subcriterion] = dict(zip(self.alternatives, weights))
        self.consistency_ratios[f'Alternatives_{subcriterion}'] = cr
        
    def calculate_final_scores(self):
        """Calculate final weighted scores"""
        final_scores = {alt: 0 for alt in self.alternatives}
        detailed_scores = {alt: {} for alt in self.alternatives}
        
        for criterion in self.criteria_names:
            criterion_weight = self.criteria_weights[criterion]
            if criterion in self.subcriteria_weights:
                for subcriterion, sub_weight in self.subcriteria_weights[criterion].items():
                    global_weight = criterion_weight * sub_weight
                    if subcriterion in self.alternative_scores:
                        for alt in self.alternatives:
                            alt_score = self.alternative_scores[subcriterion][alt]
                            contribution = global_weight * alt_score
                            final_scores[alt] += contribution
                            detailed_scores[alt][subcriterion] = contribution
        
        self.final_scores = final_scores
        self.detailed_scores = detailed_scores
        return final_scores

def generate_comparison_from_data(vf_value, tf_value, higher_is_better=True):
    """Generate AHP comparison from data"""
    if vf_value == 0 or tf_value == 0:
        return 1
    ratio = vf_value / tf_value if higher_is_better else tf_value / vf_value
    
    if ratio >= 9: return 9
    elif ratio >= 7: return 7
    elif ratio >= 5: return 5
    elif ratio >= 3: return 3
    elif ratio >= 1.5: return 2
    elif ratio >= 1.1: return 1
    elif ratio >= 1/1.1: return 1
    elif ratio >= 1/1.5: return 1/2
    elif ratio >= 1/3: return 1/3
    elif ratio >= 1/5: return 1/5
    elif ratio >= 1/7: return 1/7
    else: return 1/9

# =============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# =============================================================================

def create_lcc_visualizations(lcc_vf, lcc_tf, output_dir):
    """Generate LCC comparison visualizations"""
    
    print("Generating LCC visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Cost Breakdown Comparison (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    categories = ['CAPEX\nper kg', 'OPEX\nper kg', 'Total\nLCOv']
    vf_costs = [lcc_vf['capex_per_kg'], lcc_vf['opex_per_kg'], lcc_vf['lcov']]
    tf_costs = [lcc_tf['capex_per_kg'], lcc_tf['opex_per_kg'], lcc_tf['lcov']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, vf_costs, width, label='Vertical Farm', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, tf_costs, width, label='Traditional Farm', color='#A23B72', alpha=0.8)
    
    ax1.set_ylabel('Cost (SGD/kg)', fontsize=11, fontweight='bold')
    ax1.set_title('Cost Breakdown Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Cost Structure - VF (Top Middle)
    ax2 = plt.subplot(2, 3, 2)
    vf_structure = [lcc_vf['capex_per_kg'], lcc_vf['opex_per_kg']]
    colors_pie = ['#4472C4', '#ED7D31']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax2.pie(vf_structure, labels=['CAPEX', 'OPEX'], autopct='%1.1f%%',
                                        colors=colors_pie, explode=explode, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    ax2.set_title('Vertical Farm Cost Structure', fontsize=13, fontweight='bold')
    
    # 3. Cost Structure - TF (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    tf_structure = [lcc_tf['capex_per_kg'], lcc_tf['opex_per_kg']]
    
    wedges, texts, autotexts = ax3.pie(tf_structure, labels=['CAPEX', 'OPEX'], autopct='%1.1f%%',
                                        colors=colors_pie, explode=explode, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    ax3.set_title('Traditional Farm Cost Structure', fontsize=13, fontweight='bold')
    
    # 4. Profitability Comparison (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    metrics = ['Profit\nMargin (%)', 'Net Cashflow\n(M SGD/yr)']
    vf_profit = [lcc_vf['profit_margin'], lcc_vf['net_cashflow']/1e6]
    tf_profit = [lcc_tf['profit_margin'], lcc_tf['net_cashflow']/1e6]
    
    x = np.arange(len(metrics))
    bars1 = ax4.bar(x - width/2, vf_profit, width, label='Vertical Farm', color='#2E86AB', alpha=0.8)
    bars2 = ax4.bar(x + width/2, tf_profit, width, label='Traditional Farm', color='#A23B72', alpha=0.8)
    
    ax4.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax4.set_title('Profitability Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Production Capacity (Bottom Middle)
    ax5 = plt.subplot(2, 3, 5)
    farms = ['Vertical\nFarm', 'Traditional\nFarm']
    outputs = [lcc_vf['annual_output']/1e6, lcc_tf['annual_output']/1e6]
    colors_bar = ['#2E86AB', '#A23B72']
    
    bars = ax5.bar(farms, outputs, color=colors_bar, alpha=0.8, width=0.6)
    ax5.set_ylabel('Annual Output (Million kg/yr)', fontsize=11, fontweight='bold')
    ax5.set_title('Production Capacity', fontsize=13, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Price per kg (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    prices = [lcc_vf['product_price'], lcc_tf['product_price']]
    lcovs = [lcc_vf['lcov'], lcc_tf['lcov']]
    
    x = np.arange(len(farms))
    width_price = 0.35
    
    bars1 = ax6.bar(x - width_price/2, prices, width_price, label='Product Price', color='#70AD47', alpha=0.8)
    bars2 = ax6.bar(x + width_price/2, lcovs, width_price, label='Cost (LCOv)', color='#C00000', alpha=0.8)
    
    ax6.set_ylabel('SGD per kg', fontsize=11, fontweight='bold')
    ax6.set_title('Price vs Cost Comparison', fontsize=13, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(farms)
    ax6.legend(fontsize=9)
    ax6.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('LIFE CYCLE COSTING (LCC) ANALYSIS', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(output_dir, 'LCC_Analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_lcia_visualizations(lcia_vf, lcia_tf, output_dir):
    """Generate LCIA comparison visualizations"""
    
    print("Generating LCIA visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    
    impact_categories = {
        'GWP100': 'Climate Change\n(kg CO2-eq)',
        'HOFP': 'Human Toxicity\n(kg NOx-eq)',
        'PMFP': 'Particulate Matter\n(kg PM2.5-eq)',
        'AP': 'Acidification\n(kg SO2-eq)',
        'EOFP': 'Eutrophication\n(kg NOx-eq)',
        'FFP': 'Fossil Fuel\n(MJ)'
    }
    
    # 1. Environmental Impact Comparison (Top - spans 2 columns)
    ax1 = plt.subplot(2, 3, (1, 2))
    
    categories = list(impact_categories.values())
    vf_impacts = [lcia_vf[key] for key in impact_categories.keys()]
    tf_impacts = [lcia_tf[key] for key in impact_categories.keys()]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, vf_impacts, width, label='Vertical Farm', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, tf_impacts, width, label='Traditional Farm', color='#A23B72', alpha=0.8)
    
    ax1.set_ylabel('Impact per kg produce', fontsize=11, fontweight='bold')
    ax1.set_title('Environmental Impact Categories (per kg)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visibility
    
    # Highlight winner for each category
    for i, (vf_val, tf_val) in enumerate(zip(vf_impacts, tf_impacts)):
        if vf_val < tf_val:
            ax1.text(i, max(vf_val, tf_val) * 1.5, '✓', ha='center', 
                    fontsize=14, color='#2E86AB', fontweight='bold')
        else:
            ax1.text(i, max(vf_val, tf_val) * 1.5, '✓', ha='center',
                    fontsize=14, color='#A23B72', fontweight='bold')
    
    # 2. Normalized Environmental Performance (Top Right)
    ax2 = plt.subplot(2, 3, 3)
    
    # Normalize impacts (lower is better, so invert)
    vf_normalized = []
    tf_normalized = []
    for key in impact_categories.keys():
        total = lcia_vf[key] + lcia_tf[key]
        if total > 0:
            vf_normalized.append((lcia_tf[key] / total) * 100)  # Inverted - higher is better
            tf_normalized.append((lcia_vf[key] / total) * 100)
        else:
            vf_normalized.append(50)
            tf_normalized.append(50)
    
    vf_avg = np.mean(vf_normalized)
    tf_avg = np.mean(tf_normalized)
    
    bars = ax2.bar(['Vertical\nFarm', 'Traditional\nFarm'], [vf_avg, tf_avg],
                   color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.5)
    ax2.set_ylabel('Environmental Score (%)\n(Higher is Better)', fontsize=10, fontweight='bold')
    ax2.set_title('Overall Environmental\nPerformance', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Resource Use Comparison (Bottom Left)
    ax3 = plt.subplot(2, 3, 4)
    
    vf_inv = lcia_vf['inventory']
    tf_inv = lcia_tf['inventory']
    
    resources = ['Electricity\n(kWh)', 'Water\n(liters)', 'Fertilizer\n(grams)', 'Pesticide\n(mg)']
    vf_resources = [
        vf_inv['electricity_kwh'],
        vf_inv['water_m3'] * 1000,
        vf_inv['fertilizer_kg'] * 1000,
        vf_inv['pesticide_kg'] * 1e6
    ]
    tf_resources = [
        tf_inv['electricity_kwh'],
        tf_inv['water_m3'] * 1000,
        tf_inv['fertilizer_kg'] * 1000,
        tf_inv['pesticide_kg'] * 1e6
    ]
    
    x = np.arange(len(resources))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, vf_resources, width, label='Vertical Farm', color='#2E86AB', alpha=0.8)
    bars2 = ax3.bar(x + width/2, tf_resources, width, label='Traditional Farm', color='#A23B72', alpha=0.8)
    
    ax3.set_ylabel('Resource Use per kg', fontsize=11, fontweight='bold')
    ax3.set_title('Resource Consumption', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(resources, fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Water Efficiency (Bottom Middle)
    ax4 = plt.subplot(2, 3, 5)
    
    water_data = [vf_inv['water_m3'] * 1000, tf_inv['water_m3'] * 1000]
    bars = ax4.bar(['Vertical\nFarm', 'Traditional\nFarm'], water_data,
                   color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.5)
    ax4.set_ylabel('Water Use (liters per kg)', fontsize=11, fontweight='bold')
    ax4.set_title('Water Efficiency Comparison', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}L', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add water savings annotation
    water_saving = (1 - vf_inv['water_m3'] / tf_inv['water_m3']) * 100
    ax4.text(0.5, max(water_data) * 0.5, f'{water_saving:.0f}% less water',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 5. Carbon Footprint (Bottom Right)
    ax5 = plt.subplot(2, 3, 6)
    
    ghg_data = [lcia_vf['GWP100'], lcia_tf['GWP100']]
    bars = ax5.bar(['Vertical\nFarm', 'Traditional\nFarm'], ghg_data,
                   color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.5)
    ax5.set_ylabel('GHG Emissions (kg CO2-eq per kg)', fontsize=10, fontweight='bold')
    ax5.set_title('Carbon Footprint', fontsize=13, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add emission reduction annotation
    ghg_reduction = (1 - lcia_vf['GWP100'] / lcia_tf['GWP100']) * 100
    if ghg_reduction > 0:
        ax5.text(0.5, max(ghg_data) * 0.5, f'{ghg_reduction:.0f}% lower\nemissions',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('LIFE CYCLE IMPACT ASSESSMENT (LCIA)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(output_dir, 'LCIA_Analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # Create radar chart separately
    create_radar_chart(lcia_vf, lcia_tf, output_dir)


def create_radar_chart(lcia_vf, lcia_tf, output_dir):
    """Create radar chart for environmental impacts"""
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    categories = ['Climate\nChange', 'Human\nToxicity', 'Particulate\nMatter', 
                  'Acidification', 'Eutrophication', 'Fossil\nFuel']
    
    # Normalize values (0-1 scale, where 1 is worst)
    keys = ['GWP100', 'HOFP', 'PMFP', 'AP', 'EOFP', 'FFP']
    vf_values = []
    tf_values = []
    
    for key in keys:
        max_val = max(lcia_vf[key], lcia_tf[key])
        if max_val > 0:
            vf_values.append(lcia_vf[key] / max_val)
            tf_values.append(lcia_tf[key] / max_val)
        else:
            vf_values.append(0)
            tf_values.append(0)
    
    # Complete the circle
    vf_values += vf_values[:1]
    tf_values += tf_values[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, vf_values, 'o-', linewidth=2.5, label='Vertical Farm', color='#2E86AB', markersize=8)
    ax.fill(angles, vf_values, alpha=0.25, color='#2E86AB')
    ax.plot(angles, tf_values, 's-', linewidth=2.5, label='Traditional Farm', color='#A23B72', markersize=8)
    ax.fill(angles, tf_values, alpha=0.25, color='#A23B72')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
    ax.grid(True, linewidth=1, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=0.9)
    
    plt.title('Environmental Impact Radar Chart\n(Lower values = Better performance)', 
              size=14, fontweight='bold', pad=30)
    
    output_path = os.path.join(output_dir, 'LCIA_Radar_Chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_ahp_visualizations(ahp, output_dir):
    """Generate AHP analysis visualizations"""
    
    print("Generating AHP visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Final Scores (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    alternatives = list(ahp.final_scores.keys())
    scores = list(ahp.final_scores.values())
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax1.barh(alternatives, scores, color=colors, alpha=0.8)
    ax1.set_xlabel('AHP Score', fontsize=11, fontweight='bold')
    ax1.set_title('Final AHP Scores', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, max(scores) * 1.2)
    ax1.grid(axis='x', alpha=0.3)
    
    for bar, score in zip(bars, scores):
        ax1.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.4f} ({score*100:.1f}%)', va='center', fontsize=11, fontweight='bold')
    
    # 2. Criteria Weights (Top Middle)
    ax2 = plt.subplot(2, 3, 2)
    criteria = list(ahp.criteria_weights.keys())
    weights = [ahp.criteria_weights[c] for c in criteria]
    colors_pie = ['#4472C4', '#70AD47', '#FFC000']
    
    wedges, texts, autotexts = ax2.pie(weights, labels=criteria, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90, explode=(0.05, 0.05, 0.05))
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    ax2.set_title('Main Criteria Weights', fontsize=13, fontweight='bold')
    
    # 3. Consistency Ratios (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    
    cr_labels = list(ahp.consistency_ratios.keys())
    cr_values = list(ahp.consistency_ratios.values())
    colors_cr = ['green' if cr < 0.1 else 'orange' if cr < 0.2 else 'red' for cr in cr_values]
    
    bars = ax3.barh(cr_labels, cr_values, color=colors_cr, alpha=0.8)
    ax3.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='Threshold (0.1)')
    ax3.set_xlabel('Consistency Ratio (CR)', fontsize=11, fontweight='bold')
    ax3.set_title('Consistency Check\n(CR < 0.1 is acceptable)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='x', alpha=0.3)
    
    for bar, cr in zip(bars, cr_values):
        ax3.text(cr + 0.005, bar.get_y() + bar.get_height()/2,
                f'{cr:.4f}', va='center', fontsize=9, fontweight='bold')
    
    # 4. Economic Sub-criteria (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    if 'Economic' in ahp.subcriteria_weights:
        econ_sub = ahp.subcriteria_weights['Economic']
        econ_names = list(econ_sub.keys())
        econ_weights = [econ_sub[name] * ahp.criteria_weights['Economic'] for name in econ_names]
        
        bars = ax4.bar(range(len(econ_names)), econ_weights, color='#4472C4', alpha=0.8)
        ax4.set_ylabel('Global Weight', fontsize=11, fontweight='bold')
        ax4.set_title('Economic Sub-criteria\n(Global Weights)', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(econ_names)))
        ax4.set_xticklabels(econ_names, rotation=45, ha='right', fontsize=9)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, weight in zip(bars, econ_weights):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Environmental Sub-criteria (Bottom Middle)
    ax5 = plt.subplot(2, 3, 5)
    if 'Environmental' in ahp.subcriteria_weights:
        env_sub = ahp.subcriteria_weights['Environmental']
        env_names = list(env_sub.keys())
        env_weights = [env_sub[name] * ahp.criteria_weights['Environmental'] for name in env_names]
        
        bars = ax5.bar(range(len(env_names)), env_weights, color='#70AD47', alpha=0.8)
        ax5.set_ylabel('Global Weight', fontsize=11, fontweight='bold')
        ax5.set_title('Environmental Sub-criteria\n(Global Weights)', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(len(env_names)))
        ax5.set_xticklabels(env_names, rotation=45, ha='right', fontsize=9)
        ax5.grid(axis='y', alpha=0.3)
        
        for bar, weight in zip(bars, env_weights):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Contribution Breakdown (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    
    # Get top 8 contributors for each alternative
    vf_contrib = ahp.detailed_scores['Vertical Farm']
    tf_contrib = ahp.detailed_scores['Traditional Farm']
    
    # Sort by total contribution
    all_criteria = set(vf_contrib.keys()) | set(tf_contrib.keys())
    total_contrib = {c: vf_contrib.get(c, 0) + tf_contrib.get(c, 0) for c in all_criteria}
    top_criteria = sorted(total_contrib.items(), key=lambda x: x[1], reverse=True)[:8]
    top_names = [c[0] for c in top_criteria]
    
    vf_top = [vf_contrib.get(name, 0) for name in top_names]
    tf_top = [tf_contrib.get(name, 0) for name in top_names]
    
    y = np.arange(len(top_names))
    width = 0.35
    
    bars1 = ax6.barh(y + width/2, vf_top, width, label='Vertical Farm', color='#2E86AB', alpha=0.8)
    bars2 = ax6.barh(y - width/2, tf_top, width, label='Traditional Farm', color='#A23B72', alpha=0.8)
    
    ax6.set_yticks(y)
    ax6.set_yticklabels([name[:20] for name in top_names], fontsize=9)
    ax6.set_xlabel('Contribution to Final Score', fontsize=10, fontweight='bold')
    ax6.set_title('Top Contributing Criteria', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(axis='x', alpha=0.3)
    
    plt.suptitle('ANALYTIC HIERARCHY PROCESS (AHP) RESULTS', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = os.path.join(output_dir, 'AHP_Analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


# =============================================================================
# SECTION 5: REPORT GENERATOR
# =============================================================================

def generate_comprehensive_report(lcc_vf, lcc_tf, lcia_vf, lcia_tf, ahp, filename):
    """Generate comprehensive analysis report"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("Vertical Farming Sustainability Assessment for Singapore\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("=" * 80 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        winner = max(ahp.final_scores, key=ahp.final_scores.get)
        winner_score = ahp.final_scores[winner]
        loser = min(ahp.final_scores, key=ahp.final_scores.get)
        loser_score = ahp.final_scores[loser]
        
        f.write(f"Research Question: Is Vertical Farming a sustainable and applicable\n")
        f.write(f"food strategy for Singapore?\n\n")
        
        f.write(f"ANSWER: ")
        if winner == "Vertical Farm":
            f.write(f"YES - Vertical Farming scores {winner_score:.1%} vs {loser_score:.1%}\n")
            f.write(f"for Traditional Greenhouse Farming.\n\n")
        else:
            f.write(f"NO - Traditional Greenhouse Farming outperforms Vertical Farming\n")
            f.write(f"with {winner_score:.1%} vs {loser_score:.1%} in the multi-criteria assessment.\n\n")
        
        margin = abs(winner_score - loser_score)
        if margin < 0.05:
            strength = "MARGINAL"
        elif margin < 0.15:
            strength = "MODERATE"
        else:
            strength = "STRONG"
        
        f.write(f"Decision Strength: {strength} ({margin:.1%} difference)\n\n")
        
        # LCC Results
        f.write("=" * 80 + "\n")
        f.write("1. LIFE CYCLE COSTING (LCC) RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Cost Structure Comparison:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<35} {'Vertical Farm':>20} {'Traditional Farm':>20}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'LCOv (SGD/kg)':<35} {lcc_vf['lcov']:>20.3f} {lcc_tf['lcov']:>20.3f}\n")
        f.write(f"{'CAPEX per kg (SGD/kg)':<35} {lcc_vf['capex_per_kg']:>20.3f} {lcc_tf['capex_per_kg']:>20.3f}\n")
        f.write(f"{'OPEX per kg (SGD/kg)':<35} {lcc_vf['opex_per_kg']:>20.3f} {lcc_tf['opex_per_kg']:>20.3f}\n")
        f.write(f"{'Product Price (SGD/kg)':<35} {lcc_vf['product_price']:>20.2f} {lcc_tf['product_price']:>20.2f}\n")
        f.write(f"{'Profit Margin (%)':<35} {lcc_vf['profit_margin']:>20.1f} {lcc_tf['profit_margin']:>20.1f}\n")
        f.write(f"{'Annual Output (kg/yr)':<35} {lcc_vf['annual_output']:>20,.0f} {lcc_tf['annual_output']:>20,.0f}\n")
        f.write(f"{'Net Cash Flow (SGD/yr)':<35} {lcc_vf['net_cashflow']:>20,.0f} {lcc_tf['net_cashflow']:>20,.0f}\n\n")
        
        f.write("Economic Insights:\n")
        if lcc_vf['lcov'] < lcc_tf['lcov']:
            f.write(f"• VF has LOWER cost per kg ({lcc_vf['lcov']:.2f} vs {lcc_tf['lcov']:.2f} SGD/kg)\n")
        else:
            f.write(f"• TF has LOWER cost per kg ({lcc_tf['lcov']:.2f} vs {lcc_vf['lcov']:.2f} SGD/kg)\n")
        
        if lcc_vf['profit_margin'] > lcc_tf['profit_margin']:
            f.write(f"• VF has HIGHER profit margin ({lcc_vf['profit_margin']:.1f}% vs {lcc_tf['profit_margin']:.1f}%)\n")
        else:
            f.write(f"• TF has HIGHER profit margin ({lcc_tf['profit_margin']:.1f}% vs {lcc_vf['profit_margin']:.1f}%)\n")
        
        if lcc_vf['annual_output'] > lcc_tf['annual_output']:
            f.write(f"• VF has HIGHER production capacity (+{(lcc_vf['annual_output']/lcc_tf['annual_output']-1)*100:.0f}%)\n")
        else:
            f.write(f"• TF has HIGHER production capacity (+{(lcc_tf['annual_output']/lcc_vf['annual_output']-1)*100:.0f}%)\n")
        f.write("\n")
        
        # LCIA Results
        f.write("=" * 80 + "\n")
        f.write("2. LIFE CYCLE IMPACT ASSESSMENT (LCIA) RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Environmental Impact Comparison (per kg produce):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Impact Category':<25} {'Vertical Farm':>20} {'Traditional Farm':>18} {'Winner':<12}\n")
        f.write("-" * 80 + "\n")
        
        impact_categories = {
            'GWP100': 'Climate Change (kg CO2-eq)',
            'HOFP': 'Human Toxicity (kg NOx-eq)',
            'PMFP': 'Particulate Matter (kg PM2.5-eq)',
            'AP': 'Acidification (kg SO2-eq)',
            'EOFP': 'Eutrophication (kg NOx-eq)',
            'FFP': 'Fossil Fuel (MJ)'
        }
        
        vf_better_count = 0
        for key, name in impact_categories.items():
            vf_val = lcia_vf[key]
            tf_val = lcia_tf[key]
            winner_env = "VF" if vf_val < tf_val else "TF"
            if winner_env == "VF":
                vf_better_count += 1
            f.write(f"{name:<25} {vf_val:>20.4f} {tf_val:>18.4f} {winner_env:<12}\n")
        
        f.write("\n")
        f.write(f"Environmental Performance: VF wins in {vf_better_count}/6 categories\n\n")
        
        # Resource Use
        f.write("Resource Use Comparison (per kg):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Resource':<25} {'Vertical Farm':>20} {'Traditional Farm':>20}\n")
        f.write("-" * 80 + "\n")
        vf_inv = lcia_vf['inventory']
        tf_inv = lcia_tf['inventory']
        f.write(f"{'Electricity (kWh)':<25} {vf_inv['electricity_kwh']:>20.3f} {tf_inv['electricity_kwh']:>20.3f}\n")
        f.write(f"{'Water (liters)':<25} {vf_inv['water_m3']*1000:>20.1f} {tf_inv['water_m3']*1000:>20.1f}\n")
        f.write(f"{'Fertilizer (g)':<25} {vf_inv['fertilizer_kg']*1000:>20.1f} {tf_inv['fertilizer_kg']*1000:>20.1f}\n")
        f.write(f"{'Pesticide (mg)':<25} {vf_inv['pesticide_kg']*1000000:>20.1f} {tf_inv['pesticide_kg']*1000000:>20.1f}\n\n")
        
        # AHP Results
        f.write("=" * 80 + "\n")
        f.write("3. MULTI-CRITERIA DECISION ANALYSIS (AHP) RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Final AHP Scores:\n")
        f.write("-" * 80 + "\n")
        sorted_alts = sorted(ahp.final_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (alt, score) in enumerate(sorted_alts, 1):
            f.write(f"{i}. {alt:<30} {score:.4f} ({score*100:.2f}%)\n")
        f.write("\n")
        
        f.write("Criteria Weights:\n")
        f.write("-" * 80 + "\n")
        for criterion, weight in ahp.criteria_weights.items():
            f.write(f"{criterion:<30} {weight:.4f} ({weight*100:.1f}%)\n")
        f.write("\n")
        
        f.write("Consistency Ratios (CR < 0.1 is acceptable):\n")
        f.write("-" * 80 + "\n")
        for comparison, cr in ahp.consistency_ratios.items():
            status = "✓ PASS" if cr < 0.1 else "✗ FAIL"
            f.write(f"{comparison:<40} CR = {cr:.4f} {status}\n")
        f.write("\n")
        
        # Conclusions
        f.write("=" * 80 + "\n")
        f.write("4. CONCLUSIONS AND RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Key Findings:\n\n")
        
        # Economic
        if lcc_vf['lcov'] < lcc_tf['lcov']:
            f.write(f"1. ECONOMIC: VF is MORE cost-competitive\n")
            f.write(f"   - VF LCOv: {lcc_vf['lcov']:.2f} SGD/kg vs TF: {lcc_tf['lcov']:.2f} SGD/kg\n")
            f.write(f"   - Cost advantage: {((lcc_tf['lcov']/lcc_vf['lcov'])-1)*100:.1f}% lower\n\n")
        else:
            f.write(f"1. ECONOMIC: TF is MORE cost-competitive\n")
            f.write(f"   - TF LCOv: {lcc_tf['lcov']:.2f} SGD/kg vs VF: {lcc_vf['lcov']:.2f} SGD/kg\n")
            f.write(f"   - Cost disadvantage for VF: {((lcc_vf['lcov']/lcc_tf['lcov'])-1)*100:.1f}% higher\n\n")
        
        # Environmental
        if vf_better_count >= 4:
            f.write(f"2. ENVIRONMENTAL: VF is SUPERIOR\n")
            f.write(f"   - VF wins in {vf_better_count}/6 impact categories\n")
            ghg_reduction = (1 - lcia_vf['GWP100']/lcia_tf['GWP100']) * 100
            f.write(f"   - Climate change: {ghg_reduction:.1f}% lower GHG emissions\n")
            water_reduction = (1 - vf_inv['water_m3']/tf_inv['water_m3']) * 100
            f.write(f"   - Water use: {water_reduction:.1f}% less water per kg\n\n")
        else:
            f.write(f"2. ENVIRONMENTAL: Results are MIXED\n")
            f.write(f"   - VF wins in only {vf_better_count}/6 impact categories\n")
            f.write(f"   - Need to consider trade-offs between impact types\n\n")
        
        # Singapore Context
        f.write(f"3. SINGAPORE CONTEXT:\n")
        f.write(f"   - Land scarcity: VF produces {lcc_vf['annual_output']/lcc_tf['annual_output']:.1f}x more per unit area\n")
        f.write(f"   - Food security: Higher productivity supports '30 by 30' goal\n")
        f.write(f"   - Climate resilience: VF less affected by weather disruptions\n")
        f.write(f"   - Urban integration: VF can be located closer to consumers\n\n")
        
        # Final Recommendation
        f.write("=" * 80 + "\n")
        f.write("FINAL RECOMMENDATION\n")
        f.write("=" * 80 + "\n\n")
        
        if winner == "Vertical Farm":
            f.write("Vertical Farming IS recommended as a sustainable food strategy for Singapore.\n\n")
            f.write("Rationale:\n")
            f.write(f"• Overall score: {winner_score:.1%} (Strong multi-criteria performance)\n")
            f.write(f"• Economic viability with {lcc_vf['profit_margin']:.1f}% profit margin\n")
            f.write(f"• Superior environmental performance\n")
            f.write(f"• High productivity: {lcc_vf['annual_output']/1000000:.1f}M kg/year\n")
            f.write(f"• Addresses Singapore's land constraints\n")
            f.write(f"• Enhances food security and climate resilience\n\n")
            f.write("Implementation Considerations:\n")
            f.write("• High initial CAPEX requires financing support\n")
            f.write("• Energy efficiency is critical - continue LED technology improvements\n")
            f.write("• Scale up gradually to build expertise\n")
            f.write("• Focus on high-value crops (leafy greens, herbs)\n")
        else:
            f.write("Traditional Greenhouse Farming currently outperforms Vertical Farming.\n\n")
            f.write("However, VF may still have a role in Singapore's food strategy:\n")
            f.write(f"• VF scored {loser_score:.1%} - not negligible\n")
            f.write(f"• Strengths: {', '.join([k for k, v in ahp.detailed_scores['Vertical Farm'].items() if v > 0.02])}\n")
            f.write(f"• Consider VF for:\n")
            f.write(f"  - Premium products justifying higher costs\n")
            f.write(f"  - R&D and technology development\n")
            f.write(f"  - Niche applications (rooftop farms, urban areas)\n\n")
            f.write("Recommendations:\n")
            f.write("• Continue supporting traditional greenhouse agriculture\n")
            f.write("• Invest in VF R&D to reduce costs\n")
            f.write("• Pursue hybrid approach: both VF and greenhouse\n")
            f.write("• Monitor VF technology improvements and reassess\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main analysis execution"""
    
    print("=" * 80)
    print("COMPREHENSIVE VERTICAL FARMING ANALYSIS FOR SINGAPORE")
    print("=" * 80)
    print()
    
    # Define file path
    BASE_FILE = 'Corrected_Base_Data_Singapore.xlsx'
    
    if not os.path.exists(BASE_FILE):
        print(f"ERROR: {BASE_FILE} not found!")
        print("Please ensure the corrected base data file is in the current directory.")
        return
    
    # Debug: Check Excel structure
    print("Checking Excel file structure...")
    try:
        test_df = pd.read_excel(BASE_FILE, sheet_name='VF_Parameters', header=2, nrows=5)
        print(f"✓ Found columns: {', '.join(test_df.columns.tolist())}")
        print()
    except Exception as e:
        print(f"✗ Error reading Excel: {e}")
        print("Please check the Excel file format.")
        return
    
    print("Step 1: Calculating Life Cycle Costs (LCC)...")
    print("-" * 80)
    
    # VF LCC
    lcc_vf_analyzer = LCCAnalysis(BASE_FILE, 'VF_Parameters', 'VF_CAPEX', 'VF_OPEX')
    lcc_vf = lcc_vf_analyzer.calculate_lcc()
    print(f"VF LCOv: {lcc_vf['lcov']:.3f} SGD/kg")
    print(f"VF Profit Margin: {lcc_vf['profit_margin']:.1f}%")
    
    # TF LCC
    lcc_tf_analyzer = LCCAnalysis(BASE_FILE, 'TF_Parameters', 'TF_CAPEX', 'TF_OPEX')
    lcc_tf = lcc_tf_analyzer.calculate_lcc()
    print(f"TF LCOv: {lcc_tf['lcov']:.3f} SGD/kg")
    print(f"TF Profit Margin: {lcc_tf['profit_margin']:.1f}%")
    print()
    
    print("Step 2: Calculating Life Cycle Impacts (LCIA)...")
    print("-" * 80)
    
    # VF LCIA
    lcia_vf_analyzer = LCIAAnalysis(BASE_FILE, 'VF_Parameters')
    lcia_vf = lcia_vf_analyzer.calculate_impacts_per_kg('VF')
    print(f"VF GWP100: {lcia_vf['GWP100']:.4f} kg CO2-eq/kg")
    
    # TF LCIA
    lcia_tf_analyzer = LCIAAnalysis(BASE_FILE, 'TF_Parameters')
    lcia_tf = lcia_tf_analyzer.calculate_impacts_per_kg('TF')
    print(f"TF GWP100: {lcia_tf['GWP100']:.4f} kg CO2-eq/kg")
    print()
    
    print("Step 3: Performing AHP Multi-Criteria Analysis...")
    print("-" * 80)
    
    # Initialize AHP
    ahp = AHPAnalysis(['Vertical Farm', 'Traditional Farm'])
    
    # Criteria weights (balanced approach for Singapore)
    criteria_comparisons = {
        (0, 1): 1,      # Economic = Environmental (both critical)
        (0, 2): 3,      # Economic > Social
        (1, 2): 3       # Environmental > Social
    }
    ahp.add_criteria_comparisons(['Economic', 'Environmental', 'Social'], criteria_comparisons)
    
    # Economic sub-criteria
    economic_sub = {
        (0, 1): 3, (0, 2): 2, (0, 3): 1,
        (1, 2): 1/2, (1, 3): 1/3, (2, 3): 1/2
    }
    ahp.add_subcriteria_comparisons('Economic', ['LCOv', 'CAPEX', 'OPEX', 'Profitability'], economic_sub)
    
    # Environmental sub-criteria
    environmental_sub = {
        (0, 1): 5, (0, 2): 5, (0, 3): 3, (0, 4): 5, (0, 5): 3,
        (1, 2): 1, (1, 3): 1/3, (1, 4): 1, (1, 5): 1/5,
        (2, 3): 1/3, (2, 4): 1, (2, 5): 1/5,
        (3, 4): 3, (3, 5): 1/3, (4, 5): 1/5
    }
    ahp.add_subcriteria_comparisons('Environmental', ['GWP100', 'HOFP', 'PMFP', 'AP', 'EOFP', 'FFP'], environmental_sub)
    
    # Social sub-criteria
    social_sub = {(0, 1): 1}
    ahp.add_subcriteria_comparisons('Social', ['Food Security', 'Urban Integration'], social_sub)
    
    # Alternative comparisons from data
    # Economic
    ahp.add_alternative_scores_for_subcriterion('Economic', 'LCOv', 
        {(0, 1): generate_comparison_from_data(lcc_vf['lcov'], lcc_tf['lcov'], False)})
    ahp.add_alternative_scores_for_subcriterion('Economic', 'CAPEX',
        {(0, 1): generate_comparison_from_data(lcc_vf['capex_per_kg'], lcc_tf['capex_per_kg'], False)})
    ahp.add_alternative_scores_for_subcriterion('Economic', 'OPEX',
        {(0, 1): generate_comparison_from_data(lcc_vf['opex_per_kg'], lcc_tf['opex_per_kg'], False)})
    ahp.add_alternative_scores_for_subcriterion('Economic', 'Profitability',
        {(0, 1): generate_comparison_from_data(lcc_vf['profit_margin'], lcc_tf['profit_margin'], True)})
    
    # Environmental
    for impact in ['GWP100', 'HOFP', 'PMFP', 'AP', 'EOFP', 'FFP']:
        ahp.add_alternative_scores_for_subcriterion('Environmental', impact,
            {(0, 1): generate_comparison_from_data(lcia_vf[impact], lcia_tf[impact], False)})
    
    # Social
    ahp.add_alternative_scores_for_subcriterion('Social', 'Food Security',
        {(0, 1): generate_comparison_from_data(lcc_vf['annual_output'], lcc_tf['annual_output'], True)})
    ahp.add_alternative_scores_for_subcriterion('Social', 'Urban Integration',
        {(0, 1): 3})  # VF better for urban areas
    
    # Calculate final scores
    final_scores = ahp.calculate_final_scores()
    
    print(f"Final AHP Scores:")
    for alt, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {alt}: {score:.4f} ({score*100:.2f}%)")
    print()
    
    print("Step 4: Generating Comprehensive Report...")
    print("-" * 80)
    
    report_path = os.path.join(OUTPUT_DIR, 'VF_Sustainability_Assessment_Singapore.txt')
    generate_comprehensive_report(lcc_vf, lcc_tf, lcia_vf, lcia_tf, ahp, report_path)
    
    print(f"✓ Report saved: {report_path}")
    print()
    
    print("Step 5: Generating Visualizations...")
    print("-" * 80)
    
    # Create LCC visualizations
    create_lcc_visualizations(lcc_vf, lcc_tf, OUTPUT_DIR)
    
    # Create LCIA visualizations
    create_lcia_visualizations(lcia_vf, lcia_tf, OUTPUT_DIR)
    
    # Create AHP visualizations
    create_ahp_visualizations(ahp, OUTPUT_DIR)
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print(f"Results saved in '{OUTPUT_DIR}' folder:")
    print(f"  1. {report_path}")
    print(f"  2. {os.path.join(OUTPUT_DIR, 'LCC_Analysis.png')}")
    print(f"  3. {os.path.join(OUTPUT_DIR, 'LCIA_Analysis.png')}")
    print(f"  4. {os.path.join(OUTPUT_DIR, 'LCIA_Radar_Chart.png')}")
    print(f"  5. {os.path.join(OUTPUT_DIR, 'AHP_Analysis.png')}")
    print()
    
    # Print quick summary
    winner = max(final_scores, key=final_scores.get)
    print(f"CONCLUSION: {winner} is the recommended approach")
    print(f"Score: {final_scores[winner]:.2%}")
    print()

if __name__ == "__main__":
    main()
