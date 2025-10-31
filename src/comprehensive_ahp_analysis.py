"""
COMPREHENSIVE ANALYTIC HIERARCHY PROCESS (AHP) ANALYSIS
========================================================

Multi-criteria decision analysis for Vertical Farming vs Traditional Farming
integrating Life Cycle Costing (LCC), Life Cycle Impact Assessment (LCIA),
and operational performance data.

This script performs:
1. Data extraction from LCC analysis, LCIA outputs, and base data
2. Hierarchical criteria structuring (Economic, Environmental, Operational)
3. Pairwise comparison matrices with consistency checking
4. Normalized weights calculation
5. Alternative scoring and ranking
6. Sensitivity analysis
7. Comprehensive visualizations and reporting

Decision Hierarchy:
-------------------
Goal: Select Sustainable Farming Strategy for Singapore
├── Economic Criteria (Cost, Profitability)
│   ├── Levelized Cost of Vegetables (LCOv)
│   ├── Initial Investment (CAPEX)
│   ├── Operating Cost (OPEX per kg)
│   └── Profit Margin
├── Environmental Criteria (Impact Categories)
│   ├── Climate Change (GWP100)
│   ├── Human Health (HOFP, PMFP)
│   ├── Ecosystem Quality (AP, EOFP)
│   └── Resource Depletion (FFP)
└── Operational Criteria (Performance)
    ├── Production Capacity
    ├── Resource Efficiency
    └── Space Efficiency

Author: AHP Module
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from typing import Dict, Tuple, List
import json

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = 'ahp_analysis'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =============================================================================
# SECTION 1: DATA INTEGRATION MODULE
# =============================================================================

class DataIntegrator:
    """Integrate data from LCC, LCIA, and base data sources"""
    
    def __init__(self, base_data_file='Corrected_Base_Data_Singapore.xlsx',
                 vf_lcia_file='VF_LCIA_ready_multiimpact.xlsx',
                 tf_lcia_file='TF_LCIA_ready_multiimpact.xlsx'):
        self.base_data_file = base_data_file
        self.vf_lcia_file = vf_lcia_file
        self.tf_lcia_file = tf_lcia_file
        self.integrated_data = {}
        
    def extract_lcc_data(self):
        """Extract economic data from base data file (runs LCC internally)"""
        
        print("Extracting LCC data from base data...")
        
        # Load VF data
        vf_params = pd.read_excel(self.base_data_file, sheet_name='VF_Parameters', header=2)
        vf_capex = pd.read_excel(self.base_data_file, sheet_name='VF_CAPEX', header=2)
        vf_opex = pd.read_excel(self.base_data_file, sheet_name='VF_OPEX', header=2)
        
        # Load TF data
        tf_params = pd.read_excel(self.base_data_file, sheet_name='TF_Parameters', header=2)
        tf_capex = pd.read_excel(self.base_data_file, sheet_name='TF_CAPEX', header=2)
        tf_opex = pd.read_excel(self.base_data_file, sheet_name='TF_OPEX', header=2)
        
        def get_param(df, param_name):
            valid_df = df[df['Parameter'].notna()]
            row = valid_df[valid_df['Parameter'] == param_name]
            if len(row) > 0:
                return row['Value'].values[0]
            return None
        
        # Extract VF economic data
        vf_annual_output = get_param(vf_params, 'Annual edible output (kg/yr)')
        vf_discount_rate = get_param(vf_params, 'Discount rate r (decimal)')
        vf_lifetime = get_param(vf_params, 'Facility lifetime (years)')
        vf_product_price = get_param(vf_params, 'Product price (SGD/kg)')
        
        # Calculate VF CAPEX
        valid_vf_capex = vf_capex[vf_capex['Cost_SGD'].notna() & vf_capex['Asset'].notna()]
        valid_vf_capex = valid_vf_capex[valid_vf_capex['Asset'] != 'TOTAL']
        vf_capex_total = valid_vf_capex['Cost_SGD'].sum()
        
        # Calculate annualized CAPEX using Capital Recovery Factor (CRF)
        def annualize_cost(cost, lifetime, discount_rate):
            if lifetime == 0 or discount_rate == 0:
                return cost / 20  # Default to 20 years if parameters missing
            crf = (discount_rate * (1 + discount_rate)**lifetime) / ((1 + discount_rate)**lifetime - 1)
            return cost * crf
        
        vf_capex_annualized = sum(
            annualize_cost(row['Cost_SGD'], row['Lifetime_years'], vf_discount_rate)
            for _, row in valid_vf_capex.iterrows()
        )
        
        # Calculate VF OPEX
        valid_vf_opex = vf_opex[vf_opex['Cost_per_kg_SGD'].notna() & vf_opex['Item'].notna()]
        valid_vf_opex = valid_vf_opex[valid_vf_opex['Item'] != 'TOTAL per kg']
        vf_opex_per_kg = valid_vf_opex['Cost_per_kg_SGD'].sum()
        vf_opex_annual = vf_opex_per_kg * vf_annual_output
        
        # Calculate VF LCOv and profit
        vf_capex_per_kg = vf_capex_annualized / vf_annual_output
        vf_lcov = vf_capex_per_kg + vf_opex_per_kg
        vf_revenue_annual = vf_product_price * vf_annual_output
        vf_cost_annual = vf_capex_annualized + vf_opex_annual
        vf_profit_annual = vf_revenue_annual - vf_cost_annual
        vf_profit_margin = (vf_profit_annual / vf_revenue_annual * 100) if vf_revenue_annual > 0 else 0
        
        # Extract TF economic data
        tf_annual_output = get_param(tf_params, 'Annual edible output (kg/yr)')
        tf_discount_rate = get_param(tf_params, 'Discount rate r (decimal)')
        tf_lifetime = get_param(tf_params, 'Facility lifetime (years)')
        tf_product_price = get_param(tf_params, 'Product price (SGD/kg)')
        
        # Calculate TF CAPEX
        valid_tf_capex = tf_capex[tf_capex['Cost_SGD'].notna() & tf_capex['Asset'].notna()]
        valid_tf_capex = valid_tf_capex[valid_tf_capex['Asset'] != 'TOTAL']
        tf_capex_total = valid_tf_capex['Cost_SGD'].sum()
        
        tf_capex_annualized = sum(
            annualize_cost(row['Cost_SGD'], row['Lifetime_years'], tf_discount_rate)
            for _, row in valid_tf_capex.iterrows()
        )
        
        # Calculate TF OPEX
        valid_tf_opex = tf_opex[tf_opex['Cost_per_kg_SGD'].notna() & tf_opex['Item'].notna()]
        valid_tf_opex = valid_tf_opex[valid_tf_opex['Item'] != 'TOTAL per kg']
        tf_opex_per_kg = valid_tf_opex['Cost_per_kg_SGD'].sum()
        tf_opex_annual = tf_opex_per_kg * tf_annual_output
        
        # Calculate TF LCOv and profit
        tf_capex_per_kg = tf_capex_annualized / tf_annual_output
        tf_lcov = tf_capex_per_kg + tf_opex_per_kg
        tf_revenue_annual = tf_product_price * tf_annual_output
        tf_cost_annual = tf_capex_annualized + tf_opex_annual
        tf_profit_annual = tf_revenue_annual - tf_cost_annual
        tf_profit_margin = (tf_profit_annual / tf_revenue_annual * 100) if tf_revenue_annual > 0 else 0
        
        lcc_data = {
            'VF': {
                'lcov': vf_lcov,
                'capex_total': vf_capex_total,
                'capex_per_kg': vf_capex_per_kg,
                'opex_per_kg': vf_opex_per_kg,
                'profit_margin': vf_profit_margin,
                'annual_output': vf_annual_output,
                'product_price': vf_product_price
            },
            'TF': {
                'lcov': tf_lcov,
                'capex_total': tf_capex_total,
                'capex_per_kg': tf_capex_per_kg,
                'opex_per_kg': tf_opex_per_kg,
                'profit_margin': tf_profit_margin,
                'annual_output': tf_annual_output,
                'product_price': tf_product_price
            }
        }
        
        print("✓ LCC data extracted successfully")
        return lcc_data
    
    def extract_lcia_data(self):
        """Extract environmental impact data from LCIA files"""
        
        print("Extracting LCIA data...")
        
        # Load VF LCIA totals
        vf_lcia = pd.read_excel(self.vf_lcia_file, sheet_name='LCIA_totals_multi')
        vf_impacts = {}
        for _, row in vf_lcia.iterrows():
            vf_impacts[row['category']] = row['perkg_total']
        
        # Load TF LCIA totals
        tf_lcia = pd.read_excel(self.tf_lcia_file, sheet_name='LCIA_totals_multi')
        tf_impacts = {}
        for _, row in tf_lcia.iterrows():
            tf_impacts[row['category']] = row['perkg_total']
        
        lcia_data = {
            'VF': vf_impacts,
            'TF': tf_impacts
        }
        
        print("✓ LCIA data extracted successfully")
        return lcia_data
    
    def extract_operational_data(self):
        """Extract operational performance data"""
        
        print("Extracting operational data...")
        
        # Load VF parameters
        vf_params = pd.read_excel(self.base_data_file, sheet_name='VF_Parameters', header=2)
        vf_derived = pd.read_excel(self.vf_lcia_file, sheet_name='Derived_perkg')
        
        # Load TF parameters
        tf_params = pd.read_excel(self.base_data_file, sheet_name='TF_Parameters', header=2)
        tf_derived = pd.read_excel(self.tf_lcia_file, sheet_name='Derived_perkg')
        
        def get_param(df, param_name):
            valid_df = df[df['Parameter'].notna()]
            row = valid_df[valid_df['Parameter'] == param_name]
            if len(row) > 0:
                return row['Value'].values[0]
            return None
        
        def get_derived(df, item_name):
            row = df[df['Item'] == item_name]
            if len(row) > 0:
                return row['Value'].values[0]
            return None
        
        # Extract VF operational data
        vf_annual_output = get_derived(vf_derived, 'Annual output (kg/yr)')
        vf_electricity_per_kg = get_derived(vf_derived, 'Electricity (kWh/kg)')
        vf_water_per_kg = get_derived(vf_derived, 'Water (m3/kg)')
        
        # Extract TF operational data
        tf_annual_output = get_derived(tf_derived, 'Annual output (kg/yr)')
        tf_electricity_per_kg = get_derived(tf_derived, 'Electricity (kWh/kg)')
        tf_water_per_kg = get_derived(tf_derived, 'Water (m3/kg)')
        
        operational_data = {
            'VF': {
                'annual_output': vf_annual_output,
                'electricity_per_kg': vf_electricity_per_kg,
                'water_per_kg': vf_water_per_kg,
                'space_efficiency': 1.0  # Normalized reference value (high for VF)
            },
            'TF': {
                'annual_output': tf_annual_output,
                'electricity_per_kg': tf_electricity_per_kg,
                'water_per_kg': tf_water_per_kg,
                'space_efficiency': 0.1  # Normalized reference value (low for TF)
            }
        }
        
        print("✓ Operational data extracted successfully")
        return operational_data
    
    def integrate_all_data(self):
        """Integrate all data sources"""
        
        print("\n" + "="*60)
        print("INTEGRATING DATA FROM ALL SOURCES")
        print("="*60 + "\n")
        
        lcc_data = self.extract_lcc_data()
        lcia_data = self.extract_lcia_data()
        operational_data = self.extract_operational_data()
        
        self.integrated_data = {
            'VF': {
                'economic': lcc_data['VF'],
                'environmental': lcia_data['VF'],
                'operational': operational_data['VF']
            },
            'TF': {
                'economic': lcc_data['TF'],
                'environmental': lcia_data['TF'],
                'operational': operational_data['TF']
            }
        }
        
        print("\n✓ All data integrated successfully\n")
        return self.integrated_data

# =============================================================================
# SECTION 2: AHP CORE ENGINE
# =============================================================================

class AHPEngine:
    """Core AHP calculation engine with consistency checking"""
    
    # Saaty's Random Index for consistency checking
    RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 
          6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    
    @staticmethod
    def calculate_weights(pairwise_matrix: np.ndarray) -> Tuple[np.ndarray, float, float, bool]:
        """
        Calculate weights from pairwise comparison matrix using eigenvalue method
        
        Returns:
            weights: Normalized weights
            lambda_max: Maximum eigenvalue
            CI: Consistency Index
            CR: Consistency Ratio
            is_consistent: Whether matrix is consistent (CR < 0.1)
        """
        n = pairwise_matrix.shape[0]
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)
        
        # Find maximum eigenvalue and corresponding eigenvector
        max_idx = np.argmax(eigenvalues.real)
        lambda_max = eigenvalues[max_idx].real
        principal_eigenvector = eigenvectors[:, max_idx].real
        
        # Normalize to get weights
        weights = principal_eigenvector / principal_eigenvector.sum()
        
        # Calculate Consistency Index (CI)
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        
        # Calculate Consistency Ratio (CR)
        RI = AHPEngine.RI.get(n, 1.49)
        CR = CI / RI if RI > 0 else 0
        
        # Check consistency (CR < 0.1 is acceptable)
        is_consistent = CR < 0.1
        
        return weights, lambda_max, CI, CR, is_consistent
    
    @staticmethod
    def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        """Normalize pairwise comparison matrix by column sums"""
        column_sums = matrix.sum(axis=0)
        return matrix / column_sums
    
    @staticmethod
    def geometric_mean_method(pairwise_matrix: np.ndarray) -> np.ndarray:
        """Alternative method: Calculate weights using geometric mean"""
        n = pairwise_matrix.shape[0]
        geometric_means = np.prod(pairwise_matrix, axis=1) ** (1/n)
        weights = geometric_means / geometric_means.sum()
        return weights

# =============================================================================
# SECTION 3: AHP HIERARCHY BUILDER
# =============================================================================

class AHPHierarchy:
    """Build and manage the AHP decision hierarchy"""
    
    def __init__(self, integrated_data: Dict):
        self.data = integrated_data
        self.criteria_weights = {}
        self.subcriteria_weights = {}
        self.alternative_scores = {}
        self.final_scores = {}
        
    def define_criteria_pairwise_matrix(self) -> Dict:
        """
        Define pairwise comparison matrix for main criteria
        
        Criteria: Economic, Environmental, Operational
        
        Scale (Saaty's scale):
        1 = Equal importance
        3 = Moderate importance
        5 = Strong importance
        7 = Very strong importance
        9 = Extreme importance
        2,4,6,8 = Intermediate values
        
        For Singapore context:
        - Economic vs Environmental: 1/2 (Environmental slightly more important for sustainability)
        - Economic vs Operational: 2 (Economic moderately more important)
        - Environmental vs Operational: 3 (Environmental strongly more important)
        """
        
        # You can modify these values based on stakeholder input
        criteria_matrix = np.array([
            [1,    1/2,  2],    # Economic
            [2,    1,    3],    # Environmental
            [1/2,  1/3,  1]     # Operational
        ])
        
        criteria_names = ['Economic', 'Environmental', 'Operational']
        
        return {
            'matrix': criteria_matrix,
            'names': criteria_names,
            'description': 'Main criteria comparison for sustainable farming'
        }
    
    def define_economic_subcriteria_matrix(self) -> Dict:
        """
        Define pairwise comparison matrix for economic sub-criteria
        
        Sub-criteria: LCOv, CAPEX, OPEX, Profit Margin
        
        For Singapore context (cost-sensitive market):
        - LCOv vs CAPEX: 3 (LCOv more important than initial investment)
        - LCOv vs OPEX: 2 (LCOv moderately more important)
        - LCOv vs Profit: 1 (Equal importance)
        - CAPEX vs OPEX: 1/2 (OPEX slightly more important for sustainability)
        - CAPEX vs Profit: 1/3 (Profit more important)
        - OPEX vs Profit: 1/2 (Profit slightly more important)
        """
        
        subcriteria_matrix = np.array([
            [1,    3,    2,    1],     # LCOv (lower is better)
            [1/3,  1,    1/2,  1/3],   # CAPEX (lower is better)
            [1/2,  2,    1,    1/2],   # OPEX (lower is better)
            [1,    3,    2,    1]      # Profit Margin (higher is better)
        ])
        
        subcriteria_names = ['LCOv', 'CAPEX', 'OPEX_per_kg', 'Profit_Margin']
        
        return {
            'matrix': subcriteria_matrix,
            'names': subcriteria_names,
            'description': 'Economic sub-criteria comparison'
        }
    
    def define_environmental_subcriteria_matrix(self) -> Dict:
        """
        Define pairwise comparison matrix for environmental sub-criteria
        
        Sub-criteria: GWP100, HOFP, PMFP, AP, EOFP, FFP
        
        For Singapore context (climate and health priorities):
        - GWP100 (Climate Change): Highest priority
        - HOFP, PMFP (Health): High priority
        - AP, EOFP (Ecosystem): Moderate priority
        - FFP (Resource): Lower priority (but still important)
        """
        
        subcriteria_matrix = np.array([
            [1,    3,    3,    5,    5,    2],   # GWP100 (Climate Change)
            [1/3,  1,    1,    3,    3,    1/2], # HOFP (Human Ozone)
            [1/3,  1,    1,    3,    3,    1/2], # PMFP (Particulate Matter)
            [1/5,  1/3,  1/3,  1,    1,    1/3], # AP (Acidification)
            [1/5,  1/3,  1/3,  1,    1,    1/3], # EOFP (Ecosystem Ozone)
            [1/2,  2,    2,    3,    3,    1]    # FFP (Fossil Fuel)
        ])
        
        subcriteria_names = ['GWP100', 'HOFP', 'PMFP', 'AP', 'EOFP', 'FFP']
        
        return {
            'matrix': subcriteria_matrix,
            'names': subcriteria_names,
            'description': 'Environmental impact sub-criteria comparison'
        }
    
    def define_operational_subcriteria_matrix(self) -> Dict:
        """
        Define pairwise comparison matrix for operational sub-criteria
        
        Sub-criteria: Production Capacity, Resource Efficiency, Space Efficiency
        
        For Singapore context (land-scarce, water-conscious):
        - Production vs Resource: 1/2 (Resource efficiency slightly more important)
        - Production vs Space: 1/3 (Space efficiency more important)
        - Resource vs Space: 1/2 (Space slightly more important)
        """
        
        subcriteria_matrix = np.array([
            [1,    1/2,  1/3],  # Production Capacity
            [2,    1,    1/2],  # Resource Efficiency
            [3,    2,    1]     # Space Efficiency
        ])
        
        subcriteria_names = ['Production_Capacity', 'Resource_Efficiency', 'Space_Efficiency']
        
        return {
            'matrix': subcriteria_matrix,
            'names': subcriteria_names,
            'description': 'Operational performance sub-criteria comparison'
        }
    
    def normalize_alternatives(self, values: Dict, criterion_name: str, 
                              lower_is_better: bool = True) -> Dict:
        """
        Normalize alternative values for comparison
        
        Args:
            values: Dict with 'VF' and 'TF' values
            criterion_name: Name of the criterion
            lower_is_better: If True, lower values are better (e.g., cost, emissions)
        
        Returns:
            Normalized scores where higher is better
        """
        vf_val = values['VF']
        tf_val = values['TF']
        
        # Avoid division by zero
        if vf_val == 0 and tf_val == 0:
            return {'VF': 0.5, 'TF': 0.5}
        
        if lower_is_better:
            # For "lower is better" criteria, invert the relationship
            total = (1/vf_val) + (1/tf_val) if vf_val > 0 and tf_val > 0 else 0
            if total > 0:
                return {
                    'VF': (1/vf_val) / total,
                    'TF': (1/tf_val) / total
                }
            else:
                return {'VF': 0.5, 'TF': 0.5}
        else:
            # For "higher is better" criteria, use direct proportion
            total = vf_val + tf_val
            if total > 0:
                return {
                    'VF': vf_val / total,
                    'TF': tf_val / total
                }
            else:
                return {'VF': 0.5, 'TF': 0.5}
    
    def calculate_alternative_scores_for_criteria(self) -> Dict:
        """Calculate normalized scores for alternatives under each criterion"""
        
        scores = {
            'economic': {},
            'environmental': {},
            'operational': {}
        }
        
        # Economic scores (lower is better for costs, higher is better for profit)
        scores['economic']['LCOv'] = self.normalize_alternatives(
            {'VF': self.data['VF']['economic']['lcov'],
             'TF': self.data['TF']['economic']['lcov']},
            'LCOv', lower_is_better=True
        )
        
        scores['economic']['CAPEX'] = self.normalize_alternatives(
            {'VF': self.data['VF']['economic']['capex_per_kg'],
             'TF': self.data['TF']['economic']['capex_per_kg']},
            'CAPEX', lower_is_better=True
        )
        
        scores['economic']['OPEX_per_kg'] = self.normalize_alternatives(
            {'VF': self.data['VF']['economic']['opex_per_kg'],
             'TF': self.data['TF']['economic']['opex_per_kg']},
            'OPEX_per_kg', lower_is_better=True
        )
        
        scores['economic']['Profit_Margin'] = self.normalize_alternatives(
            {'VF': self.data['VF']['economic']['profit_margin'],
             'TF': self.data['TF']['economic']['profit_margin']},
            'Profit_Margin', lower_is_better=False
        )
        
        # Environmental scores (lower is better for all impacts)
        for impact in ['GWP100', 'HOFP', 'PMFP', 'AP', 'EOFP', 'FFP']:
            scores['environmental'][impact] = self.normalize_alternatives(
                {'VF': self.data['VF']['environmental'][impact],
                 'TF': self.data['TF']['environmental'][impact]},
                impact, lower_is_better=True
            )
        
        # Operational scores
        scores['operational']['Production_Capacity'] = self.normalize_alternatives(
            {'VF': self.data['VF']['operational']['annual_output'],
             'TF': self.data['TF']['operational']['annual_output']},
            'Production_Capacity', lower_is_better=False
        )
        
        # Resource efficiency (lower resource use per kg is better)
        vf_resource_intensity = (self.data['VF']['operational']['electricity_per_kg'] + 
                                self.data['VF']['operational']['water_per_kg'] * 10)  # Weight water higher
        tf_resource_intensity = (self.data['TF']['operational']['electricity_per_kg'] + 
                                self.data['TF']['operational']['water_per_kg'] * 10)
        
        scores['operational']['Resource_Efficiency'] = self.normalize_alternatives(
            {'VF': vf_resource_intensity,
             'TF': tf_resource_intensity},
            'Resource_Efficiency', lower_is_better=True
        )
        
        scores['operational']['Space_Efficiency'] = self.normalize_alternatives(
            {'VF': self.data['VF']['operational']['space_efficiency'],
             'TF': self.data['TF']['operational']['space_efficiency']},
            'Space_Efficiency', lower_is_better=False
        )
        
        return scores
    
    def calculate_hierarchy(self) -> Dict:
        """Calculate complete AHP hierarchy with weights and scores"""
        
        print("\n" + "="*60)
        print("CALCULATING AHP HIERARCHY")
        print("="*60 + "\n")
        
        results = {
            'criteria': {},
            'subcriteria': {},
            'alternatives': {},
            'consistency': {}
        }
        
        # 1. Calculate main criteria weights
        print("1. Calculating main criteria weights...")
        criteria_def = self.define_criteria_pairwise_matrix()
        weights, lambda_max, CI, CR, is_consistent = AHPEngine.calculate_weights(
            criteria_def['matrix']
        )
        
        results['criteria']['weights'] = dict(zip(criteria_def['names'], weights))
        results['criteria']['consistency'] = {
            'lambda_max': lambda_max,
            'CI': CI,
            'CR': CR,
            'is_consistent': is_consistent
        }
        
        print(f"   Main Criteria Weights:")
        for name, weight in zip(criteria_def['names'], weights):
            print(f"   - {name}: {weight:.4f}")
        print(f"   Consistency Ratio (CR): {CR:.4f} {'✓ Consistent' if is_consistent else '✗ Inconsistent'}\n")
        
        # 2. Calculate economic sub-criteria weights
        print("2. Calculating economic sub-criteria weights...")
        econ_def = self.define_economic_subcriteria_matrix()
        weights, lambda_max, CI, CR, is_consistent = AHPEngine.calculate_weights(
            econ_def['matrix']
        )
        
        results['subcriteria']['economic'] = {
            'weights': dict(zip(econ_def['names'], weights)),
            'consistency': {'lambda_max': lambda_max, 'CI': CI, 'CR': CR, 'is_consistent': is_consistent}
        }
        
        print(f"   Economic Sub-criteria Weights:")
        for name, weight in zip(econ_def['names'], weights):
            print(f"   - {name}: {weight:.4f}")
        print(f"   Consistency Ratio (CR): {CR:.4f} {'✓ Consistent' if is_consistent else '✗ Inconsistent'}\n")
        
        # 3. Calculate environmental sub-criteria weights
        print("3. Calculating environmental sub-criteria weights...")
        env_def = self.define_environmental_subcriteria_matrix()
        weights, lambda_max, CI, CR, is_consistent = AHPEngine.calculate_weights(
            env_def['matrix']
        )
        
        results['subcriteria']['environmental'] = {
            'weights': dict(zip(env_def['names'], weights)),
            'consistency': {'lambda_max': lambda_max, 'CI': CI, 'CR': CR, 'is_consistent': is_consistent}
        }
        
        print(f"   Environmental Sub-criteria Weights:")
        for name, weight in zip(env_def['names'], weights):
            print(f"   - {name}: {weight:.4f}")
        print(f"   Consistency Ratio (CR): {CR:.4f} {'✓ Consistent' if is_consistent else '✗ Inconsistent'}\n")
        
        # 4. Calculate operational sub-criteria weights
        print("4. Calculating operational sub-criteria weights...")
        op_def = self.define_operational_subcriteria_matrix()
        weights, lambda_max, CI, CR, is_consistent = AHPEngine.calculate_weights(
            op_def['matrix']
        )
        
        results['subcriteria']['operational'] = {
            'weights': dict(zip(op_def['names'], weights)),
            'consistency': {'lambda_max': lambda_max, 'CI': CI, 'CR': CR, 'is_consistent': is_consistent}
        }
        
        print(f"   Operational Sub-criteria Weights:")
        for name, weight in zip(op_def['names'], weights):
            print(f"   - {name}: {weight:.4f}")
        print(f"   Consistency Ratio (CR): {CR:.4f} {'✓ Consistent' if is_consistent else '✗ Inconsistent'}\n")
        
        # 5. Calculate alternative scores
        print("5. Calculating alternative scores for each criterion...")
        results['alternatives'] = self.calculate_alternative_scores_for_criteria()
        print("   ✓ Alternative scores calculated\n")
        
        # 6. Calculate global weights and final scores
        print("6. Calculating global weights and final scores...")
        final_scores = self.calculate_final_scores(results)
        results['final_scores'] = final_scores
        
        print(f"\n{'='*60}")
        print(f"FINAL AHP SCORES")
        print(f"{'='*60}")
        print(f"Vertical Farming (VF): {final_scores['VF']:.4f}")
        print(f"Traditional Farming (TF): {final_scores['TF']:.4f}")
        print(f"\nRecommended Strategy: {final_scores['winner']}")
        print(f"Advantage: {final_scores['difference']:.2%}")
        print(f"{'='*60}\n")
        
        return results
    
    def calculate_final_scores(self, results: Dict) -> Dict:
        """Calculate final weighted scores for alternatives"""
        
        vf_score = 0
        tf_score = 0
        
        # Get main criteria weights
        econ_weight = results['criteria']['weights']['Economic']
        env_weight = results['criteria']['weights']['Environmental']
        op_weight = results['criteria']['weights']['Operational']
        
        # Economic contribution
        for subcriterion, weight in results['subcriteria']['economic']['weights'].items():
            global_weight = econ_weight * weight
            vf_score += global_weight * results['alternatives']['economic'][subcriterion]['VF']
            tf_score += global_weight * results['alternatives']['economic'][subcriterion]['TF']
        
        # Environmental contribution
        for subcriterion, weight in results['subcriteria']['environmental']['weights'].items():
            global_weight = env_weight * weight
            vf_score += global_weight * results['alternatives']['environmental'][subcriterion]['VF']
            tf_score += global_weight * results['alternatives']['environmental'][subcriterion]['TF']
        
        # Operational contribution
        for subcriterion, weight in results['subcriteria']['operational']['weights'].items():
            global_weight = op_weight * weight
            vf_score += global_weight * results['alternatives']['operational'][subcriterion]['VF']
            tf_score += global_weight * results['alternatives']['operational'][subcriterion]['TF']
        
        winner = 'Vertical Farming (VF)' if vf_score > tf_score else 'Traditional Farming (TF)'
        difference = abs(vf_score - tf_score)
        
        return {
            'VF': vf_score,
            'TF': tf_score,
            'winner': winner,
            'difference': difference
        }

# =============================================================================
# SECTION 4: SENSITIVITY ANALYSIS
# =============================================================================

class SensitivityAnalyzer:
    """Perform sensitivity analysis on AHP results"""
    
    def __init__(self, hierarchy_results: Dict, integrated_data: Dict):
        self.results = hierarchy_results
        self.data = integrated_data
        
    def criteria_weight_sensitivity(self, n_points: int = 20) -> Dict:
        """
        Analyze sensitivity to changes in main criteria weights
        
        Varies each criterion weight while adjusting others proportionally
        """
        
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSIS - CRITERIA WEIGHTS")
        print("="*60 + "\n")
        
        sensitivity_results = {
            'Economic': [],
            'Environmental': [],
            'Operational': []
        }
        
        base_weights = self.results['criteria']['weights']
        
        for criterion in ['Economic', 'Environmental', 'Operational']:
            print(f"Analyzing sensitivity to {criterion} weight...")
            
            # Vary weight from 0.1 to 0.8
            for weight in np.linspace(0.1, 0.8, n_points):
                # Adjust other weights proportionally
                remaining_weight = 1 - weight
                other_criteria = [c for c in base_weights.keys() if c != criterion]
                
                # Calculate proportional distribution
                other_base_sum = sum(base_weights[c] for c in other_criteria)
                adjusted_weights = {criterion: weight}
                
                for other in other_criteria:
                    if other_base_sum > 0:
                        adjusted_weights[other] = (base_weights[other] / other_base_sum) * remaining_weight
                    else:
                        adjusted_weights[other] = remaining_weight / len(other_criteria)
                
                # Recalculate scores with adjusted weights
                vf_score, tf_score = self._calculate_scores_with_weights(adjusted_weights)
                
                sensitivity_results[criterion].append({
                    'weight': weight,
                    'VF_score': vf_score,
                    'TF_score': tf_score,
                    'winner': 'VF' if vf_score > tf_score else 'TF'
                })
        
        print("✓ Sensitivity analysis complete\n")
        return sensitivity_results
    
    def _calculate_scores_with_weights(self, criteria_weights: Dict) -> Tuple[float, float]:
        """Helper function to calculate scores with custom criteria weights"""
        
        vf_score = 0
        tf_score = 0
        
        # Economic contribution
        econ_weight = criteria_weights['Economic']
        for subcriterion, weight in self.results['subcriteria']['economic']['weights'].items():
            global_weight = econ_weight * weight
            vf_score += global_weight * self.results['alternatives']['economic'][subcriterion]['VF']
            tf_score += global_weight * self.results['alternatives']['economic'][subcriterion]['TF']
        
        # Environmental contribution
        env_weight = criteria_weights['Environmental']
        for subcriterion, weight in self.results['subcriteria']['environmental']['weights'].items():
            global_weight = env_weight * weight
            vf_score += global_weight * self.results['alternatives']['environmental'][subcriterion]['VF']
            tf_score += global_weight * self.results['alternatives']['environmental'][subcriterion]['TF']
        
        # Operational contribution
        op_weight = criteria_weights['Operational']
        for subcriterion, weight in self.results['subcriteria']['operational']['weights'].items():
            global_weight = op_weight * weight
            vf_score += global_weight * self.results['alternatives']['operational'][subcriterion]['VF']
            tf_score += global_weight * self.results['alternatives']['operational'][subcriterion]['TF']
        
        return vf_score, tf_score
    
    def parameter_perturbation_analysis(self, perturbation_pct: float = 10) -> Dict:
        """
        Analyze impact of parameter perturbations on final scores
        
        Args:
            perturbation_pct: Percentage change to apply (default 10%)
        """
        
        print("\n" + "="*60)
        print("PARAMETER PERTURBATION ANALYSIS")
        print("="*60 + "\n")
        
        perturbation_results = {}
        base_vf_score = self.results['final_scores']['VF']
        base_tf_score = self.results['final_scores']['TF']
        
        # Test key economic parameters
        print(f"Testing ±{perturbation_pct}% perturbation in key parameters...\n")
        
        # Save original data
        import copy
        original_data = copy.deepcopy(self.data)
        
        parameters_to_test = [
            ('VF', 'economic', 'lcov', 'VF LCOv'),
            ('TF', 'economic', 'lcov', 'TF LCOv'),
            ('VF', 'environmental', 'GWP100', 'VF GWP100'),
            ('TF', 'environmental', 'GWP100', 'TF GWP100'),
        ]
        
        for system, category, param, label in parameters_to_test:
            results_list = []
            
            for direction in [-1, 1]:
                # Perturb parameter
                multiplier = 1 + (direction * perturbation_pct / 100)
                self.data[system][category][param] = original_data[system][category][param] * multiplier
                
                # Recalculate hierarchy
                temp_hierarchy = AHPHierarchy(self.data)
                temp_results = temp_hierarchy.calculate_hierarchy()
                
                results_list.append({
                    'direction': 'increase' if direction > 0 else 'decrease',
                    'multiplier': multiplier,
                    'VF_score': temp_results['final_scores']['VF'],
                    'TF_score': temp_results['final_scores']['TF'],
                    'VF_change': (temp_results['final_scores']['VF'] - base_vf_score) / base_vf_score * 100,
                    'TF_change': (temp_results['final_scores']['TF'] - base_tf_score) / base_tf_score * 100
                })
                
                # Restore original value
                self.data[system][category][param] = original_data[system][category][param]
            
            perturbation_results[label] = results_list
        
        # Restore original data
        self.data = original_data
        
        print("✓ Perturbation analysis complete\n")
        return perturbation_results

# =============================================================================
# SECTION 5: VISUALIZATION MODULE
# =============================================================================

class AHPVisualizer:
    """Create comprehensive visualizations for AHP results"""
    
    def __init__(self, hierarchy_results: Dict, integrated_data: Dict, 
                 sensitivity_results: Dict = None):
        self.results = hierarchy_results
        self.data = integrated_data
        self.sensitivity = sensitivity_results
        
    def plot_criteria_weights(self):
        """Plot main criteria weights"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        criteria = list(self.results['criteria']['weights'].keys())
        weights = list(self.results['criteria']['weights'].values())
        colors = ['#3498db', '#2ecc71', '#f39c12']
        
        bars = ax.barh(criteria, weights, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax.text(weight + 0.01, i, f'{weight:.3f}', 
                   va='center', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Weight', fontsize=12, fontweight='bold')
        ax.set_title('AHP Main Criteria Weights\n(Importance for Sustainable Farming in Singapore)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, max(weights) * 1.15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/criteria_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved: criteria_weights.png")
    
    def plot_subcriteria_weights(self):
        """Plot sub-criteria weights for all categories"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        categories = ['economic', 'environmental', 'operational']
        titles = ['Economic Sub-criteria', 'Environmental Sub-criteria', 'Operational Sub-criteria']
        colors_map = {
            'economic': ['#e74c3c', '#c0392b', '#e67e22', '#d35400'],
            'environmental': ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#3498db', '#2980b9'],
            'operational': ['#9b59b6', '#8e44ad', '#34495e']
        }
        
        for ax, category, title in zip(axes, categories, titles):
            weights_dict = self.results['subcriteria'][category]['weights']
            subcriteria = list(weights_dict.keys())
            weights = list(weights_dict.values())
            colors = colors_map[category]
            
            bars = ax.barh(subcriteria, weights, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for i, (bar, weight) in enumerate(zip(bars, weights)):
                ax.text(weight + 0.01, i, f'{weight:.3f}', 
                       va='center', fontweight='bold', fontsize=9)
            
            ax.set_xlabel('Weight', fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
            ax.set_xlim(0, max(weights) * 1.2)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/subcriteria_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved: subcriteria_weights.png")
    
    def plot_alternative_comparison(self):
        """Plot comparison of alternatives across all sub-criteria"""
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 16))
        
        categories = ['economic', 'environmental', 'operational']
        titles = ['Economic Performance Comparison', 
                 'Environmental Performance Comparison',
                 'Operational Performance Comparison']
        
        for ax, category, title in zip(axes, categories, titles):
            subcriteria = list(self.results['alternatives'][category].keys())
            vf_scores = [self.results['alternatives'][category][sc]['VF'] for sc in subcriteria]
            tf_scores = [self.results['alternatives'][category][sc]['TF'] for sc in subcriteria]
            
            x = np.arange(len(subcriteria))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, vf_scores, width, label='Vertical Farming', 
                          color='#3498db', alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x + width/2, tf_scores, width, label='Traditional Farming', 
                          color='#2ecc71', alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_ylabel('Normalized Score (higher is better)', fontsize=10, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(subcriteria, rotation=45, ha='right')
            ax.legend(fontsize=10, loc='upper right')
            ax.set_ylim(0, 1.15)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/alternative_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved: alternative_comparison.png")
    
    def plot_final_scores(self):
        """Plot final AHP scores"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        alternatives = ['Vertical\nFarming', 'Traditional\nFarming']
        scores = [self.results['final_scores']['VF'], self.results['final_scores']['TF']]
        colors = ['#3498db', '#2ecc71']
        
        bars = ax1.bar(alternatives, scores, color=colors, alpha=0.8, edgecolor='black', width=0.6)
        
        # Add value labels and winner indicator
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(i, score + 0.02, f'{score:.4f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=14)
            
            # Add star for winner
            if (i == 0 and scores[0] > scores[1]) or (i == 1 and scores[1] > scores[0]):
                ax1.text(i, score + 0.08, '★ Winner', 
                        ha='center', fontweight='bold', fontsize=12, color='gold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax1.set_ylabel('AHP Score', fontsize=12, fontweight='bold')
        ax1.set_title('Final AHP Scores', fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylim(0, max(scores) * 1.2)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Pie chart
        labels = [f'VF\n({scores[0]:.4f})', f'TF\n({scores[1]:.4f})']
        ax2.pie(scores, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
               explode=(0.05, 0.05), shadow=True)
        ax2.set_title('Score Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/final_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved: final_scores.png")
    
    def plot_hierarchy_tree(self):
        """Plot AHP decision hierarchy as a tree diagram"""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('off')
        
        # Goal
        goal_box = dict(boxstyle='round', facecolor='lightgray', edgecolor='black', linewidth=2)
        ax.text(0.5, 0.95, 'Goal: Sustainable Farming\nStrategy for Singapore', 
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=goal_box, transform=ax.transAxes)
        
        # Main criteria
        criteria_y = 0.75
        criteria_x = [0.2, 0.5, 0.8]
        criteria_names = ['Economic', 'Environmental', 'Operational']
        criteria_weights = [self.results['criteria']['weights'][c] for c in criteria_names]
        criteria_colors = ['#e74c3c', '#2ecc71', '#f39c12']
        
        for x, name, weight, color in zip(criteria_x, criteria_names, criteria_weights, criteria_colors):
            box = dict(boxstyle='round', facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
            ax.text(x, criteria_y, f'{name}\n(w={weight:.3f})', 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=box, transform=ax.transAxes)
            
            # Draw connection to goal
            ax.plot([0.5, x], [0.93, criteria_y + 0.02], 'k-', linewidth=1.5, alpha=0.5,
                   transform=ax.transAxes)
        
        # Sub-criteria
        subcriteria_y = 0.45
        
        # Economic sub-criteria
        econ_subcrit = list(self.results['subcriteria']['economic']['weights'].keys())
        econ_weights = list(self.results['subcriteria']['economic']['weights'].values())
        econ_x = np.linspace(0.05, 0.35, len(econ_subcrit))
        
        for x, name, weight in zip(econ_x, econ_subcrit, econ_weights):
            box = dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#e74c3c', linewidth=1)
            ax.text(x, subcriteria_y, f'{name}\n({weight:.3f})', 
                   ha='center', va='center', fontsize=7,
                   bbox=box, transform=ax.transAxes)
            ax.plot([criteria_x[0], x], [criteria_y - 0.02, subcriteria_y + 0.02], 
                   'k-', linewidth=0.8, alpha=0.3, transform=ax.transAxes)
        
        # Environmental sub-criteria
        env_subcrit = list(self.results['subcriteria']['environmental']['weights'].keys())
        env_weights = list(self.results['subcriteria']['environmental']['weights'].values())
        env_x = np.linspace(0.38, 0.62, len(env_subcrit))
        
        for x, name, weight in zip(env_x, env_subcrit, env_weights):
            box = dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#2ecc71', linewidth=1)
            ax.text(x, subcriteria_y, f'{name}\n({weight:.3f})', 
                   ha='center', va='center', fontsize=7,
                   bbox=box, transform=ax.transAxes)
            ax.plot([criteria_x[1], x], [criteria_y - 0.02, subcriteria_y + 0.02], 
                   'k-', linewidth=0.8, alpha=0.3, transform=ax.transAxes)
        
        # Operational sub-criteria
        op_subcrit = list(self.results['subcriteria']['operational']['weights'].keys())
        op_weights = list(self.results['subcriteria']['operational']['weights'].values())
        op_x = np.linspace(0.68, 0.92, len(op_subcrit))
        
        for x, name, weight in zip(op_x, op_subcrit, op_weights):
            box = dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#f39c12', linewidth=1)
            ax.text(x, subcriteria_y, f'{name}\n({weight:.3f})', 
                   ha='center', va='center', fontsize=7,
                   bbox=box, transform=ax.transAxes)
            ax.plot([criteria_x[2], x], [criteria_y - 0.02, subcriteria_y + 0.02], 
                   'k-', linewidth=0.8, alpha=0.3, transform=ax.transAxes)
        
        # Alternatives
        alt_y = 0.15
        alt_x = [0.35, 0.65]
        alt_names = ['Vertical\nFarming', 'Traditional\nFarming']
        alt_scores = [self.results['final_scores']['VF'], self.results['final_scores']['TF']]
        alt_colors = ['#3498db', '#2ecc71']
        
        for x, name, score, color in zip(alt_x, alt_names, alt_scores, alt_colors):
            box = dict(boxstyle='round', facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
            ax.text(x, alt_y, f'{name}\nScore: {score:.4f}', 
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=box, transform=ax.transAxes, color='white')
            
            # Draw connections to all sub-criteria (simplified - just show concept)
            for sub_x in list(econ_x) + list(env_x) + list(op_x):
                ax.plot([sub_x, x], [subcriteria_y - 0.02, alt_y + 0.04], 
                       'k-', linewidth=0.3, alpha=0.1, transform=ax.transAxes)
        
        ax.set_title('AHP Decision Hierarchy\nSustainable Farming Strategy Selection for Singapore', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/hierarchy_tree.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved: hierarchy_tree.png")
    
    def plot_sensitivity_analysis(self):
        """Plot sensitivity analysis results"""
        
        if self.sensitivity is None:
            print("⚠ No sensitivity analysis data available")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        criteria_list = ['Economic', 'Environmental', 'Operational']
        colors = ['#e74c3c', '#2ecc71', '#f39c12']
        
        for ax, criterion, color in zip(axes, criteria_list, colors):
            data = self.sensitivity[criterion]
            weights = [d['weight'] for d in data]
            vf_scores = [d['VF_score'] for d in data]
            tf_scores = [d['TF_score'] for d in data]
            
            ax.plot(weights, vf_scores, 'o-', label='Vertical Farming', 
                   color='#3498db', linewidth=2, markersize=6, alpha=0.8)
            ax.plot(weights, tf_scores, 's-', label='Traditional Farming', 
                   color='#2ecc71', linewidth=2, markersize=6, alpha=0.8)
            
            # Mark intersection points
            for i in range(len(weights) - 1):
                if (vf_scores[i] < tf_scores[i] and vf_scores[i+1] > tf_scores[i+1]) or \
                   (vf_scores[i] > tf_scores[i] and vf_scores[i+1] < tf_scores[i+1]):
                    # Approximate intersection
                    mid_weight = (weights[i] + weights[i+1]) / 2
                    mid_score = (vf_scores[i] + tf_scores[i]) / 2
                    ax.plot(mid_weight, mid_score, '*', color='red', 
                           markersize=15, label='Decision Flip' if i == 0 else '')
            
            ax.set_xlabel(f'{criterion} Weight', fontsize=11, fontweight='bold')
            ax.set_ylabel('Final AHP Score', fontsize=11, fontweight='bold')
            ax.set_title(f'Sensitivity to {criterion} Weight', fontsize=12, fontweight='bold', pad=15)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved: sensitivity_analysis.png")
    
    def plot_radar_comparison(self):
        """Create radar chart comparing VF and TF across criteria"""
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Combine all sub-criteria
        categories = []
        vf_values = []
        tf_values = []
        
        for main_cat in ['economic', 'environmental', 'operational']:
            for subcat in self.results['alternatives'][main_cat].keys():
                categories.append(subcat)
                vf_values.append(self.results['alternatives'][main_cat][subcat]['VF'])
                tf_values.append(self.results['alternatives'][main_cat][subcat]['TF'])
        
        # Number of variables
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        
        # Complete the circle
        vf_values += vf_values[:1]
        tf_values += tf_values[:1]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, vf_values, 'o-', linewidth=2, label='Vertical Farming', 
               color='#3498db', markersize=8, alpha=0.8)
        ax.fill(angles, vf_values, alpha=0.25, color='#3498db')
        
        ax.plot(angles, tf_values, 's-', linewidth=2, label='Traditional Farming', 
               color='#2ecc71', markersize=8, alpha=0.8)
        ax.fill(angles, tf_values, alpha=0.25, color='#2ecc71')
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)
        
        ax.set_title('Performance Radar Chart\n(Higher values = Better performance)', 
                    fontsize=14, fontweight='bold', pad=30)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved: radar_comparison.png")
    
    def create_all_plots(self):
        """Generate all visualization plots"""
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.plot_criteria_weights()
        self.plot_subcriteria_weights()
        self.plot_alternative_comparison()
        self.plot_final_scores()
        self.plot_hierarchy_tree()
        self.plot_radar_comparison()
        
        if self.sensitivity is not None:
            self.plot_sensitivity_analysis()
        
        print("\n✓ All visualizations generated successfully\n")

# =============================================================================
# SECTION 6: REPORT GENERATOR
# =============================================================================

class AHPReportGenerator:
    """Generate comprehensive Excel report of AHP results"""
    
    def __init__(self, hierarchy_results: Dict, integrated_data: Dict,
                 sensitivity_results: Dict = None):
        self.results = hierarchy_results
        self.data = integrated_data
        self.sensitivity = sensitivity_results
        
    def generate_excel_report(self, filename: str = 'ahp_comprehensive_report.xlsx'):
        """Generate comprehensive Excel report"""
        
        print("\n" + "="*60)
        print("GENERATING EXCEL REPORT")
        print("="*60 + "\n")
        
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. Executive Summary
            self._write_executive_summary(writer)
            
            # 2. Criteria Weights
            self._write_criteria_weights(writer)
            
            # 3. Sub-criteria Weights
            self._write_subcriteria_weights(writer)
            
            # 4. Alternative Scores
            self._write_alternative_scores(writer)
            
            # 5. Final Results
            self._write_final_results(writer)
            
            # 6. Consistency Analysis
            self._write_consistency_analysis(writer)
            
            # 7. Raw Data
            self._write_raw_data(writer)
            
            # 8. Sensitivity Analysis (if available)
            if self.sensitivity is not None:
                self._write_sensitivity_analysis(writer)
        
        print(f"✓ Excel report saved: {output_path}\n")
        return output_path
    
    def _write_executive_summary(self, writer):
        """Write executive summary sheet"""
        
        summary_data = {
            'Metric': [
                'Recommended Strategy',
                'Vertical Farming Score',
                'Traditional Farming Score',
                'Score Difference',
                'Confidence Level',
                '',
                'Key Findings',
                '- Primary Criterion',
                '- VF Advantage',
                '- TF Advantage',
                '',
                'Decision Consistency',
                '- Criteria Level CR',
                '- Economic CR',
                '- Environmental CR',
                '- Operational CR'
            ],
            'Value': [
                self.results['final_scores']['winner'],
                f"{self.results['final_scores']['VF']:.4f}",
                f"{self.results['final_scores']['TF']:.4f}",
                f"{self.results['final_scores']['difference']:.4f} ({self.results['final_scores']['difference']:.2%})",
                'High' if self.results['final_scores']['difference'] > 0.1 else 'Moderate',
                '',
                '',
                max(self.results['criteria']['weights'], key=self.results['criteria']['weights'].get),
                'Environmental performance, Space efficiency',
                'Lower costs, Simpler operations',
                '',
                '',
                f"{self.results['criteria']['consistency']['CR']:.4f}",
                f"{self.results['subcriteria']['economic']['consistency']['CR']:.4f}",
                f"{self.results['subcriteria']['environmental']['consistency']['CR']:.4f}",
                f"{self.results['subcriteria']['operational']['consistency']['CR']:.4f}"
            ]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name='Executive_Summary', index=False)
        print("✓ Written: Executive Summary")
    
    def _write_criteria_weights(self, writer):
        """Write main criteria weights"""
        
        df = pd.DataFrame({
            'Criterion': list(self.results['criteria']['weights'].keys()),
            'Weight': list(self.results['criteria']['weights'].values()),
            'Percentage': [f"{w*100:.2f}%" for w in self.results['criteria']['weights'].values()]
        })
        
        df.to_excel(writer, sheet_name='Criteria_Weights', index=False)
        print("✓ Written: Criteria Weights")
    
    def _write_subcriteria_weights(self, writer):
        """Write sub-criteria weights for all categories"""
        
        all_subcriteria = []
        
        for category in ['economic', 'environmental', 'operational']:
            weights_dict = self.results['subcriteria'][category]['weights']
            main_weight = self.results['criteria']['weights'][category.capitalize()]
            
            for subcriterion, weight in weights_dict.items():
                all_subcriteria.append({
                    'Main_Criterion': category.capitalize(),
                    'Sub_Criterion': subcriterion,
                    'Local_Weight': weight,
                    'Global_Weight': weight * main_weight,
                    'Percentage': f"{weight*100:.2f}%"
                })
        
        df = pd.DataFrame(all_subcriteria)
        df.to_excel(writer, sheet_name='Subcriteria_Weights', index=False)
        print("✓ Written: Sub-criteria Weights")
    
    def _write_alternative_scores(self, writer):
        """Write alternative scores for all sub-criteria"""
        
        all_scores = []
        
        for category in ['economic', 'environmental', 'operational']:
            for subcriterion, scores in self.results['alternatives'][category].items():
                all_scores.append({
                    'Category': category.capitalize(),
                    'Sub_Criterion': subcriterion,
                    'VF_Score': scores['VF'],
                    'TF_Score': scores['TF'],
                    'VF_Advantage': scores['VF'] - scores['TF'],
                    'Winner': 'VF' if scores['VF'] > scores['TF'] else 'TF'
                })
        
        df = pd.DataFrame(all_scores)
        df.to_excel(writer, sheet_name='Alternative_Scores', index=False)
        print("✓ Written: Alternative Scores")
    
    def _write_final_results(self, writer):
        """Write final AHP scores and decision"""
        
        df = pd.DataFrame({
            'Alternative': ['Vertical Farming', 'Traditional Farming'],
            'Final_Score': [self.results['final_scores']['VF'], 
                          self.results['final_scores']['TF']],
            'Percentage': [f"{self.results['final_scores']['VF']*100:.2f}%",
                         f"{self.results['final_scores']['TF']*100:.2f}%"],
            'Rank': [1 if self.results['final_scores']['VF'] > self.results['final_scores']['TF'] else 2,
                    2 if self.results['final_scores']['VF'] > self.results['final_scores']['TF'] else 1]
        })
        
        df.to_excel(writer, sheet_name='Final_Results', index=False)
        print("✓ Written: Final Results")
    
    def _write_consistency_analysis(self, writer):
        """Write consistency analysis for all matrices"""
        
        consistency_data = []
        
        # Main criteria
        consistency_data.append({
            'Matrix': 'Main Criteria',
            'Size': 3,
            'Lambda_Max': self.results['criteria']['consistency']['lambda_max'],
            'CI': self.results['criteria']['consistency']['CI'],
            'CR': self.results['criteria']['consistency']['CR'],
            'Is_Consistent': 'Yes' if self.results['criteria']['consistency']['is_consistent'] else 'No'
        })
        
        # Sub-criteria
        for category in ['economic', 'environmental', 'operational']:
            cons = self.results['subcriteria'][category]['consistency']
            size = len(self.results['subcriteria'][category]['weights'])
            
            consistency_data.append({
                'Matrix': f'{category.capitalize()} Sub-criteria',
                'Size': size,
                'Lambda_Max': cons['lambda_max'],
                'CI': cons['CI'],
                'CR': cons['CR'],
                'Is_Consistent': 'Yes' if cons['is_consistent'] else 'No'
            })
        
        df = pd.DataFrame(consistency_data)
        df.to_excel(writer, sheet_name='Consistency_Analysis', index=False)
        print("✓ Written: Consistency Analysis")
    
    def _write_raw_data(self, writer):
        """Write integrated raw data"""
        
        # Economic data
        econ_data = []
        for system in ['VF', 'TF']:
            for key, value in self.data[system]['economic'].items():
                econ_data.append({
                    'System': system,
                    'Parameter': key,
                    'Value': value
                })
        
        df_econ = pd.DataFrame(econ_data)
        df_econ.to_excel(writer, sheet_name='Raw_Economic_Data', index=False)
        
        # Environmental data
        env_data = []
        for system in ['VF', 'TF']:
            for key, value in self.data[system]['environmental'].items():
                env_data.append({
                    'System': system,
                    'Impact_Category': key,
                    'Value_per_kg': value
                })
        
        df_env = pd.DataFrame(env_data)
        df_env.to_excel(writer, sheet_name='Raw_Environmental_Data', index=False)
        
        # Operational data
        op_data = []
        for system in ['VF', 'TF']:
            for key, value in self.data[system]['operational'].items():
                op_data.append({
                    'System': system,
                    'Parameter': key,
                    'Value': value
                })
        
        df_op = pd.DataFrame(op_data)
        df_op.to_excel(writer, sheet_name='Raw_Operational_Data', index=False)
        
        print("✓ Written: Raw Data")
    
    def _write_sensitivity_analysis(self, writer):
        """Write sensitivity analysis results"""
        
        for criterion in ['Economic', 'Environmental', 'Operational']:
            data = self.sensitivity[criterion]
            
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=f'Sensitivity_{criterion}', index=False)
        
        print("✓ Written: Sensitivity Analysis")

# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print(" "*15 + "AHP ANALYSIS FOR SUSTAINABLE FARMING")
    print(" "*15 + "Vertical Farming vs Traditional Farming")
    print(" "*20 + "Singapore Context")
    print("="*70 + "\n")
    
    # File paths
    BASE_DATA = 'Corrected_Base_Data_Singapore.xlsx'
    VF_LCIA = 'VF_LCIA_ready_multiimpact.xlsx'
    TF_LCIA = 'TF_LCIA_ready_multiimpact.xlsx'
    
    try:
        # Step 1: Integrate all data
        print("STEP 1: Data Integration")
        print("-" * 60)
        integrator = DataIntegrator(BASE_DATA, VF_LCIA, TF_LCIA)
        integrated_data = integrator.integrate_all_data()
        
        # Step 2: Build AHP hierarchy and calculate
        print("STEP 2: AHP Hierarchy Calculation")
        print("-" * 60)
        hierarchy = AHPHierarchy(integrated_data)
        results = hierarchy.calculate_hierarchy()
        
        # Step 3: Sensitivity Analysis
        print("STEP 3: Sensitivity Analysis")
        print("-" * 60)
        sensitivity_analyzer = SensitivityAnalyzer(results, integrated_data)
        sensitivity_results = sensitivity_analyzer.criteria_weight_sensitivity()
        
        # Step 4: Generate Visualizations
        print("STEP 4: Generating Visualizations")
        print("-" * 60)
        visualizer = AHPVisualizer(results, integrated_data, sensitivity_results)
        visualizer.create_all_plots()
        
        # Step 5: Generate Excel Report
        print("STEP 5: Generating Excel Report")
        print("-" * 60)
        report_generator = AHPReportGenerator(results, integrated_data, sensitivity_results)
        report_path = report_generator.generate_excel_report()
        
        # Final Summary
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\n📊 Results Summary:")
        print(f"   Winner: {results['final_scores']['winner']}")
        print(f"   VF Score: {results['final_scores']['VF']:.4f}")
        print(f"   TF Score: {results['final_scores']['TF']:.4f}")
        print(f"   Difference: {results['final_scores']['difference']:.2%}")
        print(f"\n📁 Output Directory: {OUTPUT_DIR}/")
        print(f"   - Excel Report: {report_path}")
        print(f"   - Visualizations: 7 PNG files")
        print(f"\n✓ All analysis files generated successfully!")
        print("="*70 + "\n")
        
        return results, integrated_data, sensitivity_results
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    results, data, sensitivity = main()
