"""
Decision Tree Classifier Educational Tool

This tool demonstrates how a Decision Tree Classifier works by:
1. Loading a dataset (Play Tennis by default)
2. Calculating entropy and information gain at each step
3. Building a decision tree recursively
4. Visualizing the tree building process
5. Providing detailed intermediate calculations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import networkx as nx
import math
import os
import time
from IPython.display import display, clear_output
import graphviz

class DecisionTreeNode:
    """A node in the Decision Tree"""
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None, samples=None, entropy=None):
        self.feature = feature  # Feature used for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.value = value  # Class prediction (for leaf nodes)
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.samples = samples  # Sample count at this node
        self.entropy = entropy  # Entropy at this node
        self.info_gain = None  # Information gain achieved by this split
        
    def is_leaf(self):
        """Check if the node is a leaf node"""
        return self.value is not None

class EntropyCalculator:
    """Calculates entropy and information gain"""
    
    @staticmethod
    def calculate_entropy(y):
        """
        Calculate the entropy of a target variable array
        
        Args:
            y: array-like, target variable
            
        Returns:
            float: entropy value
        """
        # Count unique classes and their frequencies
        classes, counts = np.unique(y, return_counts=True)
        # Calculate probabilities
        probabilities = counts / len(y)
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Show detailed calculation
        print(f"Entropy calculation:")
        print(f"  Classes: {classes}")
        print(f"  Counts: {counts}")
        print(f"  Probabilities: {probabilities}")
        print(f"  Entropy = -sum(p * log2(p)) = {entropy:.4f}")
        
        return entropy
    
    @staticmethod
    def calculate_information_gain(y, feature_values, feature_name):
        """
        Calculate information gain for a feature
        
        Args:
            y: array-like, target variable
            feature_values: array-like, feature values
            feature_name: str, name of the feature
            
        Returns:
            float: information gain
        """
        # Calculate entropy before split
        entropy_before = EntropyCalculator.calculate_entropy(y)
        
        # Get unique values for the feature
        unique_values = np.unique(feature_values)
        
        # Initialize variables
        weighted_entropy = 0
        split_details = []
        
        # Calculate weighted entropy after split
        for value in unique_values:
            # Get indices where feature has the current value
            indices = np.where(feature_values == value)[0]
            
            # Get corresponding target values
            subset_y = y[indices]
            
            # Calculate weight (proportion of samples with this feature value)
            weight = len(subset_y) / len(y)
            
            # Calculate entropy for this subset
            subset_entropy = EntropyCalculator.calculate_entropy(subset_y)
            
            # Add weighted entropy
            weighted_entropy += weight * subset_entropy
            
            # Store details for display
            split_details.append({
                'value': value,
                'count': len(subset_y),
                'entropy': subset_entropy,
                'weight': weight,
                'weighted_entropy': weight * subset_entropy
            })
        
        # Calculate information gain
        information_gain = entropy_before - weighted_entropy
        
        # Show detailed calculation
        print(f"\nInformation Gain calculation for feature '{feature_name}':")
        print(f"  Entropy before split: {entropy_before:.4f}")
        print("  Split details:")
        for detail in split_details:
            print(f"    Value={detail['value']}: count={detail['count']}, "
                  f"entropy={detail['entropy']:.4f}, weight={detail['weight']:.4f}, "
                  f"weighted_entropy={detail['weighted_entropy']:.4f}")
        print(f"  Weighted entropy after split: {weighted_entropy:.4f}")
        print(f"  Information Gain = {information_gain:.4f}")
        
        return information_gain

class DecisionTreeBuilder:
    """Builds a Decision Tree classifier with step-by-step visualization"""
    
    def __init__(self, min_samples_split=2):
        """
        Initialize the Decision Tree Builder
        
        Args:
            min_samples_split: minimum samples required to split a node
        """
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
        self.tree_nodes = []  # For visualization
        self.tree_edges = []  # For visualization
        self.node_count = 0
        self.step = 0
        self.vis = None  # Graphviz visualization
    
    def fit(self, X, y, feature_names=None, display_steps=True):
        """
        Build the decision tree
        
        Args:
            X: array-like of shape (n_samples, n_features), input features
            y: array-like of shape (n_samples,), target values
            feature_names: list of feature names
            display_steps: whether to display steps during training
            
        Returns:
            self: The fitted decision tree
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Store feature names
        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Reset visualization
        self.tree_nodes = []
        self.tree_edges = []
        self.node_count = 0
        self.step = 0
        self.vis = graphviz.Digraph(format='png')
        
        # Build tree recursively
        print("Starting Decision Tree Construction...\n")
        time.sleep(1)
        
        self.root = self._build_tree(X, y, depth=0, parent_id=None, display_steps=display_steps)
        
        print("\nDecision Tree Construction Complete!")
        return self
    
    def _build_tree(self, X, y, depth=0, parent_id=None, branch_value=None, display_steps=True):
        """
        Recursively build the decision tree
        
        Args:
            X: input features
            y: target values
            depth: current depth of the tree
            parent_id: ID of the parent node (for visualization)
            branch_value: value of the branch from parent to this node
            display_steps: whether to display steps
            
        Returns:
            node: The built decision tree node
        """
        n_samples = len(y)
        
        # Calculate entropy of current node
        current_entropy = EntropyCalculator.calculate_entropy(y)
        
        # Create node
        node = DecisionTreeNode(samples=n_samples, entropy=current_entropy)
        
        # Step counter
        self.step += 1
        step_id = self.step
        
        # Generate unique node ID for visualization
        node_id = self.node_count
        self.node_count += 1
        
        # Display step heading
        if display_steps:
            print(f"\n{'='*80}")
            print(f"STEP {step_id}: Building node at depth {depth}")
            print(f"{'='*80}")
            print(f"Current samples: {n_samples}")
            print(f"Class distribution: {np.unique(y, return_counts=True)}")
            print(f"Node entropy: {current_entropy:.4f}")
            time.sleep(0.5)
        
        # Check if we should make this a leaf node
        classes, counts = np.unique(y, return_counts=True)
        
        # If all samples belong to one class or min_samples_split not met
        if len(classes) == 1 or n_samples < self.min_samples_split:
            majority_class = classes[np.argmax(counts)]
            node.value = majority_class
            
            # Add to visualization
            node_label = f"Leaf: {majority_class}\nSamples: {n_samples}\nEntropy: {current_entropy:.4f}"
            self.tree_nodes.append((node_id, {'label': node_label, 'type': 'leaf', 'class': str(majority_class)}))
            
            if parent_id is not None:
                self.tree_edges.append((parent_id, node_id, {'label': branch_value}))
            
            if display_steps:
                print(f"\nCreating leaf node with value: {majority_class}")
                self._update_visualization(highlight_node=node_id)
                time.sleep(1)
            
            return node
        
        # Find the best feature to split on
        best_feature_idx, best_threshold, best_gain = self._find_best_split(X, y, self.feature_names, display_steps)
        
        if best_gain <= 0:
            # If no good split is found, create a leaf node
            majority_class = classes[np.argmax(counts)]
            node.value = majority_class
            
            # Add to visualization
            node_label = f"Leaf: {majority_class}\nSamples: {n_samples}\nEntropy: {current_entropy:.4f}"
            self.tree_nodes.append((node_id, {'label': node_label, 'type': 'leaf', 'class': str(majority_class)}))
            
            if parent_id is not None:
                self.tree_edges.append((parent_id, node_id, {'label': branch_value}))
            
            if display_steps:
                print(f"\nNo useful split found. Creating leaf node with value: {majority_class}")
                self._update_visualization(highlight_node=node_id)
                time.sleep(1)
            
            return node
        
        # Set node attributes
        node.feature = best_feature_idx
        node.threshold = best_threshold
        node.info_gain = best_gain
        
        # Add to visualization
        feature_name = self.feature_names[best_feature_idx]
        node_label = f"Feature: {feature_name}\nGain: {best_gain:.4f}\nSamples: {n_samples}\nEntropy: {current_entropy:.4f}"
        self.tree_nodes.append((node_id, {'label': node_label, 'type': 'split', 'feature': feature_name}))
        
        if parent_id is not None:
            self.tree_edges.append((parent_id, node_id, {'label': branch_value}))
        
        if display_steps:
            print(f"\nCreating split node on feature: {feature_name}, Gain: {best_gain:.4f}")
            self._update_visualization(highlight_node=node_id)
            time.sleep(1)
        
        # Split data
        feature_values = X[:, best_feature_idx]
        unique_values = np.unique(feature_values)
        
        # For each value of the feature, create a child node
        for value in unique_values:
            # Get indices where feature has this value
            indices = np.where(feature_values == value)[0]
            
            # Skip if no samples with this value
            if len(indices) == 0:
                continue
            
            # Create subset of data
            X_subset = X[indices]
            y_subset = y[indices]
            
            if display_steps:
                print(f"\nSplitting on {feature_name} = {value}")
                print(f"  Creating child node with {len(indices)} samples")
                time.sleep(0.5)
            
            # Recursively build child node
            child_node = self._build_tree(
                X_subset, y_subset, 
                depth=depth+1, 
                parent_id=node_id, 
                branch_value=f"{feature_name}={value}", 
                display_steps=display_steps
            )
            
            # Add child to parent node
            if node.left is None:
                node.left = child_node
            else:
                node.right = child_node
        
        return node
    
    def _find_best_split(self, X, y, feature_names, display_steps=True):
        """
        Find the best feature to split on
        
        Args:
            X: input features
            y: target values
            feature_names: list of feature names
            display_steps: whether to display steps
            
        Returns:
            tuple: (best_feature_idx, best_threshold, best_gain)
        """
        n_features = X.shape[1]
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        
        if display_steps:
            print("\nFinding best feature to split on...")
            time.sleep(0.5)
        
        # Calculate information gain for each feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            feature_name = feature_names[feature_idx]
            
            if display_steps:
                print(f"\nEvaluating feature: {feature_name}")
                time.sleep(0.5)
            
            # Calculate information gain
            gain = EntropyCalculator.calculate_information_gain(y, feature_values, feature_name)
            
            # Update best feature if this one has higher gain
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_threshold = None  # Categorical features don't have thresholds
        
        if display_steps:
            print(f"\nBest split: Feature={feature_names[best_feature_idx]}, Gain={best_gain:.4f}")
            time.sleep(0.5)
        
        return best_feature_idx, best_threshold, best_gain
    
    def _update_visualization(self, highlight_node=None):
        """Update and display the current tree visualization"""
        # Create visualization of current tree state
        vis = graphviz.Digraph(format='png')
        
        # Add nodes
        for node_id, node_attrs in self.tree_nodes:
            node_label = node_attrs['label']
            node_type = node_attrs.get('type', 'unknown')
            
            # Determine node style based on type and if it's highlighted
            if node_id == highlight_node:
                if node_type == 'leaf':
                    vis.node(str(node_id), node_label, style='filled', color='lightblue', shape='box')
                else:
                    vis.node(str(node_id), node_label, style='filled', color='lightblue', shape='ellipse')
            else:
                if node_type == 'leaf':
                    vis.node(str(node_id), node_label, shape='box')
                else:
                    vis.node(str(node_id), node_label, shape='ellipse')
        
        # Add edges
        for src, dst, edge_attrs in self.tree_edges:
            vis.edge(str(src), str(dst), label=edge_attrs.get('label', ''))
        
        # Render and display
        vis.render('decision_tree_visualization', view=False, cleanup=True)
        
        # Display image
        try:
            from IPython.display import Image, display
            display(Image('decision_tree_visualization.png'))
        except:
            print("Visualization created but cannot be displayed in this environment.")
            print("Check decision_tree_visualization.png file.")
    
    def predict(self, X):
        """
        Predict class for samples in X
        
        Args:
            X: array-like of shape (n_samples, n_features), input features
            
        Returns:
            array: Predicted classes
        """
        X = np.array(X)
        predictions = []
        
        for sample in X:
            predictions.append(self._predict_sample(sample, self.root))
        
        return np.array(predictions)
    
    def _predict_sample(self, sample, node):
        """
        Predict class for a single sample
        
        Args:
            sample: array-like, single sample features
            node: current node in the decision tree
            
        Returns:
            class prediction
        """
        # If leaf node, return value
        if node.is_leaf():
            return node.value
        
        # Get feature value
        feature_value = sample[node.feature]
        
        # Find child node with matching feature value
        # Here we're using a categorical approach - check left node first, then right
        if node.left and self._match_feature(feature_value, node.left):
            return self._predict_sample(sample, node.left)
        elif node.right and self._match_feature(feature_value, node.right):
            return self._predict_sample(sample, node.right)
        else:
            # If no match found (shouldn't happen with proper training data)
            # return the majority class at this node
            return node.value
    
    def _match_feature(self, feature_value, child_node):
        """Helper method to match a sample's feature value with a child node"""
        # Simplified approach for categorical data
        return True  # For this educational version, we assume proper branch navigation

    def visualize_tree(self, output_file='decision_tree_final'):
        """
        Generate a final visualization of the decision tree
        
        Args:
            output_file: name of the output file (without extension)
            
        Returns:
            None
        """
        vis = graphviz.Digraph(format='png')
        
        def add_node(node, node_id=0, parent_id=None, branch_value=None):
            if node is None:
                return
            
            # Create node label
            if node.is_leaf():
                label = f"Prediction: {node.value}\nSamples: {node.samples}\nEntropy: {node.entropy:.4f}"
                vis.node(str(node_id), label, shape='box')
            else:
                feature_name = self.feature_names[node.feature]
                label = f"Feature: {feature_name}\nGain: {node.info_gain:.4f}\nSamples: {node.samples}\nEntropy: {node.entropy:.4f}"
                vis.node(str(node_id), label)
            
            # Add edge from parent
            if parent_id is not None:
                vis.edge(str(parent_id), str(node_id), label=branch_value)
            
            # Add child nodes
            if not node.is_leaf():
                feature_name = self.feature_names[node.feature]
                # For this educational version with categorical features:
                if node.left:
                    # Assume the left branch corresponds to some value
                    # (In a real implementation, we'd store the value in the node)
                    add_node(node.left, node_id*2+1, node_id, f"{feature_name}=value1")
                if node.right:
                    # Same for right branch
                    add_node(node.right, node_id*2+2, node_id, f"{feature_name}=value2")
        
        # Build the visualization
        add_node(self.root)
        
        # Render the tree
        vis.render(output_file, view=False, cleanup=True)
        print(f"Final tree visualization saved to {output_file}.png")

class DecisionTreeSimulator:
    """Main class for the Decision Tree Simulator"""
    
    def __init__(self):
        """Initialize the simulator"""
        self.tree_builder = DecisionTreeBuilder(min_samples_split=2)
        self.data = None
        self.feature_names = None
        self.target_name = None
    
    def load_play_tennis_dataset(self):
        """Load the Play Tennis dataset"""
        # Create the Play Tennis dataset
        data = {
            'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                        'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
            'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                           'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
            'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                        'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
            'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
                    'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
            'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                    'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
        }
        
        self.data = pd.DataFrame(data)
        self.feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']
        self.target_name = 'Play'
        
        print("Play Tennis dataset loaded:")
        print(self.data)
        
        return self.data
    
    def load_dataset_from_csv(self, file_path, target_column):
        """
        Load dataset from a CSV file
        
        Args:
            file_path: path to the CSV file
            target_column: name of the target column
            
        Returns:
            pandas.DataFrame: loaded dataset
        """
        self.data = pd.read_csv(file_path)
        self.target_name = target_column
        self.feature_names = [col for col in self.data.columns if col != target_column]
        
        print(f"Dataset loaded from {file_path}:")
        print(self.data.head())
        
        return self.data
    
    def encode_categorical_features(self):
        """
        Encode categorical features to numerical values
        
        Returns:
            tuple: (X, y) encoded features and target
        """
        # For this educational tool, we'll use a simple label encoding
        data_encoded = self.data.copy()
        
        # Encode all categorical columns
        for column in data_encoded.columns:
            if data_encoded[column].dtype == 'object':
                # Get unique values and map them to integers
                unique_values = data_encoded[column].unique()
                value_to_int = {value: i for i, value in enumerate(unique_values)}
                
                # Replace values with integers
                data_encoded[column] = data_encoded[column].map(value_to_int)
                
                # Print mapping for reference
                print(f"Encoding for {column}: {value_to_int}")
        
        # Split into features and target
        X = data_encoded[self.feature_names].values
        y = data_encoded[self.target_name].values
        
        return X, y
    
    def run_simulation(self, display_steps=True):
        """
        Run the decision tree construction simulation
        
        Args:
            display_steps: whether to display intermediate steps
            
        Returns:
            DecisionTreeBuilder: the trained decision tree
        """
        if self.data is None:
            print("No dataset loaded. Loading default Play Tennis dataset...")
            self.load_play_tennis_dataset()
        
        # Encode features
        X, y = self.encode_categorical_features()
        
        # Train decision tree
        print("\nStarting Decision Tree simulation...")
        self.tree_builder.fit(X, y, feature_names=self.feature_names, display_steps=display_steps)
        
        # Visualize final tree
        self.tree_builder.visualize_tree()
        
        return self.tree_builder
    
    def predict(self, new_data):
        """
        Make predictions on new data
        
        Args:
            new_data: DataFrame of new samples
            
        Returns:
            array: Predicted classes
        """
        # Ensure tree has been built
        if self.tree_builder.root is None:
            print("Decision Tree has not been trained yet. Run simulation first.")
            return None
        
        # Encode new data
        encoded_data = new_data.copy()
        
        # Encode using the same mappings
        for column in encoded_data.columns:
            if column in self.data.columns and self.data[column].dtype == 'object':
                # Get unique values from training data
                unique_values = self.data[column].unique()
                value_to_int = {value: i for i, value in enumerate(unique_values)}
                
                # Replace values with integers
                encoded_data[column] = encoded_data[column].map(value_to_int)
        
        # Get features
        X_new = encoded_data[self.feature_names].values
        
        # Make predictions
        predictions = self.tree_builder.predict(X_new)
        
        # Convert predictions back to original values
        unique_targets = self.data[self.target_name].unique()
        int_to_target = {i: value for i, value in enumerate(unique_targets)}
        predictions_decoded = [int_to_target[pred] for pred in predictions]
        
        return predictions_decoded

# Example usage
if __name__ == "__main__":
    # Create simulator
    simulator = DecisionTreeSimulator()
    
    # Load Play Tennis dataset
    simulator.load_play_tennis_dataset()
    
    # Run simulation
    tree = simulator.run_simulation(display_steps=True)
    
    # Make predictions on new data
    new_data = pd.DataFrame({
        'Outlook': ['Sunny', 'Overcast'],
        'Temperature': ['Cool', 'Mild'],
        'Humidity': ['Normal', 'High'],
        'Wind': ['Strong', 'Weak']
    })
    
    print("\nMaking predictions on new data:")
    print(new_data)
    
    predictions = simulator.predict(new_data)
    print(f"Predictions: {predictions}")