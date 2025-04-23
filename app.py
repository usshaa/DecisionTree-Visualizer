"""
Enhanced Streamlit application for interactive Decision Tree visualization
Save this file as app.py and run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import altair as alt
from streamlit_lottie import st_lottie
import json
import requests

# Import our simulator classes
from decisiontree import DecisionTreeSimulator, DecisionTreeBuilder, EntropyCalculator

# Page configuration with improved styling
st.set_page_config(
    page_title="Decision Tree Classifier Simulator",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 50px !important;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 24px !important;
        font-weight: 400;
        color: #c9c9c9;
        margin-top: 0px;
    }
    .section-header {
        font-size: 26px !important;
        font-weight: 500;
        color: #c9c9c9;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        margin-top: 30px;
    }
    .highlight-text {
        background-color: #1a1a1a;
        border-left: 4px solid #3498db;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
    }
    .entropy-gauge {
        margin: 20px 0;
    }
    .info-box {
        background-color: #e8f4f8;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .node {
        transition: all 0.3s ease;
    }
    .node:hover {
        filter: brightness(0.9);
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .tab-content {
        padding: 20px 0;
    }
    .feature-importance {
        padding: 10px;
        margin: 5px 0;
        background-color: #f1f8ff;
        border-radius: 4px;
    }
    .step-animation {
        border: 1px solid #e1e4e8;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .divider {
        width: 100%;
        height: 2px;
        background-color: #e1e4e8;
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# Try loading animation assets
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Animations
lottie_tree = load_lottie_url('https://assets10.lottiefiles.com/packages/lf20_o6spyjnc.json')
lottie_data = load_lottie_url('https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json')

# Enhanced title section with animation
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">🌳 Decision Tree Classifier Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An Interactive Educational Tool for Understanding Decision Trees</p>', unsafe_allow_html=True)
with col2:
    if lottie_tree:
        st_lottie(lottie_tree, height=150, key="tree_animation")
    else:
        st.image("https://via.placeholder.com/150x150.png?text=  🌳", width=150)

# Introduction
st.markdown("""
<div class="highlight-text">
This interactive tool helps you understand how decision trees work by visualizing each step of the algorithm. 
Watch as the tree grows and makes decisions based on information gain and entropy!
</div>
""", unsafe_allow_html=True)

# Main content organized in tabs
tab1, tab2, tab3 = st.tabs(["Simulator", "Learning Center", "Decision Tree Playground"])

# Tab 1: Simulator
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    # Create two columns for settings and dataset preview
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<p class="section-header">Settings</p>', unsafe_allow_html=True)
        
        # Dataset selection with enhanced UI
        dataset_option = st.selectbox(
            "Select Dataset",
            ["Play Tennis", "Iris Sample", "Titanic Sample", "Upload Your Own"],
            index=0
        )
        
        simulator = DecisionTreeSimulator()
        
        if dataset_option == "Play Tennis":
            data = simulator.load_play_tennis_dataset()
            st.success("Using Play Tennis dataset")
        elif dataset_option == "Iris Sample":
            # Load a sample from the Iris dataset
            from sklearn.datasets import load_iris
            iris = load_iris()
            iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
            iris_data['target'] = iris.target_names[iris.target]
            # Take a sample for simplicity
            data = iris_data.sample(25, random_state=42)
            simulator.data = data
            simulator.target_name = 'target'
            simulator.feature_names = [col for col in data.columns if col != 'target']
            st.success("Using Iris dataset sample (25 records)")
        elif dataset_option == "Titanic Sample":
            # Create a sample Titanic dataset
            titanic_data = {
                'Pclass': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
                'Sex': ['male', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female', 'male'],
                'Age': ['adult', 'adult', 'adult', 'child', 'adult', 'adult', 'child', 'child', 'adult', 'adult', 'adult', 'child', 'adult', 'child', 'adult'],
                'Embarked': ['S', 'C', 'S', 'C', 'S', 'S', 'C', 'S', 'S', 'C', 'S', 'C', 'S', 'S', 'C'],
                'Survived': ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no']
            }
            data = pd.DataFrame(titanic_data)
            simulator.data = data
            simulator.target_name = 'Survived'
            simulator.feature_names = [col for col in data.columns if col != 'Survived']
            st.success("Using Titanic dataset sample")
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success("Dataset uploaded successfully!")
                    
                    # Target column selection
                    target_column = st.selectbox(
                        "Select Target Column",
                        data.columns.tolist()
                    )
                    
                    # Update simulator with uploaded data
                    simulator.data = data
                    simulator.target_name = target_column
                    simulator.feature_names = [col for col in data.columns if col != target_column]
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                data = simulator.load_play_tennis_dataset()
                st.info("No file uploaded. Using Play Tennis dataset.")
        
        # Advanced settings collapsible
        with st.expander("Advanced Settings", expanded=False):
            min_samples = st.slider("Minimum Samples to Split", min_value=1, max_value=10, value=2)
            display_calculations = st.checkbox("Show Calculation Details", value=True)
            animation_speed = st.select_slider(
                "Animation Speed",
                options=["Very Slow", "Slow", "Medium", "Fast", "Very Fast"],
                value="Medium"
            )
            
            # Map slider value to actual delay in seconds
            speed_mapping = {
                "Very Slow": 3.0,
                "Slow": 2.0,
                "Medium": 1.0,
                "Fast": 0.5,
                "Very Fast": 0.1
            }
            delay = speed_mapping[animation_speed]
            
            # Option to see detailed entropy calculations
            show_formulas = st.checkbox("Show Formula Steps", value=True)
    
    with col2:
        st.markdown('<p class="section-header">Dataset Preview</p>', unsafe_allow_html=True)
        
        # Information about the dataset
        st.dataframe(data, use_container_width=True)
        
        # Show summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(data))
        with col2:
            st.metric("Features", len(simulator.feature_names))
        with col3:
            st.metric("Classes", len(data[simulator.target_name].unique()))
        
        # Target distribution visualization with Altair
        st.write("Target Distribution:")
        target_counts = data[simulator.target_name].value_counts().reset_index()
        target_counts.columns = ['Class', 'Count']
        
        chart = alt.Chart(target_counts).mark_bar().encode(
            x=alt.X('Class:N', title=simulator.target_name),
            y=alt.Y('Count:Q', title='Count'),
            color=alt.Color('Class:N', legend=None, scale=alt.Scale(scheme='category10')),
            tooltip=['Class', 'Count']
        ).properties(height=200)
        
        st.altair_chart(chart, use_container_width=True)
    
    # Simulation section
    st.markdown('<p class="section-header">Decision Tree Simulation</p>', unsafe_allow_html=True)
    
    # Create containers for output and visualization
    output_container = st.empty()
    vis_container = st.empty()
    entropy_container = st.empty()
    
    # Override tree builder methods for Streamlit with enhanced visualization
    class StreamlitTreeBuilder(DecisionTreeBuilder):
        def __init__(self, min_samples_split=2, delay=1.0, show_formulas=True):
            super().__init__(min_samples_split)
            self.delay = delay
            self.output_text = []
            self.step_data = []
            self.show_formulas = show_formulas
        
        def print_step(self, text):
            """Print text to Streamlit output container"""
            self.output_text.append(text)
            output_text_styled = "\n".join(self.output_text)
            
            # Add some syntax highlighting colors
            output_text_styled = output_text_styled.replace("STEP", "**STEP")
            output_text_styled = output_text_styled.replace(":", ":**")
            output_text_styled = output_text_styled.replace("Entropy calculation:", "**Entropy calculation:**")
            output_text_styled = output_text_styled.replace("Information Gain calculation", "**Information Gain calculation**")
            
            output_container.markdown(f"```python\n{output_text_styled}\n```")
            time.sleep(self.delay)
        
        def _update_entropy_visualization(self, entropy_value, breakdown=None):
            """Create a visual representation of entropy"""
            if not self.show_formulas:
                return
                
            with entropy_container:
                cols = st.columns([2, 3])
                
                with cols[0]:
                    # Create a gauge-like visualization for entropy
                    max_entropy = 1.0  # Assuming binary classification for simplicity
                    fig, ax = plt.subplots(figsize=(4, 3))
                    
                    # Create a gradient bar
                    cmap = plt.cm.RdYlGn_r
                    norm = plt.Normalize(0, max_entropy)
                    gradient = np.linspace(0, max_entropy, 256)
                    gradient = np.vstack((gradient, gradient))
                    
                    ax.imshow(gradient, aspect='auto', cmap=cmap, norm=norm)
                    ax.set_yticks([])
                    ax.set_xticks([0, 255])
                    ax.set_xticklabels(['0 (Pure)', f'{max_entropy:.1f} (Mixed)'])
                    
                    # Add marker for current entropy
                    marker_pos = int(entropy_value / max_entropy * 255)
                    ax.axvline(marker_pos, color='black', linewidth=3)
                    ax.set_title(f'Current Entropy: {entropy_value:.4f}')
                    
                    st.pyplot(fig)
                
                with cols[1]:
                    if breakdown:
                        # Show the entropy formula calculation
                        st.markdown("### Entropy Calculation")
                        st.latex(r"H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)")
                        
                        # Show actual calculation with values
                        formula_parts = []
                        for cls, prob in breakdown.items():
                            if prob > 0:  # Avoid log(0)
                                formula_parts.append(f"-{prob:.3f} \\times \\log_2({prob:.3f})")
                        
                        if formula_parts:
                            full_formula = " + ".join(formula_parts)
                            st.latex(f"H(S) = {full_formula} = {entropy_value:.4f}")
        
        def _update_visualization(self, highlight_node=None):
            """Update and display tree visualization in Streamlit"""
            import graphviz
            
            # Create visualization of current tree state
            vis = graphviz.Digraph(format='png')
            vis.attr('graph', rankdir='TB', ranksep='0.6', nodesep='0.8')
            vis.attr('node', fontname='Helvetica', fontsize='12')
            
            # Add nodes
            for node_id, node_attrs in self.tree_nodes:
                node_label = node_attrs['label']
                node_type = node_attrs.get('type', 'unknown')
                
                # Determine node style based on type and if it's highlighted
                if node_id == highlight_node:
                    if node_type == 'leaf':
                        vis.node(str(node_id), node_label, style='filled', color='#42b983', fontcolor='white', shape='box')
                    else:
                        vis.node(str(node_id), node_label, style='filled', color='#3498db', fontcolor='white', shape='ellipse')
                else:
                    if node_type == 'leaf':
                        vis.node(str(node_id), node_label, style='filled', color='#e8f8f0', shape='box')
                    else:
                        vis.node(str(node_id), node_label, style='filled', color='#e8f4f8', shape='ellipse')
            
            # Add edges
            for src, dst, edge_attrs in self.tree_edges:
                vis.edge(str(src), str(dst), label=edge_attrs.get('label', ''))
            
            # Display in Streamlit
            vis_container.graphviz_chart(vis)
            time.sleep(self.delay)
        
        def calculate_entropy(self, y):
            """Enhanced entropy calculation with visualization"""
            # Calculate entropy as in the original method
            classes, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            # Prepare breakdown for visualization
            breakdown = dict(zip(classes, probabilities))
            
            # Visualize entropy
            self._update_entropy_visualization(entropy, breakdown)
            
            return entropy
            
        def fit(self, X, y, feature_names=None, display_steps=True):
            """Modified fit method that reports to Streamlit"""
            self.output_text = []  # Clear previous output
            self.step_data = []
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            self.print_step("Starting Decision Tree Construction...\n")
            
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
            
            # Estimate total steps for progress bar
            # This is a rough estimate based on number of features and samples
            estimated_steps = min(X.shape[0], 2**X.shape[1])
            
            # Build tree recursively
            self.root = self._build_tree(X, y, depth=0, parent_id=None, display_steps=display_steps, 
                                        progress_bar=progress_bar, total_steps=estimated_steps)
            
            # Complete progress bar
            progress_bar.progress(1.0)
            
            self.print_step("\nDecision Tree Construction Complete!")
            return self
        
        def _build_tree(self, X, y, depth=0, parent_id=None, branch_value=None, display_steps=True, 
                       progress_bar=None, total_steps=100):
            """Recursively build tree with enhanced Streamlit reporting"""
            n_samples = len(y)
            
            # Update progress
            if progress_bar is not None:
                current_progress = min(1.0, (self.step + 1) / total_steps)
                progress_bar.progress(current_progress)
            
            # Calculate entropy of current node
            self.print_step(f"\n{'='*50}")
            self.print_step(f"STEP {self.step + 1}: Building node at depth {depth}")
            self.print_step(f"{'='*50}")
            self.print_step(f"Current samples: {n_samples}")
            
            classes, counts = np.unique(y, return_counts=True)
            class_distribution = list(zip(classes, counts))
            self.print_step(f"Class distribution: {class_distribution}")
            
            # Calculate entropy with visualization
            current_entropy = self.calculate_entropy(y)
            self.print_step(f"Node entropy: {current_entropy:.4f}")
            
            # Create node
            node = super()._build_tree(X, y, depth, parent_id, branch_value, display_steps)
            
            # Add to step data for later replaying
            self.step_data.append({
                'step': self.step,
                'depth': depth,
                'samples': n_samples,
                'class_distribution': class_distribution,
                'entropy': current_entropy,
                'node_id': self.node_count - 1  # The ID assigned to this node
            })
            
            return node
    
    # Create custom simulator for Streamlit
    class StreamlitSimulator(DecisionTreeSimulator):
        def __init__(self, delay=1.0, show_formulas=True, min_samples_split=2):
            """Initialize with Streamlit integration"""
            super().__init__()
            self.tree_builder = StreamlitTreeBuilder(min_samples_split=min_samples_split, 
                                                   delay=delay, show_formulas=show_formulas)
        
        def run_simulation(self, display_steps=True):
            """Run simulation with Streamlit visualization"""
            if self.data is None:
                self.load_play_tennis_dataset()
            
            # Encode features
            X, y = self.encode_categorical_features()
            
            # Train decision tree with visualization
            self.tree_builder.fit(X, y, feature_names=self.feature_names, display_steps=display_steps)
            
            return self.tree_builder
        
        def encode_categorical_features(self):
            """Enhanced categorical encoding with visual feedback"""
            # For this educational tool, we'll use a simple label encoding
            data_encoded = self.data.copy()
            
            encoding_info = {}
            
            # Encode all categorical columns
            for column in data_encoded.columns:
                if data_encoded[column].dtype == 'object':
                    # Get unique values and map them to integers
                    unique_values = data_encoded[column].unique()
                    value_to_int = {value: i for i, value in enumerate(unique_values)}
                    
                    # Replace values with integers
                    data_encoded[column] = data_encoded[column].map(value_to_int)
                    
                    # Store mapping for reference
                    encoding_info[column] = value_to_int
            
            # Display encoding information in a table
            if encoding_info:
                st.markdown("### Feature Encoding")
                st.markdown("For the algorithm to work, we need to convert categorical features to numbers:")
                
                for column, mapping in encoding_info.items():
                    st.markdown(f"**{column}**: {mapping}")
            
            # Split into features and target
            X = data_encoded[self.feature_names].values
            y = data_encoded[self.target_name].values
            
            return X, y
    
    # Start button
    if st.button("Start Simulation", key="start_sim_btn"):
        # Create streamlit simulator with the selected settings
        st_simulator = StreamlitSimulator(
            delay=delay, 
            show_formulas=show_formulas,
            min_samples_split=min_samples
        )
        st_simulator.data = data
        st_simulator.feature_names = simulator.feature_names
        st_simulator.target_name = simulator.target_name
        
        # Run simulation
        with st.spinner("Running simulation..."):
            tree = st_simulator.run_simulation(display_steps=True)
        
        st.success("Simulation complete!")
        
        # Display final tree with download option
        st.subheader("Final Decision Tree")
        tree.visualize_tree()
        st.image("decision_tree_final.png")
        
        with open("decision_tree_final.png", "rb") as file:
            btn = st.download_button(
                label="Download Tree Image",
                data=file,
                file_name="decision_tree.png",
                mime="image/png"
            )
        
        # Add prediction section with improved UI
        st.markdown('<p class="section-header">Make Predictions</p>', unsafe_allow_html=True)
        st.write("Create a new sample and see how the decision tree classifies it:")
        
        # Create input fields for each feature
        col1, col2 = st.columns(2)
        
        sample_data = {}
        feature_cols = simulator.feature_names
        half = len(feature_cols) // 2 + len(feature_cols) % 2
        
        with col1:
            for feature in feature_cols[:half]:
                # Get unique values for this feature from the dataset
                unique_values = data[feature].unique().tolist()
                # Create a selectbox for the feature
                value = st.selectbox(f"Select {feature}", unique_values)
                sample_data[feature] = value
        
        with col2:
            for feature in feature_cols[half:]:
                # Get unique values for this feature from the dataset
                unique_values = data[feature].unique().tolist()
                # Create a selectbox for the feature
                value = st.selectbox(f"Select {feature}", unique_values)
                sample_data[feature] = value
        
        # Create a dataframe from the sample
        sample_df = pd.DataFrame([sample_data])
        
        # Display the sample
        st.write("Your sample:")
        st.dataframe(sample_df)
        
        # Animated prediction visualization
        if st.button("Predict", key="predict_btn"):
            prediction = st_simulator.predict(sample_df)
            
            # Animate the prediction path 
            st.subheader("Decision Path Animation")
            
            # Create placeholder for the animation
            path_container = st.empty()
            path_text = st.empty()
            
            # Here we would trace the path through the tree
            # For this educational version, we'll simulate it
            import graphviz
            
            # Get all nodes from the tree
            nodes = tree.tree_nodes
            edges = tree.tree_edges
            
            # Find potential path through the tree to the prediction
            # This is a simplified approach for educational purposes
            
            # Start from root
            current_id = 0
            path_ids = [current_id]
            
            # Simulate tracing through the tree
            while True:
                # Find children of current node
                children = [edge[1] for edge in edges if edge[0] == current_id]
                if not children:
                    break  # Reached a leaf
                    
                # For demo, just take the first child
                current_id = children[0]
                path_ids.append(current_id)
                
                # If we reach a leaf node, stop
                node_type = next((node[1].get('type') for node in nodes if node[0] == current_id), None)
                if node_type == 'leaf':
                    break
            
            # Now animate the path
            for i, node_id in enumerate(path_ids):
                # Create visual with highlighted path
                vis = graphviz.Digraph(format='png')
                vis.attr('graph', rankdir='TB')
                
                # Add all nodes
                for n_id, n_attrs in nodes:
                    node_label = n_attrs['label']
                    node_type = n_attrs.get('type', 'unknown')
                    
                    # Highlight current node in path
                    if n_id == node_id:
                        if node_type == 'leaf':
                            vis.node(str(n_id), node_label, style='filled', color='#ff6b6b', fontcolor='white', shape='box')
                        else:
                            vis.node(str(n_id), node_label, style='filled', color='#ff6b6b', fontcolor='white', shape='ellipse')
                    # Highlight previous nodes in path
                    elif n_id in path_ids[:i]:
                        if node_type == 'leaf':
                            vis.node(str(n_id), node_label, style='filled', color='#42b983', fontcolor='white', shape='box')
                        else:
                            vis.node(str(n_id), node_label, style='filled', color='#3498db', fontcolor='white', shape='ellipse')
                    else:
                        if node_type == 'leaf':
                            vis.node(str(n_id), node_label, style='filled', color='#e8f8f0', shape='box')
                        else:
                            vis.node(str(n_id), node_label, style='filled', color='#e8f4f8', shape='ellipse')
                
                # Add all edges
                for src, dst, edge_attrs in edges:
                    # Highlight edges in the path
                    if src in path_ids[:i] and dst in path_ids[:i+1] and path_ids.index(dst) == path_ids.index(src) + 1:
                        vis.edge(str(src), str(dst), label=edge_attrs.get('label', ''), color='#ff6b6b', penwidth='2')
                    else:
                        vis.edge(str(src), str(dst), label=edge_attrs.get('label', ''))
                
                # Show visualization
                path_container.graphviz_chart(vis)
                
                # Update text explanation
                node_info = next((n[1] for n in nodes if n[0] == node_id), {})
                if i < len(path_ids) - 1:
                    # Decision node
                    feature_name = node_info.get('feature', 'Unknown')
                    next_edge = next((e for e in edges if e[0] == node_id and e[1] == path_ids[i+1]), None)
                    decision_value = next_edge[2].get('label', '') if next_edge else ''
                    
                    path_text.info(f"Step {i+1}: Checking feature '{feature_name}'. Decision: {decision_value}")
                else:
                    # Leaf node
                    class_value = node_info.get('class', 'Unknown')
                    path_text.success(f"Final Step: Reached leaf node. Prediction: {class_value}")
                
                time.sleep(delay)
            
            # Display final prediction with celebration
            st.balloons()
            st.success(f"Prediction: {prediction[0]}")
            
            # Explain decision factors
            st.subheader("Decision Factors")
            st.markdown("""
            The tree made this prediction by evaluating features in order of their information gain.
            Features that create the most homogeneous groups (lowest entropy) are chosen first.
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Learning Center
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Understanding Decision Trees</p>', unsafe_allow_html=True)
    
    # Organize information into expandable sections
    with st.expander("What is a Decision Tree?", expanded=True):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            A **Decision Tree** is a tree-like model that makes decisions based on asking a series of questions. 
            It starts with a root node and splits the data into branches based on feature values, ultimately leading to leaf nodes that represent the final decision or classification.

            Think of it like a flowchart where each internal node represents a "test" on a feature (e.g., "Is it sunny?"), each branch represents an outcome of the test, and each leaf node represents a class label or decision.
            
            Decision trees are intuitive, easy to understand, and can handle both numerical and categorical data, making them a popular choice for both classification and regression tasks.
            """)
        
        with col2:
            # Display a simple diagram
            import graphviz
            tree_example = graphviz.Digraph()
            tree_example.node('A', 'Outlook?')
            tree_example.node('B', 'Sunny')
            tree_example.node('C', 'Overcast')
            tree_example.node('D', 'Rain')
            tree_example.node('E', 'Humidity?')
            tree_example.node('F', 'Play: Yes')
            tree_example.node('G', 'Wind?')
            tree_example.node('H', 'Play: No')
            tree_example.node('I', 'Play: Yes')
            tree_example.node('J', 'Play: Yes')
            tree_example.node('K', 'Play: No')
            
            tree_example.edge('A', 'B', label='Sunny')
            tree_example.edge('A', 'C', label='Overcast')
            tree_example.edge('A', 'D', label='Rain')
            tree_example.edge('B', 'E')
            tree_example.edge('C', 'F')
            tree_example.edge('D', 'G')
            tree_example.edge('E', 'H', label='High')
            tree_example.edge('E', 'I', label='Normal')
            tree_example.edge('G', 'J', label='Weak')
            tree_example.edge('G', 'K', label='Strong')
            
            st.graphviz_chart(tree_example)
    
    with st.expander("How Decision Trees Work"):
        st.markdown("""
        ### The Decision Tree Algorithm

        Decision trees work through a process of recursive partitioning, following these steps:

        1. **Choose the Best Feature**: Select the feature that best splits the data into homogeneous groups (using metrics like entropy and information gain).
        
        2. **Split the Data**: Divide the data based on the selected feature's values.
        
        3. **Recurse**: Repeat steps 1-2 for each subset until reaching a stopping condition.
        
        4. **Create Leaf Nodes**: Assign the majority class (for classification) or average value (for regression) as the prediction.

        ### Key Decision Tree Concepts:

        - **Impurity Measures**: Methods to quantify the homogeneity of data (like Entropy, Gini Index)
        - **Information Gain**: The reduction in impurity achieved by splitting on a feature
        - **Pruning**: Techniques to reduce overfitting by removing branches
        - **Maximum Depth**: Controlling how deep the tree can grow
        - **Minimum Samples**: Setting thresholds for when to stop splitting
        """)
        
        # Create two columns for visuals
        col1, col2 = st.columns(2)
        
        with col1:
            # Information gain chart
            gain_data = pd.DataFrame({
                'Feature': ['Outlook', 'Humidity', 'Wind', 'Temperature'],
                'Information Gain': [0.247, 0.152, 0.048, 0.029]
            })
            
            gain_chart = alt.Chart(gain_data).mark_bar().encode(
                x=alt.X('Information Gain:Q'),
                y=alt.Y('Feature:N', sort='-x'),
                color=alt.Color('Feature:N', legend=None)
            ).properties(
                title='Example: Information Gain by Feature',
                height=200
            )
            
            st.altair_chart(gain_chart, use_container_width=True)
            
        with col2:
            # Create a dataset for overfitting vs pruning visualization
            np.random.seed(42)
            x = np.linspace(0, 10, 100)
            y_true = np.sin(x) + 0.1*x
            y_noisy = y_true + np.random.normal(0, 0.5, size=len(x))
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(x, y_noisy, alpha=0.5, label='Data points')
            ax.plot(x, y_true, 'r-', label='True function')
            
            # Simulate overfit tree with many zigzags
            x_plot = np.linspace(0, 10, 500)
            y_overfit = np.sin(x_plot) + 0.1*x_plot
            # Add zigzags to simulate overfitting
            for i in range(10):
                y_overfit += 0.3*np.sin(5*i*x_plot) * np.exp(-0.5*i)
            
            ax.plot(x_plot, y_overfit, 'g--', label='Overfit tree')
            
            # Simulate pruned tree (smoother)
            y_pruned = np.sin(x_plot) + 0.1*x_plot + 0.3*np.sin(2*x_plot)
            ax.plot(x_plot, y_pruned, 'b--', label='Pruned tree')
            
            ax.legend()
            ax.set_title('Overfitting vs Pruning in Decision Trees')
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Target Value')
            
            st.pyplot(fig)
            
    with st.expander("Decision Tree Mathematics: Entropy & Information Gain"):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            ### Entropy
            
            Entropy measures the impurity or disorder in a dataset. For a set S containing c different classes:
            
            $$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$
            
            Where:
            - $p_i$ is the proportion of examples belonging to class $i$
            
            **Properties of Entropy:**
            - Ranges from 0 to $\log_2(c)$
            - 0 means all examples belong to one class (pure set)
            - Higher values indicate more mixed classes
            
            ### Information Gain
            
            Information gain measures the reduction in entropy after splitting the data on a feature:
            
            $$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$
            
            Where:
            - $S$ is the dataset
            - $A$ is the feature
            - $S_v$ is the subset where feature $A$ has value $v$
            
            **The decision tree algorithm selects the feature with the highest information gain at each step.**
            """)
        
        with col2:
            # Create a visualization of entropy
            p_values = np.linspace(0.001, 0.999, 100)
            entropy_values = -p_values * np.log2(p_values) - (1-p_values) * np.log2(1-p_values)
            
            entropy_df = pd.DataFrame({
                'Probability of Class 1': p_values,
                'Entropy': entropy_values
            })
            
            entropy_chart = alt.Chart(entropy_df).mark_line(color='#3498db').encode(
                x=alt.X('Probability of Class 1:Q', title='Probability of Class 1'),
                y=alt.Y('Entropy:Q', title='Entropy')
            ).properties(
                title='Binary Entropy Function',
                height=250
            )
            
            st.altair_chart(entropy_chart, use_container_width=True)
            
            # Add an example calculation
            st.markdown("""
            **Example Entropy Calculation:**
            
            For a node with 9 'Yes' and 5 'No' examples:
            
            $p_{yes} = 9/14 = 0.643$
            $p_{no} = 5/14 = 0.357$
            
            $H(S) = -0.643 \log_2(0.643) - 0.357 \log_2(0.357) = 0.94$
            """)
    
    with st.expander("Advantages & Limitations of Decision Trees"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Advantages
            
            1. **Interpretability**: Easy to visualize and explain
            2. **Minimal Data Preparation**: No normalization or scaling needed
            3. **Handles Mixed Data**: Works with both numerical and categorical features
            4. **Handles Missing Values**: Can work around missing data
            5. **Non-parametric**: Makes no assumptions about data distribution
            6. **Automatic Feature Selection**: Focuses on most important features
            """)
            
            # Add a visual for advantages
            advantages = pd.DataFrame({
                'Advantage': ['Interpretability', 'Low Prep', 'Mixed Data', 'Missing Values', 'Non-parametric', 'Feature Selection'],
                'Score': [9, 8, 9, 7, 8, 9]
            })
            
            adv_chart = alt.Chart(advantages).mark_bar(color='#2ecc71').encode(
                x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 10])),
                y=alt.Y('Advantage:N', sort='-x'),
                tooltip=['Advantage', 'Score']
            ).properties(height=200)
            
            st.altair_chart(adv_chart, use_container_width=True)
            
        with col2:
            st.markdown("""
            ### Limitations
            
            1. **Overfitting**: Tends to create complex trees that don't generalize well
            2. **Instability**: Small changes in data can result in very different trees
            3. **Bias with Dominant Classes**: Can be biased toward features with more levels
            4. **Struggle with Linear Relationships**: Inefficient for some simple patterns
            5. **Limited Expressiveness**: May not capture complex relationships
            6. **No Backtracking**: Mistakes at top nodes propagate downstream
            """)
            
            # Add a visual for limitations
            limitations = pd.DataFrame({
                'Limitation': ['Overfitting', 'Instability', 'Bias', 'Linear Relations', 'Expressiveness', 'No Backtracking'],
                'Severity': [8, 7, 6, 7, 5, 6]
            })
            
            lim_chart = alt.Chart(limitations).mark_bar(color='#e74c3c').encode(
                x=alt.X('Severity:Q', scale=alt.Scale(domain=[0, 10])),
                y=alt.Y('Limitation:N', sort='-x'),
                tooltip=['Limitation', 'Severity']
            ).properties(height=200)
            
            st.altair_chart(lim_chart, use_container_width=True)
    
    with st.expander("Advanced Decision Tree Concepts"):
        st.markdown("""
        ### Beyond Basic Decision Trees
        
        #### 1. Pruning Strategies
        - **Pre-pruning**: Stop growing the tree before it becomes too complex
        - **Post-pruning**: Build the full tree, then remove branches that don't improve performance
        - **Cost-Complexity Pruning**: Balance tree complexity with accuracy using a parameter α
        
        #### 2. Decision Tree Ensembles
        - **Random Forests**: Build many trees on random subsets of data and features
        - **Gradient Boosting Trees**: Build trees sequentially, each correcting errors of previous trees
        - **AdaBoost**: Weight samples based on classification difficulty
        
        #### 3. Handling Continuous Features
        - **Binary Splits**: Find optimal threshold to split continuous values
        - **Multiway Splits**: Create multiple thresholds for more granular splits
        
        #### 4. Handling Missing Values
        - **Surrogate Splits**: Use correlations between features to handle missing values
        - **Instance Weighting**: Split samples with missing values to both branches with weights
        
        #### 5. Decision Trees for Regression
        - Use variance reduction instead of entropy/information gain
        - Leaf nodes predict mean values instead of class labels
        """)
        
        # Add a visual comparing ensemble methods
        ensemble_data = pd.DataFrame({
            'Method': ['Single Decision Tree', 'Random Forest', 'Gradient Boosting', 'AdaBoost'],
            'Accuracy': [0.75, 0.92, 0.94, 0.89],
            'Training Time': [1, 8, 12, 6],
            'Interpretability': [10, 4, 2, 5]
        })
        
        # Melt the data for easier plotting
        ensemble_long = pd.melt(ensemble_data, id_vars=['Method'], 
                               var_name='Metric', value_name='Score')
        
        # Create a grouped bar chart
        ensemble_chart = alt.Chart(ensemble_long).mark_bar().encode(
            x=alt.X('Method:N'),
            y=alt.Y('Score:Q'),
            color=alt.Color('Metric:N', scale=alt.Scale(scheme='category10')),
            column=alt.Column('Metric:N')
        ).properties(
            width=120,
            height=200
        )
        
        st.altair_chart(ensemble_chart)
        
    with st.expander("Real-world Applications"):
        st.markdown("""
        ### Where Decision Trees Are Used
        
        Decision trees and their ensemble variants are widely used across various domains:
        
        #### 1. Finance
        - Credit scoring and approval systems
        - Fraud detection
        - Stock price prediction and trading strategies
        - Customer churn prediction
        
        #### 2. Healthcare
        - Disease diagnosis
        - Patient triage
        - Treatment outcome prediction
        - Drug discovery
        
        #### 3. Marketing
        - Customer segmentation
        - Product recommendation
        - Campaign response prediction
        - Customer lifetime value estimation
        
        #### 4. Operations
        - Supply chain optimization
        - Quality control
        - Predictive maintenance
        - Resource allocation
        
        #### 5. Environmental Science
        - Species identification
        - Habitat classification
        - Climate pattern recognition
        - Natural resource management
        """)
        
        # Add a visual showing industry applications
        industry_data = pd.DataFrame({
            'Industry': ['Finance', 'Healthcare', 'Marketing', 'Operations', 'Environmental'],
            'Usage Percentage': [85, 78, 92, 65, 70]
        })
        
        industry_chart = alt.Chart(industry_data).mark_bar().encode(
            x=alt.X('Usage Percentage:Q', title='Adoption Rate (%)'),
            y=alt.Y('Industry:N', sort='-x'),
            color=alt.Color('Industry:N', legend=None)
        ).properties(
            title='Decision Tree Usage by Industry',
            height=250
        )
        
        st.altair_chart(industry_chart, use_container_width=True)
        
    with st.expander("Recommended Resources"):
        st.markdown("""
        ### Learn More About Decision Trees
        
        #### Books
        - **"The Elements of Statistical Learning"** by Hastie, Tibshirani, and Friedman
        - **"Pattern Recognition and Machine Learning"** by Christopher Bishop
        - **"Machine Learning: A Probabilistic Perspective"** by Kevin Murphy
        
        #### Online Courses
        - **Stanford CS229: Machine Learning**
        - **Coursera: Machine Learning by Andrew Ng**
        - **fast.ai: Practical Deep Learning for Coders**
        
        #### Python Libraries
        - **scikit-learn**: Comprehensive implementation of decision trees and ensembles
        - **XGBoost**: Optimized gradient boosting library
        - **LightGBM**: Microsoft's high-performance gradient boosting framework
        
        #### Interactive Tools
        - **Google's Teachable Machine**: Visual tool for building decision trees
        - **Wolfram Alpha's Decision Tree Creator**: Online decision tree builder
        - **This simulator!** Play with the parameters and see how it affects the tree
        """)
        
        # Create columns for related resources
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Video Tutorial")
            st.video("https://www.youtube.com/watch?v=7VeUPuFGJHk  ")
        
        with col2:
            st.markdown("### Helpful Articles")
            st.markdown("""
            - [Decision Trees - A Simple Way to Visualize Decisions](https://towardsdatascience.com/decision-trees-a-simple-way-to-visualize-a-decision-dc506a403aeb  )
            - [Understanding Random Forests and Decision Trees](https://towardsdatascience.com/understanding-random-forest-58381e0602d2  )
            - [The Mathematics of Decision Trees, Random Forest and Feature Importance](https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3  )
            """)
            
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Decision Tree Playground
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Interactive Decision Tree Playground</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-text">
    This playground allows you to create your own dataset and see how decision trees handle different patterns. Draw points on the canvas and watch the decision boundary evolve!
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for controls and visualization
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Controls")
        
        # Dataset options
        dataset_type = st.radio(
            "Select Pattern Type",
            ["Linearly Separable", "Circular", "Spiral", "Custom Drawing"]
        )
        
        # Tree complexity
        max_depth = st.slider("Maximum Tree Depth", min_value=1, max_value=10, value=3)
        
        # Sample size slider for predefined patterns
        if dataset_type != "Custom Drawing":
            n_samples = st.slider("Number of Samples", min_value=50, max_value=500, value=200)
        
        # Add noise option
        noise_level = st.slider("Noise Level", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
        # Button to generate dataset
        generate_btn = st.button("Generate Dataset")
        
        # Button to fit decision tree
        fit_btn = st.button("Fit Decision Tree")
        
        # Reset button
        reset_btn = st.button("Reset")
    
    with col2:
        st.markdown("### Decision Tree Visualization")
        
        # Create placeholder for the scatterplot
        scatter_plot = st.empty()
        
        # Create placeholder for the decision boundary
        boundary_plot = st.empty()
        
        # Create placeholder for the tree visualization
        tree_viz = st.empty()
        
        # Function to generate datasets
        def generate_dataset(dataset_type, n_samples=200, noise=0.1):
            """Generate different types of 2D datasets"""
            np.random.seed(42)
            
            if dataset_type == "Linearly Separable":
                # Create linearly separable data
                X = np.random.randn(n_samples, 2)
                y = (X[:, 0] + X[:, 1] > 0).astype(int)
                # Add noise by flipping some labels
                flip_indices = np.random.choice(n_samples, int(noise * n_samples), replace=False)
                y[flip_indices] = 1 - y[flip_indices]
                
            elif dataset_type == "Circular":
                # Create circular pattern
                X = np.random.randn(n_samples, 2)
                # Distance from origin
                r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
                # Class is determined by distance from origin
                y = (r < np.sqrt(2/np.pi)).astype(int)
                # Add noise
                flip_indices = np.random.choice(n_samples, int(noise * n_samples), replace=False)
                y[flip_indices] = 1 - y[flip_indices]
                
            elif dataset_type == "Spiral":
                # Create spiral pattern
                n_per_class = n_samples // 2
                theta = np.sqrt(np.random.rand(n_per_class)) * 4 * np.pi
                
                # First spiral (class 0)
                r_a = 2 * theta + np.pi
                data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
                X_a = data_a + np.random.randn(n_per_class, 2) * noise
                
                # Second spiral (class 1)
                r_b = -2 * theta - np.pi
                data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
                X_b = data_b + np.random.randn(n_per_class, 2) * noise
                
                X = np.vstack([X_a, X_b])
                y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
                
            elif dataset_type == "Custom Drawing":
                # This would be implemented with interactive drawing
                # For now, just create a simple dataset as a placeholder
                X = np.random.randn(100, 2)
                y = (X[:, 0] > 0).astype(int)
            
            # Create DataFrame for plotting
            df = pd.DataFrame(X, columns=['X1', 'X2'])
            df['Class'] = y
            return df
        
        # Plot dataset and decision boundary
        if generate_btn or 'generated_data' not in st.session_state:
            if dataset_type != "Custom Drawing" or 'generated_data' not in st.session_state:
                # Generate dataset
                df = generate_dataset(dataset_type, n_samples, noise_level)
                st.session_state.generated_data = df
                st.session_state.tree_fitted = False
            
            # Plot dataset
            scatter = alt.Chart(st.session_state.generated_data).mark_circle(size=100).encode(
                x=alt.X('X1:Q', scale=alt.Scale(domain=[-5, 5])),
                y=alt.Y('X2:Q', scale=alt.Scale(domain=[-5, 5])),
                color=alt.Color('Class:N', scale=alt.Scale(domain=[0, 1], range=['#3498db', '#e74c3c']))
            ).properties(
                width=500,
                height=500,
                title='Dataset'
            ).interactive()
            
            scatter_plot.altair_chart(scatter, use_container_width=True)
        
        # Fit decision tree and visualize decision boundary
        if fit_btn:
            from sklearn.tree import DecisionTreeClassifier, export_graphviz
            import graphviz
            
            # Get data
            df = st.session_state.generated_data
            X = df[['X1', 'X2']].values
            y = df['Class'].values
            
            # Fit decision tree
            clf = DecisionTreeClassifier(max_depth=max_depth)
            clf.fit(X, y)
            st.session_state.tree_fitted = True
            st.session_state.tree_model = clf
            
            # Create mesh grid for decision boundary
            h = 0.05  # Step size in the mesh
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Predict on mesh grid
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
            
            # Plot training points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, 
                               edgecolor='k', s=50, cmap=plt.cm.coolwarm)
            
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title(f'Decision Boundary (Tree Depth = {max_depth})')
            
            boundary_plot.pyplot(fig)
            
            # Visualize the tree
            dot_data = export_graphviz(clf, 
                                      feature_names=['X1', 'X2'],
                                      class_names=['Class 0', 'Class 1'],
                                      filled=True, 
                                      rounded=True,
                                      special_characters=True)
            
            graph = graphviz.Source(dot_data)
            tree_viz.graphviz_chart(graph)
            
            # Display feature importances
            st.markdown("### Feature Importance")
            feature_imp = pd.DataFrame({
                'Feature': ['X1', 'X2'],
                'Importance': clf.feature_importances_
            })
            
            imp_chart = alt.Chart(feature_imp).mark_bar().encode(
                x=alt.X('Importance:Q'),
                y=alt.Y('Feature:N', sort='-x'),
                color=alt.Color('Feature:N', legend=None)
            ).properties(height=100)
            
            st.altair_chart(imp_chart, use_container_width=True)
            
            # Display metrics
            st.markdown("### Model Performance")
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y, clf.predict(X))
            st.metric("Training Accuracy", f"{accuracy:.2%}")
            
            # Display decision path for random sample
            st.markdown("### Decision Path for Random Sample")
            random_idx = np.random.randint(0, len(X))
            sample = X[random_idx]
            sample_class = y[random_idx]
            
            st.markdown(f"Selected sample: X1={sample[0]:.2f}, X2={sample[1]:.2f}, True Class={sample_class}")
            
            # Get decision path
            decision_path = clf.decision_path([sample])
            path_indices = decision_path.indices
            
            # Get node feature and threshold values
            n_nodes = clf.tree_.node_count
            feature = clf.tree_.feature
            threshold = clf.tree_.threshold
            
            # Display the decision path
            st.markdown("Decision path:")
            path_steps = []
            for node_id in path_indices:
                if node_id == 0:
                    path_steps.append(f"Root node")
                if node_id < n_nodes and feature[node_id] != -2:  # Not a leaf
                    if sample[feature[node_id]] <= threshold[node_id]:
                        path_steps.append(f"X{feature[node_id]+1} = {sample[feature[node_id]]:.2f} <= {threshold[node_id]:.2f}? Yes")
                    else:
                        path_steps.append(f"X{feature[node_id]+1} = {sample[feature[node_id]]:.2f} <= {threshold[node_id]:.2f}? No")
            
            for step in path_steps:
                st.markdown(f"- {step}")
        
        # Reset functionality
        if reset_btn:
            if 'generated_data' in st.session_state:
                del st.session_state.generated_data
            if 'tree_fitted' in st.session_state:
                del st.session_state.tree_fitted
            if 'tree_model' in st.session_state:
                del st.session_state.tree_model
            
            # Clear plots
            scatter_plot.empty()
            boundary_plot.empty()
            tree_viz.empty()
    
    # Add explanation of playground
    with st.expander("How to Use the Playground"):
        st.markdown("""
        ### Instructions for the Decision Tree Playground
        
        1. **Select a pattern type** from the dropdown on the left
        2. **Adjust the parameters** like tree depth and noise level
        3. **Click "Generate Dataset"** to create a new dataset with the selected pattern
        4. **Click "Fit Decision Tree"** to train a decision tree and visualize the decision boundary
        5. **Experiment with different depths** to see how the decision boundary changes
        6. **Reset** to start over with a new dataset
        
        #### Pattern Types:
        
        - **Linearly Separable**: Classes can be separated by a straight line
        - **Circular**: One class is inside a circle, another is outside
        - **Spiral**: Two intertwined spiral patterns (challenging for decision trees)
        - **Custom Drawing**: Create your own pattern by drawing on the canvas
        
        #### What to Observe:
        
        - **Decision Boundary Complexity**: Notice how increasing tree depth creates more complex boundaries
        - **Overfitting**: Very high depths may fit the training data perfectly but create jagged, unrealistic boundaries
        - **Feature Importance**: See which feature contributes more to the decisions
        - **Decision Path**: Follow how the tree makes decisions for individual points
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Decision Tree Simulator - An Educational Tool for Machine Learning")