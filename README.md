# Decision Tree Visualizer 🌳

An interactive educational tool for understanding decision tree algorithms through step-by-step visualization.

## Features ✨
- Step-by-step tree construction visualization
- Interactive entropy and information gain calculations
- Real-time formula explanations
- Multiple dataset support (Play Tennis demo + CSV upload)
- Speed-controlled simulation
- Prediction interface with real-time results
- Modern UI with animated components

## Installation 🛠️
1. Clone the repository:
```bash
git clone https://github.com/yourusername/decision-tree-visualizer.git
cd decision-tree-visualizer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🚀
Start the Streamlit app:
```bash
streamlit run app.py
```

Key functionalities:
- Choose between built-in "Play Tennis" dataset or upload your own CSV
- Control simulation speed using the slider
- Use step-by-step navigation to follow the tree construction
- View real-time entropy and information gain calculations
- Make predictions using interactive input fields

## Requirements 📦
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- Graphviz

## License 📄
MIT License
```

**requirements.txt**
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.22.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
graphviz>=0.20.0
python-dotenv>=0.19.0
```

**.gitignore**
```
# Virtual environment
venv/
.env

# Python cache
__pycache__/
*.py[cod]

# IDE files
.vscode/
.idea/

# Logs and databases
*.log
*.sqlite

# Local development
*.local
.DS_Store

# Streamlit temporary files
.streamlit/

# Visualization outputs
decision_tree_visualization.png
decision_tree_final.png

# Build artifacts
dist/
build/
*.egg-info/

# Jupyter notebooks
*.ipynb_checkpoints

# Test files
test/
tests/

# Dataset files
*.csv
*.data
*.json
```

This setup provides:
1. Clear documentation in README.md
2. Complete dependency list
3. Comprehensive .gitignore for Python/Streamlit projects
4. Standard project structure

The .gitignore file excludes:
- Development environment files
- Cache and temporary files
- Sensitive credentials
- Large binary files
- IDE-specific configurations
- Local configuration files

You can customize these files further based on your specific needs and add screenshots to the README for better visual demonstration.