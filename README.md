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

**requirements.txt**
```
numpy
pandas
matplotlib
networkx
graphviz
streamlit
streamlit_lottie
altair
IPython
```
