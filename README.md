# Multimodal Product Classifier

> Image and Text Embeddings for Product Category Classification

This project implements a multimodal product classification pipeline that classifies e-commerce products into predefined categories using pre-trained deep learning models for feature extraction (image and text embeddings) followed by classical machine learning and MLP classifiers.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Modeling Approach](#modeling-approach)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [How to Run](#how-to-run)
- [Architecture Diagram](#architecture-diagram)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)

---

## Problem Statement

E-commerce platforms require accurate product categorization to enable search, recommendations, inventory management, and user navigation. Manual categorization does not scale for large catalogs.

This project addresses the following:

### Classification Task
- **Multiclass product classification** into predefined category hierarchies
- **Multimodal input**: Product images and text descriptions must be leveraged jointly
- **Performance targets**:
  - Multimodal model: at least 85% accuracy and 80% F1-score
  - Text-only model: at least 85% accuracy and 80% F1-score
  - Image-only model: at least 75% accuracy and 70% F1-score

### Technical Requirements
- Extract image embeddings using pre-trained computer vision models (ConvNextV2, ResNet50)
- Extract text embeddings using pre-trained language models (e.g., MiniLM-L6-v2)
- Train and evaluate classical ML models (Random Forest, Logistic Regression)
- Train and evaluate MLP models for single-modality and multimodal fusion
- Generate reproducible results with classification reports and confusion matrices

---

## Dataset Description

### Data Source

The `processed_products_with_images.csv` dataset contains product listings from BestBuy.com with structured metadata, descriptions, and image URLs.

### Schema

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| `sku`         | Unique product identifier                        |
| `name`        | Product name                                     |
| `description` | Product description (text)                       |
| `image`       | URL of the product image                         |
| `type`        | Product type (e.g., HardGood, Software)          |
| `price`       | Product price                                    |
| `shipping`    | Shipping cost                                    |
| `manufacturer`| Manufacturer name                                |
| `class_id`    | Product class identifier                         |
| `sub_class1_id` | Product sub-class identifier                   |
| `image_path`  | Local path where the image is stored             |

### Data Preparation

- Images are downloaded from URLs and stored in `data/images/`
- Default image resolution: 224x224 pixels (JPG format)
- Optional: `categories.json` provides hierarchical category relationships

### Data Access

1. Place the preprocessed dataset at `data/processed_products_with_images.csv`
2. [Download the images zip file](https://drive.google.com/file/d/14s2aDNTEWse86cWyLhvVIKmob6EbQrm_/view?usp=sharing) and extract to `data/images/`

---

## Modeling Approach

### 1. Image Embedding Extraction (`src/vision_embeddings_tf.py`)

- **Backbone models**: ConvNextV2 (Hugging Face) and ResNet50 (TensorFlow Keras)
- **Input**: 224x224 RGB images, normalized to [0, 1]
- **Output**: Fixed-size embedding vectors per image
- **Models supported**: ResNet50, ResNet101, DenseNet121/169, InceptionV3, ConvNextV2, ViT, Swin Transformer

### 2. Text Embedding Extraction (`src/nlp_models.py`)

- **Primary model**: `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
- **Alternative**: OpenAI API (GPT class) for proprietary embeddings
- **Input**: Product name and description
- **Output**: Fixed-size text embedding vectors

### 3. Embedding Fusion and Preprocessing (`src/utils.py`)

- Merge image and text embeddings into a single feature matrix
- Split dataset into train and test sets
- Extract feature columns for text-only, image-only, and combined inputs

### 4. Classical ML Classifiers (`src/classifiers_classic_ml.py`)

- **Models**: Random Forest, Logistic Regression (minimum required)
- **Modalities**: Image-only, text-only, and early-fusion (combined embeddings)
- **Embedding visualization**: PCA and t-SNE (2D and 3D) for exploratory analysis

### 5. MLP Classifiers (`src/classifiers_mlp.py`)

- **Architecture**: Early-fusion MLP with BatchNorm, Dropout
- **Input**: Supports single-modality or multimodal concatenated embeddings
- **Training**: Categorical cross-entropy, Adam optimizer, early stopping
- **Modalities**: Image-only, text-only, and combined embeddings

---

## Evaluation Metrics

### Primary Metrics

- **Accuracy**: Overall proportion of correct predictions
- **F1-score**: Macro-averaged F1 (balance of precision and recall across classes)
- **Precision** and **Recall**: Per-class and macro-averaged

### Secondary Metrics

- **Confusion matrix**: Per-class performance visualization
- **Classification report**: Precision, recall, F1 per class

### Validation

- Train/test split for held-out evaluation
- Unit tests validate model implementations and expected behavior (`pytest tests/`)

---

## Results

### Expected Performance (Project Requirements)

| Model Type | Modality   | Target Accuracy | Target F1-Score |
|------------|------------|-----------------|-----------------|
| MLP        | Multimodal | 85%             | 80%             |
| MLP        | Text-only  | 85%             | 80%             |
| MLP        | Image-only | 75%             | 70%             |

### Output Artifacts

- `results/multimodal_results.csv`: Predictions for combined embeddings
- `results/image_results.csv`: Predictions for image-only model
- `results/text_results.csv`: Predictions for text-only model
- Classification reports and confusion matrices rendered in the Jupyter notebook

---

## How to Run

### Prerequisites

- Python 3.9+
- Git (for cloning the repository)
- GPU recommended for embedding extraction (optional; CPU supported)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd sprint4
```

### Step 2: Set Up Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Mac (M-series GPU):**

```bash
pip install -r requirements_mac.txt
```

### Step 4: Prepare Data

1. Place `processed_products_with_images.csv` in `data/`
2. [Download images](https://drive.google.com/file/d/14s2aDNTEWse86cWyLhvVIKmob6EbQrm_/view?usp=sharing) and extract to `data/images/`

**Verify structure:**

```
data/
├── processed_products_with_images.csv
└── images/
    └── (product images)
```

### Step 5: Run the Notebook

```bash
jupyter notebook
```

Open `AnyoneAI - Sprint Project 04.ipynb` and run cells in order:

1. **EDA and Image Downloading** – Load data, download images if needed
2. **Generating Image Embeddings** – Extract image features (ConvNextV2, ResNet50)
3. **Generating Text Embeddings** – Extract text features (MiniLM-L6-v2)
4. **Merge Embeddings** – Combine image and text embeddings
5. **Classical ML Training** – Train Random Forest and Logistic Regression
6. **MLP Training** – Train MLP for image-only, text-only, and multimodal

### Step 6: Run Tests

```bash
pytest tests/
```

or without warnings:

```bash
pytest tests/ --disable-warnings
```

### Docker (Alternative)

```bash
docker build -t anyoneai-project .
docker run -p 8888:8888 -v $(pwd):/app anyoneai-project
```

Then open the Jupyter URL with the token from the console.

### Optional: Code Formatting

```bash
black --line-length=88 .
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────┐  ┌─────────────────────────┐ │
│  │  processed_products_         │  │  Product Images         │ │
│  │  with_images.csv             │  │  (data/images/)         │ │
│  │  (metadata + descriptions)   │  │  224x224 JPG            │ │
│  └──────────────────────────────┘  └─────────────────────────┘ │
│               │                              │                  │
└───────────────┼──────────────────────────────┼──────────────────┘
                │                              │
                ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EMBEDDING EXTRACTION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────┐  ┌──────────────────────────┐  │
│  │  Text Embeddings           │  │  Image Embeddings         │  │
│  │  (src/nlp_models.py)       │  │  (src/vision_embeddings_  │  │
│  ├────────────────────────────┤  │   tf.py)                  │  │
│  │  • MiniLM-L6-v2            │  ├──────────────────────────┤  │
│  │  • BERT / GPT (optional)   │  │  • ConvNextV2             │  │
│  └────────────────────────────┘  │  • ResNet50               │  │
│               │                  └──────────────────────────┘  │
│               │                              │                  │
└───────────────┼──────────────────────────────┼──────────────────┘
                │                              │
                └──────────────┬───────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                │
│                    (src/utils.py)                               │
├─────────────────────────────────────────────────────────────────┤
│  • Merge text + image embeddings                                 │
│  • Train/test split                                              │
│  • Feature extraction (text_cols, image_cols, label_col)         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CLASSIFICATION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────┐  ┌────────────────────────────┐   │
│  │  Classical ML           │  │  MLP (src/classifiers_mlp)  │   │
│  │  (src/classifiers_      │  ├────────────────────────────┤   │
│  │   classic_ml.py)        │  │  • Early fusion             │   │
│  ├─────────────────────────┤  │  • BatchNorm, Dropout       │   │
│  │  • Random Forest        │  │  • Early stopping           │   │
│  │  • Logistic Regression  │  └────────────────────────────┘   │
│  └─────────────────────────┘                                    │
│                                                                 │
│  Modalities: Image-only | Text-only | Combined                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT                                    │
├─────────────────────────────────────────────────────────────────┤
│  • results/multimodal_results.csv                                │
│  • results/image_results.csv                                     │
│  • results/text_results.csv                                      │
│  • Classification reports and confusion matrices                 │
│  • Embeddings/*.csv (for reuse)                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

1. **Extract** – Load products and images from CSV and disk
2. **Embed** – Generate image embeddings (ConvNextV2/ResNet50) and text embeddings (MiniLM)
3. **Merge** – Combine embeddings into a single feature matrix
4. **Train** – Fit classical ML and MLP models per modality
5. **Evaluate** – Report accuracy, F1, confusion matrices, and save predictions

---

## Project Structure

```
sprint4/
├── data/                              # Data files (download separately)
│   ├── processed_products_with_images.csv
│   └── images/                        # Product images
├── Embeddings/                        # Generated embeddings (large; exclude from repo)
│   ├── Embeddings_*.csv
│   └── text_embeddings_*.csv
├── results/                           # Model predictions
│   ├── multimodal_results.csv
│   ├── image_results.csv
│   └── text_results.csv
├── src/                               # Source code
│   ├── vision_embeddings_tf.py        # Image embedding extraction
│   ├── nlp_models.py                  # Text embedding extraction
│   ├── utils.py                       # Preprocessing and train/test split
│   ├── classifiers_classic_ml.py      # Random Forest, Logistic Regression
│   └── classifiers_mlp.py             # MLP classifier
├── tests/                             # Unit tests
│   └── test_models.py
├── AnyoneAI - Sprint Project 04.ipynb # Main notebook
├── README.md                          # This file
├── README_og.md                       # Original project instructions
├── requirements.txt                   # Python dependencies
└── Dockerfile                         # Optional container setup
```

---

## Technologies Used

- **Python 3.9+**: Core programming language
- **TensorFlow**: Image embedding backbones (ResNet50, ConvNextV2)
- **Transformers (Hugging Face)**: ConvNextV2, ViT, MiniLM-L6-v2
- **scikit-learn**: Classical ML models, preprocessing, metrics
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib / Seaborn / Plotly**: Visualizations
- **Pytest**: Testing framework
- **Black**: Code formatter

---

## License

This project is part of an educational course assignment.

---

## Author

Marissa Singh

---

## Acknowledgments

- Best Buy for product data (educational use)
- AnyoneAI for the project framework and guidance
- Hugging Face and TensorFlow for pre-trained models
