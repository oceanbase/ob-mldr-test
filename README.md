[English](README.md) | [中文](README_CN.md) 

# MLDR dataset testing framework

A modified Multi-Language Document Retrieval (MLDR) testing framework based on the [Infinity MLDR Benchmark](https://github.com/infiniflow/infinity/tree/main/python/benchmark/mldr_benchmark).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Project Introduction

This project is a comprehensive MLDR (Multi-Lingual Document Retrieval) dataset testing framework designed to evaluate and test the performance of multi-language document retrieval systems. It supports various query types, including BM25, Dense Vector, and hybrid retrieval.

## Functional Features

### 🗄️ Supported database backends
- **OceanBase**: supports full-text search, vector search and hybrid search
- **seekdb**: supports full-text search, vector search and hybrid search

### 🔍 Supported query types

| Query Type | Description | Supported Backends |
|---------|------|---------|
| `bm25` | BM25 Full-Text Search | OceanBase, seekdb |
| `dense` | Dense Vector Search | OceanBase, seekdb |
| `hybrid_dense_bm25` | Dense+BM25 Hybrid Search | OceanBase, seekdb |

### 📊 Evaluation Metrics 
- **Recall@10**: Recall rate of the top 10 results
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Average Query Time**: Average response time per query

## Environmental requirements

### System Requirements
- **Python**: 3.11 or above
- **Java**: JDK 11 or above (for pyserini)

### Dependent Services
- **OceanBase Database**: Versions supporting vector retrieval and full-text search (4.4.1 and above)
- **seekdb Database**: Supports vector retrieval and full-text search

---

## Installation Guide

### 1. Install Java environment

**Linux (Alibaba Cloud Linux / CentOS):**
```bash
sudo dnf install java-11-openjdk-devel -y
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so
```

### 2. Create a Python virtual environment

```bash
# Download and install Conda 
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Reopen the terminal and initialize the Conda environment 
source ~/miniconda3/bin/activate
conda init --all

# Create and initialize the Python environment required for mdlr
conda create -n test python=3.11
conda activate test
```

### 3. Install Python dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 4. Prepare the database

**Start OceanBase or seekdb database:**
```bash
# Ensure the database is started and the connection information is configured
# Configure the connection parameters for the corresponding database in config.yaml
```

---

## Quick Start

> **⚠️ Important Note: Please configure the database information before use**
> 
> Before running the test, please complete the following configuration steps:
> 
> ```bash
> # 1. Copy the example configuration file
> cp config.yaml.example config.yaml
> 
> # 2. Edit the configuration file and replace the database information with your own configuration
> vim config.yaml  
> ```
> 
> The main configuration items that need to be modified are:
> - `oceanbase.host`: Database host address
> - `oceanbase.port`: Database port number
> - `oceanbase.user`: Database username
> - `oceanbase.password`: Database password
> - `oceanbase.database`: Database name
> - `embedding.vector_download_url`: Vector file download URL (if vector retrieval is used)

### Run a complete test with one click

```bash
# English mixed retrieval test
python mldr_test_runner.py --lang en --query-type hybrid_dense_bm25

# Skip data insertion (data already exists)
python mldr_test_runner.py --lang en --query-type hybrid_dense_bm25 --skip-insert
```

### Test process description

The testing framework will automatically execute the following steps:

1. **Data insertion**: Insert 200,000 test data entries and create indexes
2. **Warm-up test** (optional): Perform a search warm-up
3. **Formal testing**: Perform a specified number of searches and evaluations
4. **Result Output**: Display the average Recall@10, NDCG@10, and average latency

---

## Detailed Usage

### Command line parameters

```bash
python mldr_test_runner.py [OPTIONS]
```

**Basic parameters:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lang` | str | `en` | Test language (e.g. `en`, `zh`, etc.) |
| `--backend` | str | `oceanbase` | Database backend (`oceanbase` or `seekdb`) |
| `--query-type` | str | `bm25` | Query type, optional values: `hybrid_dense_bm25` (hybrid retrieval), `dense` (vector retrieval), `bm25` (full-text retrieval) |
| `--skip-insert` | flag | `False` | Skip the data insertion step (used when the data already exists)  |
| `--result-dir` | str | `/tmp/` | directory for saving results |
| `--config` | str | `None` | Path to the configuration file (in YAML format), defaults to `config.yaml` in the current directory |