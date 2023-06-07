# HAR

Code for the paper "Efficient Deep Clustering of Human Activities and How to Improve Evaluation" https://proceedings.mlr.press/v189/mahon23a.html, which highlights the difference between subject-dependent and subject-independent clustering, and also proposes a new state-of-the-art method for deep clustering of human activities.

Steps to reproduce:

- Create a conda environment, python>=3.7, and install dependencies with . manual_requirements_conda.txt
- Download the datasets, with . download_datasets.sh
- Train and test the model with python har_cnn.py --PAMAP --num_meta_epochs 10 --num_meta_meta_epochs 10
- Observe the effect of subject dependence, by retraining and retesting with the --subject_independent flag
