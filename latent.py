import os
import sys
import logging
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.express as px
import torch.nn.functional as F
from typing import Tuple, List
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from models.vae import BetaVAE


def load_partition_data(project_dir: str, partition_name: str) -> torch.Tensor:
    """
    Loads a .pt file containing partition data (train, val, or test).
    """
    file_path = os.path.join(project_dir, f"{partition_name}_data.pt")
    logging.info(f"Loading {partition_name} data from {file_path} ...")
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        logging.error(f"Error loading {partition_name} data: {e}")
        sys.exit(1)
    return data


def map_group_to_three_categories(group: str) -> str:
    """
    Map original ResearchGroup to one of three categories:
      - 'AD'
      - 'CN'
      - 'Others' (includes 'MCI', 'LMCI', 'EMCI', or anything else)
    """
    group_upper = group.strip().upper()
    if group_upper == 'AD':
        return 'AD'
    elif group_upper == 'CN':
        return 'CN'
    else:
        # For MCI, LMCI, EMCI, etc., return 'Others'
        return 'Others'


def load_subject_metadata(csv_path: str):
    """
    Loads SubjectID, ResearchGroup, Sex, and Age from a CSV.
    Then maps ResearchGroup into AD, CN, or Others.
    
    Assumes the CSV has columns:
        - SubjectID
        - ResearchGroup
        - Sex
        - Age
    """
    df = pd.read_csv(csv_path)
    metadata = {}
    for _, row in df.iterrows():
        # Map the group into the 3 categories:
        new_group = map_group_to_three_categories(row['ResearchGroup'])

        metadata[row['SubjectID']] = {
            'Group': new_group,       # AD, CN, or Others
            'Sex': row['Sex'],
            'Age': row['Age']
        }
    return metadata


def get_latent_embeddings_with_labels(
    model: BetaVAE,
    data: torch.Tensor,
    subject_ids: List[str],
    subject_metadata: dict,
    device: torch.device
):
    """
    Pass each sample through the encoder to get latent embeddings (mu).
    Also extract the 'Group', 'Sex', and 'Age' from subject_metadata.
    """
    model.eval()
    embeddings = []
    group_labels = []
    sex_labels = []
    age_values = []
    subj_ids = []

    for i in range(len(data)):
        sample = data[i:i+1].to(device, dtype=torch.float)
        with torch.no_grad():
            mu, _ = model.encode(sample)
        embeddings.append(mu.cpu().numpy())

        # Retrieve metadata
        subj_id = subject_ids[i]
        meta = subject_metadata.get(subj_id, {'Group': 'Others', 'Sex': 'Unknown', 'Age': -1})
        group_labels.append(meta['Group'])
        sex_labels.append(meta['Sex'])
        age_values.append(meta['Age'])
        subj_ids.append(subj_id)

    return (
        np.vstack(embeddings),
        group_labels,
        sex_labels,
        age_values,
        subj_ids
    )


def plot_2d_projection_with_labels(
    embedding_2d,
    group_labels,
    sex_labels,
    age_values,
    subj_ids,
    title
):
    """
    Create an interactive 2D scatter plot of the embeddings using Plotly.
    - Color by the 3-category group (AD, CN, Others)
    - Hover shows SubjectID, Sex, and Age
    """
    fig = px.scatter(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        color=group_labels,  # This is AD, CN, or Others
        hover_data={
            "SubjectID": subj_ids,
            "Sex": sex_labels,
            "Age": age_values
        },
        labels={"x": "Dim 1", "y": "Dim 2"},
        title=title
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.show()


def main(args):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load subject metadata (mapped to AD, CN, or Others)
    csv_path = os.path.join(args.project_dir, 'DataBaseSubjects.csv')
    subject_metadata = load_subject_metadata(csv_path)

    # 2. Load trained BetaVAE model
    model_path = os.path.join(args.project_dir, "best_beta_vae_model.pth")
    model = BetaVAE(latent_dim=args.latent_dim, beta=args.beta)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Load data partitions
    train_data = load_partition_data(args.project_dir, "train")
    val_data = load_partition_data(args.project_dir, "val")
    test_data = load_partition_data(args.project_dir, "test")

    # Concatenate all samples
    all_data = torch.cat([train_data, val_data, test_data], dim=0)

    # 4. Build list of subject IDs (must match the data order!)
    subject_ids = list(subject_metadata.keys())[:len(all_data)]

    # 5. Extract latent embeddings + metadata labels
    (
        embeddings,
        group_labels,
        sex_labels,
        age_values,
        subj_ids
    ) = get_latent_embeddings_with_labels(
        model, all_data, subject_ids, subject_metadata, device
    )

    # 6. Compute 2D projections (PCA, t-SNE, UMAP)
    pca_2d = PCA(n_components=2).fit_transform(embeddings)
    tsne_2d = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)
    umap_2d = umap.UMAP(n_components=2).fit_transform(embeddings)

    # 7. Plot each projection with color-coded groups (AD, CN, Others)
    plot_2d_projection_with_labels(
        pca_2d, group_labels, sex_labels, age_values, subj_ids,
        "PCA Projection (AD, CN, Others)"
    )
    plot_2d_projection_with_labels(
        tsne_2d, group_labels, sex_labels, age_values, subj_ids,
        "t-SNE Projection (AD, CN, Others)"
    )
    plot_2d_projection_with_labels(
        umap_2d, group_labels, sex_labels, age_values, subj_ids,
        "UMAP Projection (AD, CN, Others)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze BetaVAE latent space with AD/CN/Others color-coding.")
    parser.add_argument("--project_dir", type=str, required=True, help="Project directory path")
    parser.add_argument("--latent_dim", type=int, default=250, help="Latent dimension size")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta coefficient for VAE loss")
    args = parser.parse_args()
    main(args)

