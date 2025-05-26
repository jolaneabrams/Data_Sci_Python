#2 - using different features, PCA, GMM, choose right models, get clusters then create log odds leads to make predictions of
# who will get worse - improvement comes from adding CSF info - if leave it out, problem gets more similar to no.1.
# Not the modeling approach, it's how much info you have

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 2a load columns from ADNIMERGE_final_proj
a_merge_df = pd.read_excel('ADNIMERGE_final_proj.xlsx')

# keep the measures we care about, drop missing data and sort by RID/Years_bl
measures_we_care_about = ['RID', 'AGE_bl', 'Years_bl', 'PTGENDER', 'PTEDUCAT', 'DX', 'ADAS13', 'MMSE', 'ABETA',
                              'TAU', 'Hippocampus', 'Ventricles', 'WholeBrain', 'ICV']
a_merge_df   = a_merge_df[measures_we_care_about]
a_merge_df   = a_merge_df.dropna(subset=measures_we_care_about)

# renaming columns in a_merge_df
columns_to_rename = {
        'AGE_bl': 'age_bl',
        'PTEDUCAT': 'edu',
        'Years_bl': 'years_bl',
        'PTGENDER': 'sex',
    }
# rename columns in a_merge_df
a_merge_df = a_merge_df.rename(columns=columns_to_rename)

# replace "Dementia" with "AD" in 'DX' column of a_merge_df
a_merge_df['DX'] = a_merge_df['DX'].replace('Dementia', 'AD')

# 2b create feature matrix, standardize, run PCA & plot

# filter the data to include only years_bl = 0 rows
year_bl_0_df = a_merge_df[a_merge_df['years_bl'] == 0]

# make feature matrix
feature_columns = ['ADAS13', 'MMSE', 'ABETA', 'TAU', 'Hippocampus', 'Ventricles', 'WholeBrain', 'ICV']
f_mat_df = year_bl_0_df[feature_columns]

# standardize feature matrix
scaler = StandardScaler()
f_mat_scaled = scaler.fit_transform(f_mat_df)
f_mat_scaled_df = pd.DataFrame(f_mat_scaled, columns=feature_columns)

# do PCA n = features in matrix
n_components = 8
pca = PCA(n_components=n_components)

# fit PCA to standardized feature matrix, transform the data
principal_components = pca.fit_transform(f_mat_scaled_df)

# convert principal components back to dataframe
pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components + 1)])

# calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# plotting
plt.figure(figsize=(8, 6))
plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-')
plt.xlabel('Number of PCs')
plt.ylabel('Cumulative explained variance')
plt.xticks(range(1, n_components + 1))
plt.show()

# 2c - project feature matrix onto first 2 PCs, run GMM w/K from 1 - 5, make elbow plots of AIC & BIC

# project features onto 2 principal components
pca = PCA(n_components=2)
projected_features = pca.fit_transform(f_mat_scaled_df)

# Check the number of features after PCA
n_features_pca = projected_features.shape[1]
print("Number features after PCA: ", n_features_pca)

# do KMeans clustering with same number features as in PCA
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(projected_features)

# Check the number of features used in KMeans
n_features_kmeans = kmeans.cluster_centers_.shape[1]
print("Number features used in KMeans: ", n_features_kmeans)

# do GMM initialization with same number of features as in PCA
gmm = GaussianMixture(n_components=2, means_init=kmeans.cluster_centers_, random_state=67)
gmm.fit(projected_features)

# Check the number of features used in GMM initialization
n_features_gmm = gmm.means_.shape[1]
print("Number features in GMM init: ", n_features_gmm)

# assert that number of features consistent between KMeans and GMM
assert n_features_kmeans == n_features_gmm == n_features_pca, "Number of features mismatch between KMeans and GMM initialization"

# fit GMM models with varying number of components
aic_values = []
bic_values = []

for k in range(1, 6):
    gmm = GaussianMixture(n_components=k, random_state=67)
    gmm.fit(projected_features)
    aic_values.append(gmm.aic(projected_features))
    bic_values.append(gmm.bic(projected_features))

# Elbow plot for BIC
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, 6), bic_values, marker='o', linestyle='-', label='BIC')
plt.title('BIC')
plt.xlabel('K')
plt.xticks(range(1, 6))

# Elbow plot for AIC
plt.subplot(1, 2, 2)
plt.plot(range(1, 6), aic_values, marker='o', linestyle='-', label='AIC')
plt.title('AIC')
plt.xlabel('K')
plt.xticks(range(1, 6))

plt.tight_layout()
plt.show()

# 2d - visualize scatter of features by diagnostic group then by GMM cluster using
# estimated mean vectors and covariance matrices for each cluster

# set up subplots
plt.figure(figsize=(12, 6))

# diagnostic
plt.subplot(1, 2, 1)

# initialize colors
colors = {'CN': 'blue', 'MCI': 'purple', 'AD': 'red'}

# keep only first visit for each subject
baseline_visits = year_bl_0_df.reset_index()

# filter based on the "DX" column
for diagnosis in np.unique(baseline_visits['DX']):
    indices = baseline_visits[baseline_visits['DX'] == diagnosis].index

    # filter baseline_visits for CNs
    if diagnosis == 'CN':
        cn_group = baseline_visits[baseline_visits['DX'] == 'CN']

     # filter projected_features based on indices of baseline_visits
    selected_features = projected_features[np.isin(np.arange(len(projected_features)), indices)]

    # filter baseline_visits for MCIs
    if diagnosis == 'MCI':
        mci_group = baseline_visits[baseline_visits['DX'] == 'MCI']

    # filter projected_features based on the indices of baseline_visits
    selected_features = projected_features[np.isin(np.arange(len(projected_features)), indices)]

    # filter baseline_visits for ADs
    if diagnosis == 'AD':
        ad_group = baseline_visits[baseline_visits['DX'] == 'AD']

    # filter projected_features based on the indices of baseline_visits
    selected_features = projected_features[np.isin(np.arange(len(projected_features)), indices)]

    # plot scatter points for diagnostic group
    plt.scatter(selected_features[:, 0], selected_features[:, 1], label=diagnosis, color = colors[diagnosis])

# fit GMM model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(projected_features)

# add labels and legend
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot by diagnostic group')

# define legend label order
legend_order = ['CN', 'MCI', 'AD']

# get handles and labels of scatter plots
handles, labels = plt.gca().get_legend_handles_labels()

# make dictionary to map labels to handles
label_handles = dict(zip(labels, handles))

# make list of handles in desired order
ordered_handles = [label_handles[label] for label in legend_order]

# make list of labels in desired order
ordered_labels = [label for label in legend_order]

# plot legend with ordered handles and labels
plt.legend(ordered_handles, ordered_labels)

# GMM clusters

plt.subplot(1, 2, 2)

# fit GMM model
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(projected_features)

# compute cluster means & covariances
cluster_means = gmm.means_
cluster_covs = gmm.covariances_

# make predicted labels
predicted_labels = gmm.predict(projected_features)

# initialize colors
colors = {'cluster 1': 'red', 'cluster 2': 'blue'}

# plot scatter points colored by predicted cluster labels
for label, color in zip(np.unique(predicted_labels), ['red', 'blue']):
    cluster_data = projected_features[predicted_labels == label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}', color=color)

# plot ellipses for each GMM cluster
for i in range(len(cluster_means)):
    cov_matrix = cluster_covs[i]
    lambda_, v = np.linalg.eig(cov_matrix)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=cluster_means[i], width=lambda_[0]*4, height=lambda_[1]*4,  # enlarge ellipses
                  angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])), edgecolor='none')  # rotate ellipses
    if i == 0:
        ell.set_facecolor('red')  # fill with red for the first cluster
    else:
        ell.set_facecolor('blue')   # fill with blue for the second cluster
    ell.set_alpha(0.1)  # set transparency
    plt.gca().add_patch(ell)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot by GMM Cluster')

# create custom legend labels
legend_labels = [f'cluster {i+1}' for i in range(len(cluster_means))]

# plot legend with custom labels
plt.legend(labels=legend_labels, loc='upper left')

plt.xticks(np.arange(-4, 8, 2))  # Set x-axis ticks at intervals of 2
plt.yticks(np.arange(-2, 6, 2))  # Set y-axis ticks at intervals of 2

# show plot
plt.show()

# 2e calculate and plot log odds of being in disease cluster & plot for 2-yr MCI converters & non

# Filter the data for baseline MCI subjects with at least one visit 2+ years later
baseline_mci = a_merge_df[(a_merge_df['DX'] == 'MCI') & (a_merge_df['years_bl'] == 0)]
mci_followup = a_merge_df[(a_merge_df['DX'] == 'MCI') & (a_merge_df['years_bl'] >= 2)].drop_duplicates(subset='RID', keep='first')
ad_followup = a_merge_df[(a_merge_df['DX'] == 'AD') & (a_merge_df['years_bl'] >= 2)].drop_duplicates(subset='RID', keep='last')

# Merge baseline MCI and follow-up data - these are the non-converters
non_converters = baseline_mci.merge(mci_followup[['RID', 'DX']], on='RID', suffixes=('_baseline', '_followup'))

# Merge baseline_mci and ad_followup - these are the converters
converters = pd.merge(baseline_mci, ad_followup, on='RID', suffixes=('_baseline', '_followup'))

# Fit GMM model for baseline data
gmm_baseline = GaussianMixture(n_components=2, random_state=67)
gmm_baseline.fit(projected_features)

# Predict cluster labels for converters and non-converters using the baseline GMM model
converters['baseline_cluster'] = gmm_baseline.predict(projected_features)
non_converters['baseline_cluster'] = gmm_baseline.predict(projected_features)

# Calculate the log probabilities of each sample for each cluster
log_probs_converters = gmm_baseline.predict_log_proba(projected_features)
log_probs_non_converters = gmm_baseline.predict_log_proba(projected_features)

# Calculate the log odds of being in the disease cluster
log_odds_converters = log_probs_converters[:, 1] - log_probs_converters[:, 0]
log_odds_non_converters = log_probs_non_converters[:, 1] - log_probs_non_converters[:, 0]

# Add log odds to the data frames
converters['Log Odds'] = log_odds_converters
non_converters['Log Odds'] = log_odds_non_converters

# Combine converters and non-converters data frames
combined_data = pd.concat([converters, non_converters])

# Plot box and scatter plots for converters and non-converters together
plt.figure(figsize=(10, 6))
sns.boxplot(x='conversion_status', y='Log Odds', data=combined_data, palette='Set2')
sns.stripplot(x='conversion_status', y='Log Odds', data=combined_data, jitter=True, color='black', alpha=0.5)
plt.title('Log Odds of Being in Disease Cluster for Baseline MCI Subjects')
plt.xlabel('Conversion Status')
plt.ylabel('Log Odds')
plt.show()