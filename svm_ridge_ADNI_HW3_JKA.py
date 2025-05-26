import pandas as pd
import numpy as np
import glob2
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from compute_atlas_ROIs import load_ROI_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge

#complete this for Q1
if __name__ == '__main__':
# ***************** ADNIMERGE part
# needs: pip install xlrd==1.2.0
    df_raw = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    df_raw['Age_visit'] = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns={'PTGENDER': 'Sex', 'PTEDUCAT': 'Education_Years'}, inplace=True)

# we care about all the basics as before, plus Image UID
measures_we_care_about = ['RID', 'DX', 'MMSE', 'IMAGEUID']
df = df_raw[measures_we_care_about]
print(df.shape)

# ****************** CREATE IMAGING FEATURES
# *** load images
# get the file names we want to load
nii_files = glob2.glob('../data_lesson3/ADNI_60_mwrc1/mwrc1*.nii')

# get the Image UIDs from the raw filenames
imageUIDs = np.array([int(str_i.split('_')[-1].split('.')[0][1:]) for str_i in nii_files])

# get the atlas - in this case it's the Harvard-Oxford cortical atlas
nii_atlas = nib.load('../data_lesson3/HarvardOxford_Cortical_warped.nii')
atlas_mat = nii_atlas.get_fdata()
# compute ROI means based on the atlas
X, atlas_vals_vec = load_ROI_matrix(nii_files, atlas_mat)

# sort them from smallest to largest
index_sorted = np.argsort(imageUIDs)
X = X[index_sorted, :]
imageUIDs = imageUIDs[index_sorted]

# intersect ADNIMERGE with available image UIDs and sort them
df = df.loc[np.isin(df.IMAGEUID, imageUIDs)]
df = df.sort_values(by=['IMAGEUID'])
assert (np.all(df.IMAGEUID == imageUIDs))

n = df.shape[0]

# coding AD vs (CN + MCI)
y_svc = np.array(df.DX == 'Dementia').astype(int).reshape(n, 1)
# change from 0/1 to -1/1 labelling to be consistent with SVC setup
y_svc[y_svc == 0] = -1

def cross_validated_SVM(X, y):

    n_total                 = X.shape[0]

    #using 10-fold cross-validation here
    num_folds               = 10

    #possible C-cost values
    SVC_C_vals              = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1, 100])

#create evenly-spaced values based on n-total (starts at 0, so will be n_total -1):
    all_indices             = np.arange(n_total)

#make sure our random values stay the same throughout reruns
    np.random.seed(1)

#list all possible permutations given the input values
    shuffled_indices        = np.random.permutation(all_indices)

#evenly splits up shuffled_indices into 10 groups (= NUM_FOLDS)
    list_indices            = np.array_split(shuffled_indices, num_folds)

#set up zero matrices for training and validation set values to live in
    error_training_mat      = np.zeros((num_folds, len(SVC_C_vals)))
    error_validation_mat    = np.zeros((num_folds, len(SVC_C_vals)))

#don't forget to scale training/validation data properly
    scaler                  =  StandardScaler(copy=True, with_mean=True, with_std=True)

    #****FILLING IN
    #
    for i, fold_indices in enumerate(list_indices):

        #set validation set
        validation_set_X = X[fold_indices]
        validation_set_y = y[fold_indices]
        validation_set_y = validation_set_y.ravel()


        #set training set
        train_indices = np.concatenate([fold_indices for j, fold_indices in enumerate(list_indices) if j != i])
        train_set_X = X[train_indices]
        train_set_y = y[train_indices]
        train_set_y = train_set_y.ravel()

        #scale data
        scaler.fit(train_set_X)
        scaled_train_X = scaler.transform(train_set_X)
        scaled_val_X = scaler.transform(validation_set_X)

        for j, C_val in enumerate(SVC_C_vals):

            #train SVM model
            svm_model = LinearSVC(C=C_val)
            svm_model.fit(scaled_train_X, train_set_y)

            #predict/calculate errors
            train_predictions = svm_model.predict(scaled_train_X)
            val_predictions = svm_model.predict(scaled_val_X)

            train_error = np.mean(train_predictions != train_set_y)
            val_error = np.mean(val_predictions != validation_set_y)

            #save errors
            error_training_mat[i, j] = train_error
            error_validation_mat[i, j] = val_error

    #calculate mean training and validation errors across all folds for each model
    #when you've written the above code, uncomment this

    mean_training_errors = np.mean(error_training_mat, axis=0)
    mean_validation_errors = np.mean(error_validation_mat, axis=0)

        #pick best model based on lowest mean validation error
    best_model_index = np.argmin(mean_validation_errors)
    best_C_value = SVC_C_vals[best_model_index]
    best_mean_error = mean_validation_errors[best_model_index]

    #print best model
    print('Best model is C = %f, with mean validation misclassification error of %.2f%%.' % (
    best_C_value, best_mean_error))

    #create the bias-variance trade-off figure
    plt.plot(np.log10(SVC_C_vals), mean_training_errors, color='b', marker='.', markersize="3", label='training error')
    plt.plot(np.log10(SVC_C_vals), mean_validation_errors, color='r', marker='.', markersize="3", label='validation error')
    plt.xlabel('Model complexity: Log10 C', fontsize=12)
    plt.ylabel('Prediction error',fontsize=12)
    plt.legend()
    plt.show()

#train up best SVC model on full dataset (original X, y)
#don't forget to standardize
# **** FILL THIS IN
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    svc_best = LinearSVC(C=best_C_value).fit(X_scaled, y)
#**** FILL THIS IN
    #get coefficients from full model and return them
    coef_full = svc_best.coef_

    #when you've written the above code, uncomment this
    return coef_full

#pass

#complete this for Q2

#very similar to above, but using root mean squared error (RMSE)
    #to assess train/validation error within each fold

    #tune the ridge regression using these lambda values: 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8
def cross_validated_ridge(X, y):

    # **** FILL THIS IN

    n_total = X.shape[0]

    # using 10-fold cross-validation here
    num_folds = 10

    # possible Lambda values
    Lamb_vals = np.array([1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8])

    # create evenly-spaced values based on n-total (starts at 0, so will be n_total -1):
    all_indices = np.arange(n_total)

    # make sure our random values stay the same throughout reruns
    np.random.seed(1)

    # list all possible permutations given the input values
    shuffled_indices = np.random.permutation(all_indices)

    # evenly splits up shuffled_indices into 10 groups (= num_folds)
    list_indices = np.array_split(shuffled_indices, num_folds)

    # set up zero matrices for training and validation set values to live in
    error_training_mat = np.zeros((num_folds, len(Lamb_vals)))
    error_validation_mat = np.zeros((num_folds, len(Lamb_vals)))

    # don't forget to scale training/validation data properly
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

    #make homes for best_mean_error and best_Lamb_value
    best_mean_error = float('inf')
    best_Lamb_value = None

    # ****FILLING IN
    #
    for i, fold_indices in enumerate(list_indices):
        # set validation set
        validation_set_X = X[fold_indices]
        validation_set_y = y[fold_indices]
        # set training set
        train_indices = np.concatenate([fold_indices for j, fold_indices in enumerate(list_indices) if j != i])
        train_set_X = X[train_indices]
        train_set_y = y[train_indices]
        train_set_y = train_set_y.ravel()
        # scale data
        scaler.fit(X)
        scaled_train_X = scaler.transform(train_set_X)
        scaled_val_X = scaler.transform(validation_set_X)

        for j, R_Lamb_vals in enumerate(Lamb_vals):
            # train RMSE model
            Lamb_model = Ridge(alpha=R_Lamb_vals)
            Lamb_model.fit(scaled_train_X, train_set_y)
            # predict/calculate errors
            R_train_predictions = Lamb_model.predict(scaled_train_X)
            R_val_predictions = Lamb_model.predict(scaled_val_X)
            Lamb_train_error = np.sqrt(np.mean((R_train_predictions.ravel() - train_set_y) ** 2))
            Lamb_val_error = np.sqrt(np.mean((R_val_predictions - validation_set_y.ravel()) ** 2))
            # save errors
            error_training_mat[i, j] = Lamb_train_error
            error_validation_mat[i, j] = Lamb_val_error
            if Lamb_val_error < best_mean_error:
                best_mean_error = Lamb_val_error
                best_Lamb_value = R_Lamb_vals

    print('Best model is Lambda = %f, with mean validation RMSE of %.2f%%.' % (best_Lamb_value, best_mean_error))

    # calculate mean training and validation errors across all folds for each model
    # when you've written the above code, uncomment this

    mean_training_errors = np.mean(error_training_mat, axis=0)
    mean_validation_errors = np.mean(error_validation_mat, axis=0)

#plot bias-variance graph
    plt.plot(-np.log10(Lamb_vals), mean_training_errors, color='b', marker='.', markersize="3", label='training error')
    plt.plot(-np.log10(Lamb_vals), mean_validation_errors, color='r', marker='.', markersize="3",
             label='validation error')
    plt.xlabel('Model complexity: -Log10 Lambda', fontsize=12)
    plt.ylabel('Prediction error', fontsize=12)
    plt.legend()
    plt.show()

    # train up best RMSE model on full dataset (original X, y)
    # don't forget to standardize
    # **** FILL THIS IN
    scaler.fit(X)
    scaled_X = scaler.transform(X)
    Lamb_best = Ridge(alpha=best_Lamb_value)
    Lamb_best.fit(scaled_X, y)

    # **** FILL THIS IN
    # get coefficients from full model and return them
    coef_full = Lamb_best.coef_

    # when you've written the above code, uncomment this
    return coef_full

    #pass

#visualize the coefficients of classifier/regression as a brain image
def plot_weight_map(beta_vec, atlas_vals_vec, atlas_mat):

    #initialize the brain image of beta (i.e. coefficient) values and set background values (i.e. not in atlas) to NaN
    beta_image              = np.zeros(atlas_mat.shape)
    beta_image[:]           = np.NaN

    #create beta image using atlas_vals_vec (the vector of atlas values that corresponds to beta_vec) and atlas_mat
    #******* FILL THIS IN
    beta_vec = beta_vec.reshape(48,1)
    for u, roi in enumerate (atlas_vals_vec):
        beta_image[atlas_mat == int(roi)] = beta_vec[u]

    mid_indices             = np.array(atlas_mat.shape) // 2

    #************ visualize coefficients as a brain

    #choose the 'jet' colormap and set the 'bad' values (i.e. NaNs) to black, creating a black background
    import copy
    cmap_local              = copy.copy(plt.cm.jet)
    cmap_local.set_bad(color='black')

    V_ABS_MAX               = np.max(np.abs(beta_vec))

    plt.subplot(2, 2, 1)
    img                     = plt.imshow(np.rot90(beta_image[mid_indices[0], :, :]), vmax=V_ABS_MAX, vmin=-V_ABS_MAX, cmap=cmap_local)
    plt.colorbar(img)

    plt.subplot(2, 2, 2)
    img                     = plt.imshow(np.rot90(beta_image[:, mid_indices[1], :]), vmax=V_ABS_MAX, vmin=-V_ABS_MAX, cmap=cmap_local)
    plt.colorbar(img)

    plt.subplot(2, 2, 3)
    img                     = plt.imshow(np.rot90(beta_image[:, :, mid_indices[2]]), vmax=V_ABS_MAX, vmin=-V_ABS_MAX, cmap=cmap_local)
    plt.colorbar(img)

    #done outside function
    #plt.show()

    # ****************** SVC PART
print('***** Building AD vs (MCI + CN) classifier')
beta_svm = cross_validated_SVM(X, y_svc)

plot_weight_map(beta_svm, atlas_vals_vec, atlas_mat)
plt.suptitle('SVC coefficients from AD vs (MCI + CN) classifier')
plt.show()

    # ****************** RIDGE PART
print('***** Building ridge-based predictor of MMSE ')
y_ridge                 = np.array(df.MMSE).reshape(n,1)
beta_ridge              = cross_validated_ridge(X, y_ridge)

plot_weight_map(beta_ridge, atlas_vals_vec, atlas_mat)
plt.suptitle('Ridge regression coefficients from MMSE predictor')
plt.show()
