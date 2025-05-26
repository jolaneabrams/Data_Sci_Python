import pandas as pd
import numpy as np
import glob
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from compute_atlas_ROIs import load_ROI_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import Ridge

#complete this for Q1
def cross_validated_SVM(X, y):

    n_total                 = X.shape[0]
    #X.shape[0] is indicating that n_total will be just the first value of what .shape is equal to.
    # ex: .shape by itself will output the size of a matrix ex: (3,4)
    # using .shape[0] is saying take just the value in the 0th place - in the ex, .shape[0] = 3

    #using 10-fold cross-validation here
    NUM_FOLDS               = 10

    #possible C-cost values
    SVC_C_vals              = np.array([1e-8, 1e-6, 1e-4, 1e-2, 1, 100])

    #********************** cross-validation section *************************

    #so for the record he doesn't want you to actually DO this- this is just fun information on how easy a cross validation is
    #but he then has us do it the DIY way so we learn the ins and outs instead
    #anything that doesn't say COMMENT or FILL IN is not part of the HW basically

    # ******* the scikit learn version - the way you do it in the 'real-world'
    # #create the K-fold cross-validation object
    # from sklearn import model_selection
    # cv                      = model_selection.KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=1)
    #
    # i                       = 0
    # for train_indices_i, validation_indices_i in cv.split(X):
    #
    #     #create train/test split for this CV fold
    #     X_train_i, X_val_i = X[train_indices_i, :], X[validation_indices_i]
    #     y_train_i, y_val_i = y[train_indices_i],    y[validation_indices_i]
    #
    #     ... rest of code ...

    # ******* non-scikit learn verson, to better understand how it all works
    #COMMENT:
    all_indices             = np.arange(n_total)
    #we used .arange on the last HW, it's creating evenly spaced values based on the n_total variable here- remember that
    # its up until the input value but doesnt include the final value (so np.arange(5) would be 0,1,2,3,4)

    np.random.seed(1)
    #again like the last HW this is making sure our random values stay the same thru reruns
    shuffled_indices        = np.random.permutation(all_indices)
    #.permutation() provides a list of all possible permutations given the input values

    #COMMENT:
    list_indices            = np.array_split(shuffled_indices, NUM_FOLDS)
    #so np.array_split(input array, # ) basically evenly splits up an inputted array (here the shuffled_indices) into how many
    # groups as was input as the # wanted. ex: np.arange(array, 3) would split the array into 3 groups with equal amt of items in each (or
    #as equal as possible, there might be one group with one more or less than others if it can't be split perfectly even.)

    #COMMENT:
    error_training_mat      = np.zeros((NUM_FOLDS, len(SVC_C_vals)))
    #similar to the last HW, this is creating a zeros matrix that has  x rows, and y columns based on the input variables
    # and these are being made for the training and validiation sets we will do
    error_validation_mat    = np.zeros((NUM_FOLDS, len(SVC_C_vals)))

    #don't forget to scale training/validation data properly
    scaler                  =  StandardScaler(copy=True, with_mean=True, with_std=True)
    # just a common way to standarize and scale values going forward, this is one of scikit's abilities so make sure its imported
    #when you want to use it ( luckily it already is here)
    # here specifically we are setting it up (to center data w men and scale w std) to use later - to use it we call on the variable we attached it to (scaler)

    #**** FILL THIS IN
    # outer loop through list_indices:
        #HINT FOR THIS PART:
         # we need to set up a for loop that is going to iterate thru every list created in list_indices
            #so its like our double for loop - and remember we want to place these in our empty training/validation matrices at the end
         #   so we are going to use the enumerate so the final inputs know where to go in that matrix (since we use i,j ) to place it

    #   1. setting current list as validation set, remainder as training set for current fold
        #HINT: first make a copy of the array so we don't mess with the real data
        # then we need to take the current list we have out of the  array - we can do this by using the .pop(i) command - it removes
    #   the item of interest from the list entirely so your new variable has the popped list but the old variable doesn't

    #   2. creating current training and validation split of X and y
        #HINT: easiest way is actually the way he showed in the "cross validation q scikit " up above
    #     #create train/test split for this CV fold
    #     X_train_i= X[train_indices_i, :]
    #     X_val_i = X[validation_indices_i]
    #     y_train_i, y_val_i = y[train_indices_i, :],    y[validation_indices_i]
        #where you are making the training = to the current list youre on in your for loop and the validation the rest of the lists


    #   3. scaling data appropriately
        #HINT:this is back to the scaler variable dr aksman made earlier, to use it  we do variable = scaler.fit(training_data_variable)
        # we only want to scale the training data since that's what we compare vs the validation set here
        #ISSUE THAT WILL ARISE: the scaler model can only take in 2D objects - but the X_train_i and y_train_i are 3D - they should be (9x6)x48(or1)
        #(they are this size due to the fact the train indicies are a list of 9 arrays and every array has 6 values and then a # of columns)
        #so to fix this we need to reshape the dfs and turn them 2D before trying to fit them to the svm model
        # we just need to have the 9 listed arrays all compile together into one - we can do this by using the .reshape() command
        # it would look something like X_train_i_reshape = X_train_i.reshape(#_of_list_items*#_of_rows , #_of_columns )
        #if you dont want to put the specific # for these you can also just call on the X_train_i.shape[0] and .shape[1] etc to get the list# row# column#


    # inner loop through all possible C values
        #HINT: the inner for loop needs to enumerate thru SVC_C_vals
       # for i, SVC_i in enumerate(SVC_C_vals)

    #    1. training current model
        #HINT: the setup of LinearSVC is LinearSVC( C value wanted)
        # it'll look something similar to svc_model = svm.LinearSVC( whatever your variable name for the SVC_i in SVC_C_vals is)
        #so with the above line you've set up your linear svc and now we can actually train the model
        # so we need to .fit it
        # ex trained_svc = svc_model.fit(X_scaled_training_set, y_training_set)
        #the thing to look out for here is sometimes theres an issue with shape - scaler only likes 2D data and you might have to use .reshape
        #to fix the arrays so itll input correctly - can also try .ravel(order='C') that will merge everything into one list but keep the order

    #    2. predicting training and validation labels
        #HINT: we use the model we've created to get the predictions!
        # do this via training_prediciton = trained_svc.predict(X_scaled_training_set)
        # same for validation set -> validation_prediction = trained_svc.predict(X_scaled_validation_set)

    #    3. saving current training and validation errors
        #HINT: for this we are inputting the error value into the premade matrices, error_training_mat and error_validation_mat
        #simialr to the last hw we'd do error_training_mat [i] [j] =
        # to get the error we are basically looking for how many values in our training set didnt match our predicitons and finding the percentage
        # so we can do that via np.sum( scaled_training_set != validation prediction) / len(scaled_training_set) and then do the same
        #thing for the validation errors

    #calculate mean training and validation erros across all folds for each model
    #when you've written the above code, uncomment this
    #mean_training_errors         = np.mean(error_training_mat, axis=0)
    #mean_validation_errors       = np.mean(error_validation_mat, axis=0)

    # select best model based on lowest mean validation error
    # **** FILL THIS IN

    #HINT: to find the best model based on low mean validation error you can use np.argmin
    # example of how this functions: https://www.geeksforgeeks.org/numpy-argmin-python/
    # best_model = np.argmin(mean_validation_errors) -> this will go thru each value in the matrix and output where in the matrix
        #has the best model - note this gives you the LOCATION (the indicies) of the minimum, not the value of the minimum
        #to get the value you just apply best_model to the matrix : best_mean = mean_validation_errors[best_model] -
    #   generally in python the [] usually indicate location w/n a matrix while () will usually indicate value/name
    #then you do the same thing with the svc_C_vals and best_model to get the C_best

    #print best model when you're ready
    #print('Best model is C = %f, with mean validation misclassification error of %.2f%%.' % (C_best, best_mean_error))

    #create the bias-variance trade-off figure
    # **** FILL THIS IN

    #HINT: copy paste the way you did the plot from HW two - you need most of the same info we're just gonna change some
    #values and variables
    #tho you can just use plt.figure(1) since we are only making one figuree for this part instead of 2
    # importantly dr aksman wants mean prediction error vs log10
    #to do the log10 use np.log10
    #plt.plot(np.log10(SVC_C_vals), mean_training_errors, color='', label='')
    #do it again for the validation errors and then just start changing the X axis label and y label
    # and mess with the X_lim and y_lim if needed
    #then show legend and add a title and do plt.show() !

#your X_train is your train_set_x variable(aka the original X your X_train is your train_set_x variable (aka the original X dataframe w real data, but only the rows of interest corresponding to
# what your current fold_indicies list has) which you then scaled and it became the scaled_train_x you can put into the svm model.

#your current_indices is your fold_indices and train_indicies variable (the fold_indices being your current validation indices group and the train_indices being the other 9 of the 10 fold_indicies).
#the validation_set_X/y and train_set_X/y are our real data (X and y_svm) but split into training and validation set
#from a bird's eye view what we're doing is seeing if the training set of all the other values can predict the validation set values we've set aside correctly and find the error % of how wrong it is.
# and we test using many different SSVC_C_vals to see which svm model outputs the closest to the real validation values consistently (which is why we loop thru the training/validation multiple times
# and make each fold_indicies our validation set once as we loop thru)
#for this line:
#don't forget to standardize
# **** FILL THIS IN
    svc_best = svm.LinearSVC(C=C_best).fit(X, y)
#make sure to standardize/scale the X before placing into the SVC model (earlier we scaled the train_set_X but not the X itself)

#train up best SVC model on full dataset (original X, y)
    #don't forget to standardize
    # **** FILL THIS IN
    #HINT: just a mini version of what we did before
        #but we are using the original X and y from the top
        # so  apply the scaler model we made (the original) so like scaler.fit(X) to scale and standardize
        #then use the svc model with C = to C_best so we fit the OG data to the best C ( so it'll look something like
        # svc_best = svm.LinearSVC(C=C_best).fit(X,y)


    #get coefficients from full model and return them
    # **** FILL THIS IN
    #HINT: this isn't too bad - we can use the nifty command of .coef_ on the svc_best we just made to obtain coefficients
    # so just have the coef_full variable equal the svc_best matrix.coeff_


    # when you've written the above code, uncomment this
    ##return coef_full

    pass

#complete this for Q2
def cross_validated_ridge(X, y):

    # **** FILL THIS IN

    #very similar to above, but using root mean squared error (RMSE)
    #to assess train/validation error within each fold


    #tune the ridge regression using these lambda values: 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8

    #HINT: truly this is almost exactly the same from beginning to end - the only thing is instead of SVC_C_vals  we use these
        #lambda values instead besides that basically just copy/paste from above and replace the SVC_C_vals variable with the lambda one
        #other primary thing is we are using a ridge model here so: ridge_model = Ridge(alpha='current_lambda_vale') - and then you
        #need to fit it like before
        # the only other thing is to find RMSE (instead of error amt this time) we use the sqrt(mean((predict- target)*2)) equation for both
        # the training and validation prediction and y_ridge modified sets
    #remember to plot it as well!


    #when you've written the function, uncomment this:
    ##return coef_full

    pass



#visualize the coefficients of classifier/regression as a brain image
def plot_weight_map(beta_vec, atlas_vals_vec, atlas_mat):

    #initialize the brain image of beta (i.e. coefficient) values and set background values (i.e. not in atlas) to NaN
    beta_image              = np.zeros(atlas_mat.shape)
    beta_image[:]           = np.NaN

    #create beta image using atlas_vals_vec (the vector of atlas values that corresponds to beta_vec) and atlas_mat
    #******* FILL THIS IN
        #HINT: if you get size errors make sure to reshape the beta_vec so that all values are in 1 column - beta_vec.reshape(:,1)
        #to get this beta image values, we have to make a small for loop that goes thru the atlas_vals_vec so it can find each beta image value
        # so start with having the foor loop enumerate thru the atlas_vals_vec and the goal is to code for
        # replacing the empty beta_image matrix values where the atlas_mat equals the atlas_vals_vec currently being called on with the beta_vec value
        #bc like it says the atlas_vals_vec is connected to beta_vec (raw values v beta vals) - the beta iamge matrix
        # is basically being told where in the atlas_mat the beta_vec values equal the atlas_vals_vec values
        #atlas_vals_vec grabs the ROI means based on the atlas while atlas_mat is just the raw data.
#you use this function for both parts - everything in the function stays exactly the same the only difference is when u run the
        #entire function it'll be  plot_weight_map(beta_svm, atlas_vals_vec, atlas_mat) for the first part (looking at beta_svm)
        # vs plot_weight_map(beta_ridge, atlas_vals_vec, atlas_mat) for the second time around (looking at the ridge) these are
        #lines 306 and 315 respectively) - bc this code is made up of fucntions remember to run all variables in the if __main__ statement
        #in order to mess around with values or ur variables wont be assigned!

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



#so above a bunch of different functions were made - functions are activated via being called upon  cross_validated_SVM(X, y)
if __name__ == '__main__':

    #***************** ADNIMERGE part
    # needs: pip install xlrd==1.2.0
    df_raw              = pd.read_excel('../data_lesson4/ADNIMERGE_thin.xlsx')

    df_raw['Age_visit']  = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns  = {'PTGENDER':'Sex', 'PTEDUCAT':'Education_Years'}, inplace=True)

    #we care about all the basics as before, plus Image UID
    measures_we_care_about = ['RID', 'DX', 'MMSE', 'IMAGEUID']
    df                  = df_raw[measures_we_care_about]
    print(df.shape)

    #****************** CREATE IMAGING FEATURES
    # *** load images
    #get the file names we want to load
    nii_files               = glob.glob('../data_lesson3/ADNI_60_mwrc1/mwrc1*.nii')

    #get the Image UIDs from the raw filenames
    imageUIDs               = np.array([int(str_i.split('_')[-1].split('.')[0][1:]) for str_i in nii_files])

    #get the atlas - in this case it's the Harvard-Oxford cortical atlas
    nii_atlas               = nib.load('../data_lesson3/HarvardOxford_Cortical_warped.nii')
    atlas_mat               = nii_atlas.get_fdata()
    #compute ROI means based on the atlas
    X, atlas_vals_vec       = load_ROI_matrix(nii_files, atlas_mat)

        #so this is the where a lot of our values we input into our functions above are stemming from


    #sort them from smallest to largest
    index_sorted            = np.argsort(imageUIDs)
    X                       = X[index_sorted, :]
    imageUIDs               = imageUIDs[index_sorted]

    #intersect ADNIMERGE with available image UIDs and sort them
    df                      = df.loc[np.isin(df.IMAGEUID, imageUIDs)]
    df                      = df.sort_values(by = ['IMAGEUID'])
    assert(np.all(df.IMAGEUID==imageUIDs))

    n                       = df.shape[0]

    #coding AD vs (CN + MCI)
    y_svc                   = np.array(df.DX == 'Dementia').astype(int).reshape(n, 1)
    #this codes for any DX that is Dementia is = 1 and anything else is 0

    #change from 0/1 to -1/1 labelling to be consistent with SVC setup
    y_svc[y_svc==0]         = -1

    # ****************** SVC PART
    print('***** Building AD vs (MCI + CN) classifier')
    beta_svm                = cross_validated_SVM(X, y_svc)
    #this is the X,y being put into the function we make above - so to run this line by line u need to put in the X variable and the y_svc variable
    #this is comparing all subjects and then looking at the AD(1) v CN/MCI(-1) - the order matches subj to DX bc we index sorted them

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
