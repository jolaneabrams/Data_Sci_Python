import pandas as pd
import glob2
import nibabel as nib
from compute_atlas_ROIs import load_ROI_matrix
import numpy as np
import matplotlib.pyplot as plt
import basic_ADNI_analysis
import scipy.stats as stats

if __name__ == '__main__':

    #***************** ADNIMERGE part
    # needs: pip install xlrd==1.2.0
    df_raw              = pd.read_excel('../ADNIMERGE_thin.xlsx')

    df_raw['Age_visit']  = df_raw['AGE_bl'] + df_raw['Years_bl']
    df_raw.rename(columns  = {'PTGENDER':'Sex', 'PTEDUCAT':'Education_Years'}, inplace=True)

    #we care about all the basics as before, plus Image UID
    measures_we_care_about = ['RID', 'VISCODE','Age_visit','Years_bl', 'Sex', 'Education_Years',  'DX', 'MMSE', 'IMAGEUID']
    df                  = df_raw[measures_we_care_about]
    print(df.shape)

    #****************** IMAGING PART
    #just like in compute_atlas_ROIs.py

    #get the file names we want to load
    nii_files               = glob2.glob('../data_lesson3/ADNI_60_mwrc1/mwrc1*.nii')

    #get the atlas - in this case it's the Harvard-Oxford cortical atlas
    nii_atlas               = nib.load('../data_lesson3/HarvardOxford_Cortical_warped.nii')
    atlas_mat               = nii_atlas.get_fdata()

    #*********** PREVIOUS SECTION
    # df_atlas                = pd.read_csv('../data_lesson3/HarvardOxford_mapping.csv')
    #
    # #create dictionary, mapping label integer to label string
    #dict_atlas              = {}
    #for i, roi in enumerate(df_atlas['ROI_Name']):
    #   dict_atlas[df_atlas['ROI_Label'][i]] = df_atlas['ROI_Name'][i]

    #*********** NEW SECTION: fill this in
    df_lobes                = pd.read_excel('../homework/HarvardOxford_mapping_4lobes.xlsx')

    #create a new atlas based on the lobes we want
    #each lobe is composed of a set of Harvard-Oxford atlas ROIs
    atlas_mat_lobes         = np.zeros(atlas_mat.shape)

    #create the dictionary that maps New_Label (integer) to Lobe (string)
    dict_atlas              = {}

    #we want to build up dict_atlas and atlas_mat_lobes and we can do it with a single nested loop
    #OUTER LOOP: go through each row of df_lobes, build up dict_atlas
    #INNER LOOP: assign new label to atlas_mat_lobes in the locations where atlas_mat's label is in the current lobe's ROI list
    #
    # FILL THIS IN
    #
    #**********************************


    #compute ROI means based on the atlas
    roi_means, atlas_num_vals = load_ROI_matrix(nii_files, atlas_mat_lobes)

    #get the Image UIDs from the raw filenames
    # example filename: #ADNI_024_S_4084_MR_MT1__GradWarp__N3m_Br_20111008155405044_S122287_I260273.nii
    imageUIDs               = np.array([int(str_i.split('_')[-1].split('.')[0][1:]) for str_i in nii_files])

    #get sorted indices and re-index roi_means rows based on it
    index_sorted            = np.argsort(imageUIDs)
    imageUIDs               = imageUIDs[index_sorted]
    roi_means               = roi_means[index_sorted, :]


    #******************** WANT THE DX: INTERSECT ADNIMERGE WITH IMAGING
    df                      = df.loc[df['IMAGEUID'].isin(imageUIDs)]

    #make sure they're in the same order
    df                      = df.sort_values(by = ['IMAGEUID'])
    assert(np.all(df.IMAGEUID==imageUIDs))

    #******************* ROI GROUP DIFFS

    #We're doing a bunch of comparisons, so we need some kind of multiple comparisons correction
    #Simplest one: Bonferroni - multiply the p-val by number of tests (here: # ROIs)
    Bonferroni_factor       = roi_means.shape[1]

    nlog_p_val_mat          = np.zeros(atlas_mat.shape)

    for i in range(0, roi_means.shape[1]):
        cn_i, mci_i, ad_i   = roi_means[df.DX=='CN', i], roi_means[df.DX=='MCI', i], roi_means[df.DX=='Dementia', i]
        F_i, p_i            = stats.f_oneway(cn_i, mci_i, ad_i)

        #multiply raw p-value by Bonferroni factor - increase the threshold for significance
        p_i_corrected       = p_i * Bonferroni_factor

        str_sig             = '*' if p_i_corrected < 0.05 else ' '

        nlog10_p            = -np.log10(p_i_corrected)

        print('%50s F = %4.1f, raw p-value: %.5f, corrected p-val: %.5f %s -log10: %3.1f' % (dict_atlas[atlas_num_vals[i]], F_i, p_i, p_i_corrected, str_sig, nlog10_p))

        nlog_p_val_mat[atlas_mat_lobes == atlas_num_vals[i]] = nlog10_p

    #********* FROM 4.5 to 5
    #visualize the negative log p-val matrix on the same scale (-10 log p from 0 to 4.5)
    ax1         = plt.subplot(2, 2, 1)
    cax         = ax1.matshow(np.rot90(nlog_p_val_mat[60,:,:]), vmax=0, vmin=4.5, cmap=plt.cm.gist_heat)
    plt.colorbar(cax)

    ax2         = plt.subplot(2, 2, 2)
    cax         = ax2.matshow(np.rot90(nlog_p_val_mat[:,72,:]), vmax=0, vmin=4.5, cmap=plt.cm.gist_heat)
    plt.colorbar(cax)

    ax2         = plt.subplot(2, 2, 3)
    cax         = ax2.matshow(np.rot90(nlog_p_val_mat[:,:,60]), vmax=0, vmin=4.5, cmap=plt.cm.gist_heat)
    plt.colorbar(cax)

    plt.suptitle('-log10(corrected p-value)')

    plt.show()


    print('done')