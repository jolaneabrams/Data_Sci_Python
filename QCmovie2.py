import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def QCmovie2(list_nii):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        plt.show()

        qc_notes = []

        for i in range(len(list_nii)):
            rerun_current_image = False

            while True:
                nii_filename_i = list_nii[i]
                print('{}/{} loading {}'.format(i + 1, len(list_nii), nii_filename_i))
                img_i = nib.load(nii_filename_i)
                data_i = img_i.get_fdata()

                size_dim_0 = data_i.shape[0]
                startNum = np.round(size_dim_0 * 0.10).astype(int)
                endNum = np.round(size_dim_0 * 0.90).astype(int)
                iterStep = 5

                for j in range(startNum, endNum, iterStep):
                    ax.clear()
                    ax.matshow(np.rot90(data_i[j, :, :]), cmap=plt.cm.gray)
                    plt.draw()
                    plt.pause(0.35)

                if rerun_current_image:
                    rerun_current_image = False
                    continue

                user_choice = input("Return (r) or input QC note (i)? ").lower()
                if user_choice == 'r':
                    rerun_current_image = True
                    continue
                elif user_choice == 'i':
                    qc_note_i = input('Enter QC note: ')
                    qc_notes.append(qc_note_i)
                    break
                else:
                    print("Invalid choice. Assuming rerun.")
                    rerun_current_image = True
                    continue

        # close figure
        plt.close(fig)
        return qc_notes


