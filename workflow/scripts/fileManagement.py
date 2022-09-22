import os

def mkdir(patient_list, output_dir):
    folder = output_dir + '/HippiEEGAtlas'
    subdirs = ['anat', 'warps']
    if not os.path.exists(folder):
        os.mkdir(folder)
    for patient in patient_list:
        patients_dir = folder + f'/{patient}'
        if not os.path.exists(patients_dir):
            os.mkdir(patients_dir)
        for dir in subdirs:
            path_subdir = patients_dir + f'/{dir}'
            if not os.path.exists(path_subdir):
                os.mkdir(path_subdir)

# /home/ROBARTS/mcespedes/graham/scratch/Test/warps/