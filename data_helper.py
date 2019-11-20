import h5py

def get_numpy_dataset(file_name):
    f_data = h5py.File(file_name, 'r')
    key_data = list(f_data.keys())
    output_x = []
    output_y = []

    for key in key_data:
        ## pick first person
        = f_train[key][0].