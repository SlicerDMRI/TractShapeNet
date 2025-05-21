import numpy
import whitematteranalysis as wma
import fibers
import argparse
import os
import h5py

def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to generate the data format used for the method from .vtp files')
    parser.add_argument(
        'indir', help='A folder of white matter tracts (one specifiv tract such as AF) from multiple subjects as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        'outdir', help='Output folder for the generated data.')
    args = parser.parse_args()

    data_dir = args.indir
    out_dir=args.outdir
    tmp = "\nData preparation\nReading data from:\t./" + data_dir
    print(tmp)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    input_pd_fnames = wma.io.list_vtk_files(data_dir)
    num_pd = len(input_pd_fnames)
    print('num_pd', num_pd)
    tract_name = os.path.basename(data_dir)
    print('tract_name', tract_name)

    fname=os.path.join(out_dir,'feats_{}.h5'.format(tract_name))
    f = h5py.File(fname, 'w')
    print('save output file as:', fname)
    for i in range(num_pd):
        inputFile = input_pd_fnames[i]
        sub_id=(os.path.basename(inputFile)).split('.')[0]
        sub_id = sub_id.replace("cluster_", "")
        input_data = wma.io.read_polydata(inputFile)
        Nosc=input_data.GetNumberOfLines()
        fiber_array = fibers.FiberArray()
        fiber_array.convert_from_polydata(input_data)
        Nosc_array=list(numpy.repeat(Nosc,len(fiber_array.FA1)))
        feat = numpy.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s)).squeeze()
        f[str(sub_id)] = feat
    f.close()
