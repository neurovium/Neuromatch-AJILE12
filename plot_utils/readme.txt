Utility functions to process and plot NWB data, including ECoG, pose, movement events
and behavior labels. (Modified from CatalystNeuro's original implementation by Steve Peterson)
Use when streaming data directly from DANDI instead of running on local NWB files.

Currently uses fsspec for streaming due to its compatibility with Google Colab.
If you wish to use ros3 driver for streaming, you have to build hdf5 with ros3 support:

In notebook, try this:
!pip install numpy==1.23.0
!wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz
!tar -xzf hdf5-1.12.1.tar.gz
!cd hdf5-1.12.1 && ./configure --prefix=/usr/local --enable-shared --enable-ros3-vfd
!cd hdf5-1.12.1 && make -j 2 install
!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

This will build hdf5 library for colab instance with ros3 support

If you run:
!h5cc -show

you should get:
gcc -L/usr/local/lib /usr/local/lib/libhdf5_hl.a /usr/local/lib/libhdf5.a -lcrypto -lcurl -lz -ldl -lm -Wl,-rpath -Wl,/usr/local/lib


Then you need to build h5py locally using the version of hdf5lib that supports ros3:
!pip uninstall h5py
!apt-get install libhdf5-dev
!pip install cython
!git clone https://github.com/h5py/h5py.git
%cd h5py
!python3 setup.py configure --hdf5=/usr/local/
!python3 setup.py build
!python3 setup.py install
%cd ..


Then run:
import h5py; print(h5py.__file__)
print(h5py.version.hdf5_version)
print(f'Registered drivers: {h5py.registered_drivers()}')

if you get:
Registered drivers: frozenset({ 'ros3', 'sec2', 'fileobj', 'core', 'family', 'split', 'stdio', 'mpio'})

then you can use ros3 for streaming :)

