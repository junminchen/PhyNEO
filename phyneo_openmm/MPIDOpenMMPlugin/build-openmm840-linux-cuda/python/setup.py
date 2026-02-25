from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform
import numpy

openmm_dir = '/home/am3-peichenzhong-group/miniconda3/envs/mpid'
mpidplugin_header_dir = '/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/openmmapi/include/openmm'
mpidplugin_library_dir = '/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_mpidplugin',
                      sources=['MPIDPluginWrapper.cpp'],
                      libraries=['OpenMM', 'MPIDPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), mpidplugin_header_dir, numpy.get_include()],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), mpidplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='mpidplugin',
      version='1.0',
      py_modules=['mpidplugin'],
      ext_modules=[extension],
     )
