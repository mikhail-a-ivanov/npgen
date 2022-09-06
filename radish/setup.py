from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_module = Extension(
    "ctopo",
    ["radish/ctopo.pyx"],
    extra_compile_args = ["-O3", "-ffast-math", "-fopenmp" ],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()],
)

setup(name='radish',
      version='0.1',
      description='Extract molecular topologies from trajectories',
      author='Erik G. Brandt',
      author_email='erik.brandt@mmk.su.se',
      license='MIT',
      packages=['radish'],
      install_requires=['numpy','pandas','mdtraj','mendeleev','tqdm'],
      include_package_data=True,
      cmdclass = {'build_ext': build_ext},
      ext_modules = [ext_module],
      zip_safe=False)
