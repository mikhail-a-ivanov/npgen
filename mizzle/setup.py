from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules = Extension(
    "*",
    ["mizzle/*.pyx"],
    extra_compile_args = ["-O3", "-ffast-math", "-fopenmp" ],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()],
)

setup(name='mizzle',
      version='0.2',
      description='Hydrates arbitrary metal-oxide surfaces',
      author='Samuel Stenberg',
      author_email='samuel.stenberg@hotmail.com',
      license='MIT',
      packages=['mizzle'],
      install_requires=['argcomplete','numpy','pandas','mdtraj','tqdm', 'scipy', 'radish', 'sphinx_rtd_theme'],
      scripts=['bin/mizzler'],
      include_package_data=True,
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize([ext_modules]),
      zip_safe=False)
