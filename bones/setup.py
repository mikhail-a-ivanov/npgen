from setuptools import setup

setup(name='bones',
      version='0.1',
      description='Prepare molecular simulations with the BONES force field',
      author='Erik G. Brandt',
      author_email='erik.brandt@mmk.su.se',
      license='MIT',
      packages=['bones'],
      install_requires=['argcomplete', 'numpy>=1.10', 'pandas>=0.18',
                        'mdtraj>=1.7', 'networkx>=2.0',
                        'mendeleev>=0.2.10', 'tqdm>=4.8', 'IPython>=4'],
      scripts=['bin/bonify', 'bin/flesh-out'],
      include_package_data=True,
      zip_safe=False)
