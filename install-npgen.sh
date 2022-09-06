# Install radish
cd radish
pip install .
python setup.py build_ext --inplace

# Install mizzle
cd ../mizzle
pip install .
python setup.py build_ext --inplace

# Install bones
cd ../bones
pip install .
python setup.py build_ext --inplace
