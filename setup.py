from setuptools import setup, find_packages

setup(
    name='yourpackage',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit==0.89.0',
        'pandas==1.3.3',
        'scikit-learn==0.24.2',
        'imbalanced-learn==0.8.1'

    ],
)
