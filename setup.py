from setuptools import setup, find_packages

setup(
    name='tools_ml',
    version='0.1',
    packages=find_packages(),
    install_requires=['statsmodels == 0.14.2',
                    'pandas == 2.1.2',
                    'numpy == 1.26.1',
                    'seaborn == 0.13.2',
                    'matplotlib  ==3.8.1',
                    'scikit-learn==1.3.2',
                    'yellowbrick == 1.5',
                    'scipy == 1.11.3',
                    'plotly == 5.24.0'],
)
