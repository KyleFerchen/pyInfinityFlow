from setuptools import setup

setup(name='pyInfinityFlow',
    python_requires=">=3.8",
    version='0.1',
    description='Impute Flow Cytometry values between overlapping panels with '\
    'XGBoost regression.',
    url='http://github.com/',
    author='Kyle Ferchen',
    author_email='ferchenkyle@gmail.com',
    license='MIT',
    install_requires=[
        'umap-learn >= 0.5',
        'xgboost >= 1.6',
        'scanpy >= 1.9',
        'pyarrow >= 9.0',
        'leidenalg',
    ],
    packages=['pyInfinityFlow'],
    entry_points = {
        'console_scripts': ['pyInfinityFlow=pyInfinityFlow.pyInfinityFlow:main',
            'pyInfinityFlow-list_channels=pyInfinityFlow.pyInfinityFlow_list_channels:main'],
    },
    zip_safe=False,
    include_package_data=True)