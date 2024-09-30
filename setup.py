from setuptools import setup

setup(name='pyInfinityFlow',
    python_requires=">=3.8",
    version='1.0.0',
    description='Impute Flow Cytometry values between overlapping panels with '\
    'XGBoost regression.',
    long_description="""""",
    long_description_content_type='text/markdown',
    url='https://github.com/KyleFerchen/pyInfinityFlow',
    author='Kyle Ferchen',
    author_email='ferchenkyle@gmail.com',
    packages=['pyInfinityFlow'],
    zip_safe=False,
    include_package_data=True)