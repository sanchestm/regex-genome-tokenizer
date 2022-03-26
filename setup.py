import setuptools

setuptools.setup(
    name='regexTokenizer',
    version='0.0.1',
    author='Thiago Sanches',
    author_email='none',
    description='Testing installation of Package',
    url='https://github.com/sanchestm/regex-genome-tokenizer',
    license='MIT',
    py_modules=['regexTokenizer'],
    long_description_content_type = 'Package to tokenize genomic data and convert fasta/q files to pandas dataframe',
    #packages=[''],
    install_requires=['numpy','scipy','ipywidgets','scikit-learn','seaborn','matplotlib','pandas','statsmodels'],
)
