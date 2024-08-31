from setuptools import setup, find_packages

setup(
    name='tevatron',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/texttron/tevatron',
    license='Apache 2.0',
    author='Luyu Gao',
    author_email='luyug@cs.cmu.edu',
    description='Tevatron: A toolkit for learning and running deep dense retrieval models.',
    python_requires='>=3.7',
    install_requires=[
        "torch==2.0.1",
        "transformers==4.30.2",
        "datasets==2.13.1",
        "accelerate==0.20.3",
        "faiss-cpu==1.7.2",
        "sentencepiece==0.1.99",
        "tokenizers==0.13.3"
    ]
)
