from setuptools import setup, find_packages

setup(
    name = 'flaxOptimizersBenchmark',
    version = '0.1',
    license = 'apache-2.0',
    description = 'My optimizer implementations for Flax.',
    author = 'NestorDemeure',
    # author_email = 'your.email@domain.com',
    url = 'https://github.com/nestordemeure/flaxOptimizersBenchmark',
    # download_url = 'https://github.com/nestordemeure/flaxOptimizersBenchmark/archive/v?.?.tar.gz',
    keywords = ['deep-learning', 'optimizer', 'benchmark', 'flax'],
    install_requires=['jax', 'flax', 'batchup'],
    classifiers=[ # https://pypi.org/classifiers/
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    packages=find_packages(),
)
