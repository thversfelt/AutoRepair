import setuptools

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='autorepair',
    version='0.1.0',
    install_requires=requirements
)
