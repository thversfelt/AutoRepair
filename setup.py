from setuptools import setup, find_packages

setup(
    name='autorepair',
    version='0.1.3',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pymoo==0.6.0.1',
        'autotest'
    ],
    dependency_links=[
        'git+https://github.com/thversfelt/AutoTest.git#egg=autotest'
    ]
)
