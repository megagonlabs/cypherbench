from setuptools import setup, find_packages


# Read the requirements from the requirements.txt file
def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


setup(
    name='cypherbench',
    version='0.1',
    description='Natural language to Cypher Benchmark',
    author='Yanlin Feng',
    author_email='yanlinf23@email.com',
    packages=find_packages(include=['cypherbench', 'cypherbench.*']),
    install_requires=[parse_requirements('requirements.txt')],
)
