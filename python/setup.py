from setuptools import setup

with open('requirements.txt') as requirements_file:
    requirements = list(requirements_file)

setup(
    name='greg-utils',
    description="Greg's useful stuff",
    author='Gregory Werbin',
    author_email='outthere@me.gregwerbin.com',
    packages=find_packages(exclude='tests'),
    install_requires=requirements,
    python_requires='>=3.6'
)
