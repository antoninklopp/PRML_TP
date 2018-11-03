from setuptools import setup, find_packages

setup(name="PRML", 
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    packages=find_packages(), 
    author=["Yoan Souty", "Florent Geslin", "Antonin Klopp-Tosser"])

    