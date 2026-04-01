from setuptools import setup, find_packages

setup(
    name="revenue-intelligence-platform",
    version="1.0.0",
    description="Production-grade Revenue Intelligence Platform for SaaS companies",
    author="Revenue Intelligence Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "rip-run=run:main",
        ]
    },
)
