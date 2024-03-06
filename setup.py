from setuptools import setup, find_packages

setup(
    name="s4wm",  # Replace with your own package name
    version="0.1.0",  # The initial release version
    author="Your Name",  # Your name
    author_email="your.email@example.com",  # Your email
    description="A short description of the package",  # A short description
    long_description=open("README.md").read(),  # A long description from README.md
    long_description_content_type="text/markdown",  # Type of the long description
    url="https://github.com/yourusername/your_package_name",  # Home page for your project
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        "numpy",  # List of packages required by your package
        "jax",
        "torch",
        # Add other dependencies as needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose the appropriate license
        "Operating System :: OS Independent",
    ],
)
