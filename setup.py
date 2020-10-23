import setuptools
from dendritic_arborization_tracer.minimal_DAT_GUI import __MAJOR__, __MINOR__, __MICRO__, __AUTHOR__, __VERSION__, __NAME__, __EMAIL__

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='dendritic-arborization-tracer',
    version=__VERSION__,
    author=__AUTHOR__,
    author_email=__EMAIL__,
    description='A tool to segment dendrites',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/baigouy/DAT',
    package_data={'': ['*.md']}, # include all .md files
    license='BSD',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    # TODO put this in requirements.txt file and read the file from here --> can have both methods work
    # below are the required files (they will be installed together with dendritic-arborization-tracer unless the '--no-deps' tag is used)
    install_requires=[
    	"opencv-python",
        "czifile",
        "matplotlib",
        "numpy",
        "numpydoc",
        "Pillow",
        "PyQt5",
        "PyQtWebEngine",
        "read-lif",
        "scikit-image",
        "scipy",
        "tifffile",
        "natsort",
        "numexpr"
    ],
    python_requires='>=3.6' # tensorflow is now supported by python 3.8
)