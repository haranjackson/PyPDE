import os
import platform
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', **kwa):
        Extension.__init__(self, name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    def build_extensions(self):

        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        for ext in self.extensions:

            extdir = os.path.abspath(
                os.path.dirname(self.get_ext_fullpath(ext.name)))

            cfg = 'Release'

            cmake_args = [
                # Place result in the directory containing the extension
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), extdir + '/' + ext.name + '/build'),

                # Place intermediate static libraries in temporary directory
                '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), self.build_temp)
            ]

            # We can handle some platform-specific settings at our discretion
            if platform.system() == 'Windows':
                plat = ('x64'
                        if platform.architecture()[0] == '64bit' else 'Win32')
                cmake_args += [
                    # These options are likely to be needed under Windows
                    '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                    '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(
                        cfg.upper(), extdir)
                ]
                # Assuming that Visual Studio and MinGW are supported compilers
                if self.compiler.compiler_type == 'msvc':
                    cmake_args += ['-DCMAKE_GENERATOR_PLATFORM=%s' % plat]
                else:
                    cmake_args += ['-G', 'MinGW Makefiles']

            else:
                cmake_args.append('-DCMAKE_BUILD_TYPE={}'.format(cfg))

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config
            subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                                  cwd=self.build_temp)

            # Build
            subprocess.check_call(['cmake', '--build', '.'],
                                  cwd=self.build_temp)


setup(
    name='PyPDE',
    version='0.9.1',
    author='Haran Jackson',
    author_email='jackson.haran@gmail.com',
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    cmdclass={'build_ext': cmake_build_ext},
    description='Solve any hyperbolic/parabolic system of PDEs',
    ext_modules=[CMakeExtension('pypde')],
    install_requires=["numba>=0.46", "numpy>=1.14"],
    keywords=[
        'ADER', 'WENO', 'Discontinuous Galerkin', 'Finite Volume', 'PDEs',
        'Partial Differential Equations', 'Hyperbolic', 'Parabolic'
    ],
    long_description=open("README.rst").read(),
    packages=find_packages(),
    python_requires=">=3.6",
    setup_requires=["cmake>=3.5"],
    tests_require=["matplotlib>=2.0"],
    url="https://github.com/haranjackson/pypde",
    zip_safe=False)
