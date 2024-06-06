from setuptools import setup

package_name = 'p3at_deepvio'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Cedric Melancon',
    author_email='cedric.melancon.1@ens.etsmtl.ca',
    maintainer='Cedric Melancon',
    maintainer_email='cedric.melancon.1@ens.etsmtl.ca',
    description='Examples of minimal publishers using rclpy.',
    license='TODO: Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'p3at_deepvio = p3at_deepvio.p3at_deepvio:main',
        ],
    },
)
