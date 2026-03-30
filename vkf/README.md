# vkf-string-sep
String Separation via Vold-Kalman Filter (VKF) prototype using groundtruth string information (F0 and String Assignment Information).

## Getting started

The guitar mixture is processed in two bands (low band for lower 3 strings and high band for higher 3 strings) to ensure better frequency resolution for the lower band. 

Before filtering the audio is flipped as this showed to mitigate filtering artifacts introduced by the VKF due to transient plucking excitation signals (Controllable with flag ``FLIP_AUDIO``)

There is also a block-based implementation of the VKF filter to dec`rease computational demand. (Controllable with flag ``BLOCK_PROCESS``). This feature is although not thoroughly tested.

The bandwidths of the filters at each harmonic are designed relatively to their respective harmonic frequency (Controllable with flag ``F0_REL_BW`` and bandwidth parameters ``BW_PERCENT``).


## Setting up a conda environment
### Create a new conda env with
``conda create -n vkf_env_new python=3.11.11``
### Install pip requirements with
``pip install -r ./requirements.txt``

