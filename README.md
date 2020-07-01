1) finger_test.py shows the finger environment with some random motions
2) finger_mppi_main.py runs the finger environment and trains model.
3) pytorch_mppy/tests/pendulum_approximate_continouous.py contains the original code for pendulum task
4) pytorch_mppy/pytorch_mppi/mppi_finger.py contains the actual control strategy.
5) custom_envs/ contains the finger environment
all other files are different finger models I tried.

you probably need to change the file path in requirements.txt for some of the packages. 
