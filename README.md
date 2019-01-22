# Software Simulation of R-STDP Experiment on BrainScaleS-2
This is a NEST-based software simulation of a reinforcement learning experiment on the BrainScaleS-2 prototype. It was used to compare the speed and energy efficiency of the neuromorphic emulation to a software simulation on a digital processor.
It includes a simulation of the Pong environment ([pong.py](pong.py)) and the spiking neural network ([pang.py](pang.py)). All parameters are set to be equivalent to those used in the experiment on the BrainScaleS-2 prototype.

### Dependencies
- NEST with Python support
- Numpy
- Matplotlib
- Sphinx (only for documentation)

### Quick Start
Run 10000 iterations, saving every 10th iteration to folder _exp_ while printing debug messages:
```sh
python pang.py --runs 10000 --save-every 10 --folder exp --debug
```
Run this command multiple times to acquire more data. Plot the resulting experiment data (mean reward vs. iterations):
```sh
python plot_mean_reward.py --folder exp
```
The figure will be saved as _mean_reward.pdf_ in the _exp_ folder.

### Documentation
Run the following command to generate HTML documentation in folder _docs_:
```sh
make html
```

### Customization
Adapt the module-level constants in [pang.py](pang.py) to customize the simulation. See the documentation for details.
