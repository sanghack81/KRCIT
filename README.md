# A Kernel Relational Conditional Independence Test


### Installation

If you are using macOS, `wget` is required, which can be installed using homebrew.

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install wget
```


Other than standard python packages (e.g., numpy, scipy, matplotlib, etc), there are dependent repositories to be installed -- `pyGK` for graph kernels, `pyRCDs` for the implementation of relational causal model (RCM), and `SDCIT` for a suite of kernel-based (un)conditional independence tests.
 
Furthermore, [`tensorflow`](https://www.tensorflow.org) and [`GPflow`](https://github.com/GPflow/GPflow) are required.
An easy way to install them is using [anaconda](https://www.continuum.io/downloads) and a separate environment.

```
cd ~/anaconda/bin
./conda create --name your_env_name python=3.6 --yes
source activate your_env_name
conda install six numpy wheel scipy matplotlib pandas  --yes 
conda install -c conda-forge tensorflow --yes
cd ~/Downloads
git clone https://github.com/GPflow/GPflow
git clone https://github.com/sanghack81/pyGK
git clone https://github.com/sanghack81/SDCIT
git clone https://github.com/sanghack81/pyRCDs
git clone https://github.com/sanghack81/KRCIT
cd GPflow
python3 setup.py install
cd ../SDCIT
conda install --yes --file requirements.txt
./setup.sh
python3 setup.py install
cd ../pyGK
conda install --yes --file requirements.txt
python3 setup.py install
cd ../pyRCDs
conda install --yes --file requirements.txt
python3 setup.py install
cd ..
rm -rf SDCIT pyGK pyRCDs GPflow KRCIT
```

Test whether all required packages are installed correctly.

```python
import tensorflow
import GPflow
import pygk
from sdcit.sdcit import SDCIT
import pyrcds
```

