# launch the script after anaconda login

conda convert -f -p win-64 -o /Users/slgentil/wheels /Users/slgentil/wheels/osx-64/pynsitu-0.0.1-py39_0.tar.bz2
conda convert -f -p win-64 -o /Users/slgentil/wheels /Users/slgentil/wheels/osx-64/pynsitu-0.0.1-py310_0.tar.bz2
conda convert -f -p linux-64 -o /Users/slgentil/wheels /Users/slgentil/wheels/osx-64/pynsitu-0.0.1-py39_0.tar.bz2
conda convert -f -p linux-64 -o /Users/slgentil/wheels /Users/slgentil/wheels/osx-64/pynsitu-0.0.1-py310_0.tar.bz2
 
anaconda upload /Users/slgentil/wheels/osx-64/pynsitu-0.0.1-py39_0.tar.bz2
anaconda upload /Users/slgentil/wheels/osx-64/pynsitu-0.0.1-py310_0.tar.bz2
anaconda upload /Users/slgentil/wheels/win-64/pynsitu-0.0.1-py39_0.tar.bz2
anaconda upload /Users/slgentil/wheels/win-64/pynsitu-0.0.1-py310_0.tar.bz2
anaconda upload /Users/slgentil/wheels/linux-64/pynsitu-0.0.1-py39_0.tar.bz2
anaconda upload /Users/slgentil/wheels/linux-64/pynsitu-0.0.1-py310_0.tar.bz2
