cd $HOME/Desktop
git clone https://github.com/vcg-uvic/csc205
cd csc205/labs/lab6

pip3 install virtualenv --user
$HOME/Library/Python/3.7/bin/virtualenv venv -p $(which python3)
source venv/bin/activate
pip3 install jupyter matplotlib opencv-python
jupyter notebook --ip 0.0.0.0
