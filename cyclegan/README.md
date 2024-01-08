## Train
+ Hyper parameters can be adjusted in class `Configs`;
+ If you are the first time to train, then call `train1` instead of `train` in the `main` program; if you want to train based on the previous checkpoint, then correct `trained_run_uuid` in function `train` and then run the file.
+ The train set would be split into `trainA` and `trainB`, which could be acquired in JBOX. Then put them at `./data/cycle_gan/real2mural/trainA` and `./data/cycle_gan/real2mural/trainB` respectively. 
## Test
+ Create two folders `.data/cycle_gan/my_test/testA` and `./data/cycle_gan/my_test/testB`,
then put your image in the two folders;

+ Run `eval.py`, and the result will be saved in `.data/cycle_gan/my_test/Final<your image name>.jpg` 