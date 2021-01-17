[![Build Status](https://travis-ci.com/maksymshylo/statistical_pattern_recognition.svg?token=j9Kqn8jNSznud7EAtsqm&branch=main)](https://travis-ci.com/maksymshylo/statistical_pattern_recognition)

# Statistical Pattern Recognition
Labs for University Course     


[tasks](https://github.com/maksymshylo/statistical_pattern_recognition/blob/main/tasks.pdf "tasks"),  [solutions](https://github.com/maksymshylo/statistical_pattern_recognition/blob/main/solutions.pdf "solutions")

## Lab 1 - DP algorithm for chain-structured graphical models 
### Image Denoising (Bernoulli noise)
#### Examples
```bash
python3 lab1/main.py lab1/frequencies.json lab1/alphabet path_to_input_image noise_level

python3 lab1/main.py lab1/frequencies.json lab1/alphabet 'lab1/test_images/hello sweety_0.3.png' 0.3
python3 lab1/main.py lab1/frequencies.json lab1/alphabet 'lab1/test_images/but thence i learn and find the lesson true drugs poison him that so feil sick of you_0.45.png' 0.45
```
## Lab 2 - Min-Sum Diffusion
### Image Segmentation
#### Examples
```bash
python3 lab2/main.py input_image alpha n_iter colors

python3 lab2/main.py lab2/test_images/ipt.png 1 10 blue white yellow
python3 lab2/main.py lab2/test_images/map_hsv.png 3 100 blue lime
```
## Lab 3 - Tree Reweighted Message Passing (TRW-S)
### Image Inpainting
#### Examples
```bash
python3 lab3/main.py input_image alpha Epsilon n_labels n_iter

python3 lab3/main.py lab3/test_images/mona-lisa-damaged.png 1 0 18 1
```
## Lab 4 - "GrabCut"
**_NOTE:_**  TRW-S as an energy minimization algorithm (instead of Min-Cut/Max-Flow algorithm)
### Interactive Foreground Extraction
#### Examples
```bash
python3 lab4/main.py image_path mask_path gamma n_bg n_fg color_bg color_fg em_n_iter trws_n_iter n_iter 

python3 lab4/main.py lab4/test_images/alpaca.jpg lab4/test_images/alpaca-segmentation.png  50 3 3 blue red 10 10 1
python3 lab4/main.py lab4/test_images/lotus.jpg lab4/test_images/lotus-segmentation.png  50 3 3 lime blue 10 10 1
```
