# Multi-GPU Training with PyTorch and TensorFlow

## About

This workshop shows participants how to optimize single-GPU training. The concepts of multi-GPU training are introduced before demonstrating the use of Distributed Data Parallel (DDP) in PyTorch. Other distributed deep learning frameworks are discussed. While the workshop is largely focused on PyTorch, demonstrations for TensorFlow are available.

## Setup

Make sure you can run the Python script `test.py` on Adroit:

```bash
$ ssh <YourNetID>@adroit.princeton.edu  # VPN required if off-campus
$ git clone https://github.com/PrincetonUniversity/multi_gpu_training.git
$ cd multi_gpu_training/test
$ module load anaconda3/2021.5
$ python test.py
Success
```

## Attendance

[https://cglink.me/2gi/c1471627125105938](https://cglink.me/2gi/c1471627125105938)

## Workshop Survey

Toward the end of the workshop please complete [this survey](https://forms.gle/pGi2tkzb7WCtVMcQ6).

## Reminders

- The live workshop will be recorded
- Please check-in using this link: [https://cglink.me/2gi/c1471627125105938](https://cglink.me/2gi/c1471627125105938)
- [Zoom link](https://princeton.zoom.us/my/picscieworkshop2)
- Request an account on [Adroit](https://forms.rc.princeton.edu/registration/?q=adroit)
- To use GPUs during the workshop: `#SBATCH --reservation=multigpu`

## Getting Help

If you encounter any difficulties with the material in this guide then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.

## Authorship

This guide was created by Jonathan Halverson and members of PICSciE and Research Computing.

