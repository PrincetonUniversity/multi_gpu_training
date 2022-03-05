# Multi-GPU Training with PyTorch and TensorFlow

## About

This workshop provides demostrations of multi-GPU training for PyTorch Distributed Data Parallel (DDP) and PyTorch Lightning. Multi-GPU training in TensorFlow is demonstrated using `MirroredStrategy`.

## Setup

Make sure you can run Python on Adroit:

```bash
$ ssh <YourNetID>@adroit.princeton.edu  # VPN required if off-campus
$ git clone https://github.com/PrincetonUniversity/multi_gpu_training.git
$ cd multi_gpu_training
$ module load anaconda3/2021.11
(base) $ python --version
Python 3.9.7
```

<!--
## Attendance

- Please check-in using [this link](https://cglink.me/2gi/c1471627125105938).

## Workshop Survey

Toward the end of the workshop please complete [this survey](https://forms.gle/pGi2tkzb7WCtVMcQ6).
-->

## Reminders

- The live workshop will be recorded
- Zoom: [https://princeton.zoom.us/my/picscieworkshop](https://princeton.zoom.us/my/picscieworkshop)
- Request an account on [Adroit](https://forms.rc.princeton.edu/registration/?q=adroit) if needed
- To use GPUs during the workshop: `#SBATCH --reservation=multigpu`

## Getting Help

If you encounter any difficulties with the material in this guide then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.

## Authorship

This guide was created by Jonathan Halverson and members of PICSciE and Research Computing.
