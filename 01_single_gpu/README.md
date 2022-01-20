# Single-GPU Training

## Installation and Single-GPU Training Example

See the installation directions for [PyTorch](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch) and then work through the MNIST example on that page. Below we will extended the MNIST example to use multiple GPUs.

## Optimizing the Single GPU Case

Make sure you optimize the single GPU case before going to multiple GPUs by working through the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html). You should also profile your code using [line_profiler](https://researchcomputing.princeton.edu/python-profiling) or another tools like dlprof.
