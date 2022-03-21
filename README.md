# NPLM
Welcome to NPLM (Noisy Partial Label Model), a programmatic weak supervision system that supports (partial) labeling functions with supervision granuarity ranging from class to a set of classes.

Reference paper: [Learning from Multiple Noisy Partial Labelers](https://arxiv.org/pdf/2106.04530.pdf).

The experiments included in the paper can be found [Here](https://github.com/BatsResearch/yu-aistats22-code).

![alt text](assets/example.png)

## Introduction
Programmatic weak supervision (PWS) creates models without hand-labeled training
data by combining the outputs of noisy, user-written rules and other heuristic
labelers. Labelers are typically representated programmatically to output certain candidate tasks. 
NPLM enables users to create partial labelers that output subsets of possible class labels would
greatly expand the expressivity of programmatic weaksupervision.


## Example Usage

We will update further tutorials and documentations shortly for real-world applications

## Checklist

- [x] Label Model + PLF Abstraction Interface
- [ ] Documentation
- [ ] Tutorials

<!-- ## Contact

Please feel free to reach out to the author at <first_name>_<last_name>@brown.edu regarding any questions! -->

## Citation

Please cite the following paper if you are using our tool. Thank you!

[Peilin Yu](https://www.yupeilin.com), [Tiffany Ding](https://tiffanyding.github.io/), [Stephen H. Bach](http://cs.brown.edu/people/sbach/). "Learning from Multiple Noisy Partial Labelers". Artificial Intelligence and Statistics (AISTATS), 2022.

```
@inproceedings{yu2022nplm,
  title = {Learning from Multiple Noisy Partial Labelers}, 
  author = {Yu, Peilin and Ding, Tiffany and Bach, Stephen H.}, 
  booktitle = {Artificial Intelligence and Statistics (AISTATS)}, 
  year = 2022, 
}
```

