
Trace Norm Regression
==========================

Implementation of Ji, S. & Ye, J., _An accelerated gradient method for trace norm minimization_, Proceedings of the 26th Annual International Conference on Machine Learning, 2009, 457-464
and comparison to Ridge regression.

This regression method is useful for multi-task learning of sparse representations.

Given two matrices 'U' and 'V', the function to minimize is:
```mathjax
![| UA - V |^2_2 + |A|](http://www.sciweavers.org/tex2img.php?eq=%5C%7C%20UA%20-%20B%5C%7C%5E2_2%20%2B%20%7CA%7C%5E%2A&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

where the right term is the sum of the singular values of 'A'.

