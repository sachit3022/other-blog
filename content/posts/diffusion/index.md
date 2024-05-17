+++
title = 'Generative models: VAE, Score, and Diffusion'
date = 2024-04-29T17:37:27-04:00
draft = false
math = true
pygmentsUseClasses=true
pygmentsCodeFences=true
tableOfContents = true
+++

Diffusion models have been extensively studied, and there is a lot of literature available to learn from [\[1](#1) - [4\]](#4). In this blog we will try to explain intutively with the example of 8-mode gaussian generation.
![Diffusion Evolution](density_evo.gif)

### Generative modelling
We can think of generative modeling as learning a transformation function that goes from a known distribution to the unknown. For example, suppose we want to sample from a Gaussian distribution. We first sample from a uniform distribution and then apply the inverse CDF of the Gaussian to get the new random variable whose distribution is Gaussian. We can express this as follows: for a random variable $z$ with a known distribution (uniform in our example), we learn a function $f$ (the inverse CDF of the distribution we want to sample from). Then, we sample $x$ such that $x = f(z)$, where $z \sim U[0,1]$. Since we have no knowledge of the inverse CDF of an unknown sampling process, we approximate it with a neural network.

### Diffusion as VAE 
<img src="DDPM.png" alt="DDPM" style="width:200px; margin: 20px;" align="left" /> 
The image dipicts how VAE and diffusion are connected. In VAE we go from Image domain to the latent domain. We enforce the latent to be gaussian $\mathcal{N}(0,1)$  by constrianing latent with KL divergence. Then the decoder learns the inverse transformation of it. This notion of VAE is very similar to 












### References
<a id="1">[1]</a> Chan, S. H. (2024). Tutorial on Diffusion Models for Imaging and Vision. arXiv preprint arXiv:2403.18103. https://arxiv.org/pdf/2403.18103.pdf.

<a id="2">[2]</a>  Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

<a id="3">[3]</a>  Vishnu Boddeti. (2024). Deep Learning. https://hal.cse.msu.edu/teaching/2024-spring-deep-learning/

<a id="4">[4]</a>  Arash Vahdat. et al. (2022). CVPR. https://cvpr2022-tutorial-diffusion-models.github.io/
