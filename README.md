# A test for an application
Copy from  https://github.com/tkipf/pygcn


# Answers
---
## Q1.2

According to Lipschitz continuous gradient, we have
$$
\begin{array}{l}
\mathbb{E}\left[f\left(\mathbf{x}_{t+1}\right)\right] \leq \mathbb{E}\left[f\left(\mathbf{x}_{t}\right)\right]+\mathbb{E}\left[\left\langle\nabla f\left(\mathbf{x}_{t}\right), \mathbf{x}_{t+1}-\mathbf{x}_{t}\right\rangle\right]+\frac{L}{2} \mathbb{E}\left[\left\|\mathbf{x}_{t+1}-\mathbf{x}_{t}\right\|^{2}\right] \\
=\mathbb{E}\left[f\left(\mathbf{x}_{t}\right)\right]-\eta \mathbb{E}\left[\left\langle\nabla f\left(\mathbf{x}_{t}\right), \mathbf{v}_{t}\right\rangle\right]+\frac{\eta^{2} L}{2} \mathbb{E}\left[\left\|\mathbf{v}_{t}\right\|^{2}\right] \\
\end{array}
$$
Summing up over $t$ from 0 to $T − 1$, we have:
$$
\begin{equation}
\mathbb{E}\left[f\left(\mathbf{x}_{{T}}\right)\right] \leq  f(x_0) - \eta \sum_{t=0}^{T}  \mathbb{E}\left[\left\langle\nabla f\left(\mathbf{x}_{t}\right), \mathbf{v}_{t}\right\rangle\right] + 
\frac{\eta^{2} L}{2} \sum_{t=0}^{T}  \mathbb{E}\left[\left\|\mathbf{v}_{t}\right\|^{2}\right]
\end{equation}
$$

Eq. (12) can be expaned as:
$$
\begin{split}
  \sum_{t=0}^{T}  \mathbb{E}\left[\left\|\mathbf{v}_{t}\right\|^{2}\right] + 
  2 \sum_{t=0}^{T}  \mathbb{E}\left[\left\langle \nabla f\left(\mathbf{x}_{t}\right), \mathbf{v}_{t}\right\rangle\right] + 
  \sum_{t=0}^{T}  \mathbb{E}\left[\left\| \nabla f\left(\mathbf{x}_{t}\right) \right\|^{2}\right] \\
  \le 
  \eta^2 L^2 \sum_{t=0}^{T}  \mathbb{E}\left[\left\|\mathbf{v}_{t}\right\|^{2}\right].
\end{split}
$$

In addition, we have the following inequality according to Cauchy–Schwarz inequality:
$$
2 \sum_{t=0}^{T}  \mathbb{E}\left[\left\langle \nabla f\left(\mathbf{x}_{t}\right), \mathbf{v}_{t}\right\rangle\right] \le \sum_{t=0}^{T}  \mathbb{E}\left[\left\|\mathbf{v}_{t}\right\|^{2}\right] + \sum_{t=0}^{T}  \mathbb{E}\left[\left\| \nabla f\left(\mathbf{x}_{t}\right) \right\|^{2}\right]
$$

To simplify the calculations, we set:
$$
\begin{array}{l}
X = \sum_{t=0}^{T}  \mathbb{E}\left[\left\langle\nabla f\left(\mathbf{x}_{t}\right), \mathbf{v}_{t}\right\rangle\right],\\
D = \sum_{t=0}^{T}  \mathbb{E}\left[ \left\| \nabla f(\mathbf{x}_{t})  \right\|^2\right],\\
V = \sum_{t=0}^{T}  \mathbb{E}\left[ \left\|  \mathbf{v}_{t} \right\|^2\right],\\
C = \mathbb{E}\left[f\left(\mathbf{x}_{{T}}\right)\right] -  f(x_0).
\end{array}
$$

Then, we have
$$
\begin{split}
  C \le -\eta X + \frac{\eta^2L}{2} V, &   \rule{5.6em}{0em}①\\
  V + D - 2 X \le \eta^2 L^2 V, &\rule{5.6em}{0em}②\\
  2 X \le V + D.  &\rule{5.6em}{0em}③\\
\end{split}
$$
We are going to find a shuitable $\eta$ to 
$$
 \frac{1}{T} D \le O(\frac{1}{T}).
$$

To solve it, substitute equations ①,③ into ② and eliminate $V$ and $X$. We therefore use $a$,$b$ to scale equations ①,③ and substitute them into ②. $V$ can be eliminated by solving the equation.
$$
a+b=2,\\
a \frac{\text{$\eta $L}}{2}+b \frac{1}{2}=1-\eta. ^2 L^2
$$
Then, we get the solutions：
$$
a\to -\frac{2 \eta ^2 L^2}{\text{$\eta $L}-1},b\to \frac{2 \left(\text{$\eta $L}+\eta ^2 L^2-1\right)}{\text{$\eta $L}-1}
$$
Finally, we have:
$$
D \le \frac{-2 C}{\eta }
$$

Therefore, it turns to be
$$
\frac{1}{T} \sum_{t=0}^{T}  \mathbb{E}\left[ \left\| \nabla f(\mathbf{x}_{t})  \right\|^2\right] \le 2\frac{f(x_0) - f(x_T)}{T \eta} \le  O(\frac{1}{T}),
$$
for any constant $\eta$.


# Q2

The code and result are in the [notebook](https://github.com/erow/pygcn-mini-batch/blob/master/pygcn/exp.ipynb).

# Q3

It is implemented in [code](question3.py).
The macro f1-score for the demo is 0.48.
The micro f1-score for the demo is 0.46513720197930725.
