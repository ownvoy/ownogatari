---
title: "Ddpg"
date: 2023-09-23T00:53:01+09:00
draft: false
---

## 0. BackGround

observation: \\(x_t\\)

state: \\(s_t\\)

state:  \\(a_t\\)

reward:  \\(r_t\\)

policy: \\(\pi, \ \ S \to P(A)\\) 

&nbsp;

transition dynamics : \\(p(s_{t+1} \mid s_t,a_t)\\)

discounted future reward : \\(R_T = \sum_{i=t}^{T}\gamma^{(i-t)}r(s_i,a_i)\\)

objective function: \\(E_{r_i,s_i \sim E, a_i \sim \pi}[R_1]\\)

&nbsp;

\\(Q^{\pi}(s_t,a_t) = E_{r_i\geq t, s_i >t \sim E , a_i > t \sim \pi}[R_t \mid s_t, a_t]\\)

\\(Q^{\pi}(s_t,a_t) = E_{r_t,s_{t+1} \sim E} [r(s_t,a_t)+ \gamma E_{a_{t+1} \sim \pi} [Q^{\pi}(s_{t+1}, a_{t+1})]]\\)

\\(Q^{\mu}(s_t,a_t) = E_{r_t,s_{t+1} \sim E} [r(s_t,a_t)+ \gamma Q^{\mu}(s_{t+1}, \mu(s_{t+1}))]\\)


## Q-learning

\\(Q(S_t,a_t) \leftarrow Q(S_t,a_t)\\)


\\(Q(s_t,a_t)\\)

\\(\mu (s_t)\\)

\\(Q(s_t,a_t \mid \theta^Q)\\)

\\(\mu(s_t\mid\theta^{\mu})\\)

\\(\nabla_{\theta^{\mu}}J(\theta) = E_{s_t \sim \rho^{\beta}}[\nabla_{\theta^{\mu}}Q(s,a \mid \theta^{Q}) \mid_{s=s_t,a=\mu(s_t\mid\theta^{\mu})}]\\)

\\(\mu(s\mid\theta^{\mu}) = a_{t+1}\\)

\\(\nabla_{\theta^{\mu}} = \frac{d}{d\theta^{\mu}}  \Rightarrow \frac{d}{da}\times\frac{da}{d\theta^{\mu}}\\)

\\(\nabla_a \times\nabla_{\theta^{\mu}}\times \mu(s_t\mid\theta^{\mu})  \\)

\\(\nabla_{\theta^{\mu}}J(\theta) = E_{s_t \sim \rho^{\beta}}[\nabla_{a}Q(s,a \mid \theta^{Q}) \mid_{s=s_t,a=\mu(s_t)}\nabla_{\theta^{\mu}}\mu(s_t\mid\theta^{\mu}) \mid_{s=s_t} ] \\) 


\\((s_t,a_t,r_t,s_{t+1})\\)

\\(\theta_{targ} \leftarrow \tau\theta + (1-\tau)\theta_{targ}\\)

\\(\tau\\) is close to zero.
