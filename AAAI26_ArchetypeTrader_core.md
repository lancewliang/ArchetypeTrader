# ArchetypeTrader: Reinforcement Learning for Selecting and Refining Learnable
LaTeX+markdown format
    
## ArchetypeTrader
To overcome the pitfalls of human-engineered segmentation and underutilized demonstrations, we propose ArchetypeTrader, a hierarchical framework with three training phases illustrated in Fig. 1: 
1) In phase one, we design a DP planner to generate demonstration trajectories. A VQ-based encoder–decoder then compresses these trajectories into a discrete set of archetypes capturing key trading behaviors. 
2) In phase two, a horizon-level RL agent selects the most suitable archetype whose expert insights best match current market features. 
3) In phase three, a step-level policy adapter refines the archetype’s base actions in response to real-time market observations and interim performance, ensuring robustness and agility under volatile conditions.

### Archetype Discovery
We propose a self-supervised learning pipeline that extracts compact and reusable trading archetypes directly from high-quality trading trajectories, eliminating reliance on human design or heuristics. Concretely, we first sample n data chunks of fixed length h from the training dataset and apply a DP planner (Algorithm 1) to identify profitable trading actions for each chunk. Unlike prior approaches that attempt to capitalize on every profitable fluctuation (Qin et al. 2023; Zong et al. 2024), we deliberately limit each data chunk to a single trade to emphasize the most significant and impactful opportunities within each horizon. By capturing only the primary movements, our demonstration trajectories filter out small, short-lived fluctuations, thereby reducing noise, sim-plifying the subsequent learning, and providing a clean foundation for coherent and reusable trading archetypes.

#### Algorithm 1: Single-trade DP planner
---

**Input:** price series $P$ of length $N$, action set $\mathcal{A}$  
**Output:** $\{\hat{a}_t\}_{0}^{N-1}$

--- 

1:  $V[N+1, |\mathcal{A}|, 2] \leftarrow 0, \quad \Pi[N, |\mathcal{A}|, 2] \leftarrow -1 $
2:  for $t$ = $N$-1 downto 0 do
3:      for $i \in \mathcal{A}, c \in \{0,1\}$ do
4:          $\Pi[t,i,c] = \arg\max_{j \in \mathcal{A}:\; c + \mathbf{1}[i \neq j] \leq 1} \left( r_t(i \rightarrow j) + \gamma V[t+1, j, c + \mathbf{1}[i \neq j]] \right)$
5:          $V[t,i,c] = r_t(i \rightarrow \Pi[t,i,c]) + \gamma V[t+1, \Pi[t,i,c], c + \mathbf{1}[i \neq \Pi[t,i,c]]]$
6:      end for
7:  end for
8:  $i \leftarrow 1,\quad c \leftarrow 0$
9:  for $t$ = 0 to $N$-2 do
10:     $i' \leftarrow \Pi[t,i,c]$;$\hat{a}_t \leftarrow \mathcal{A}[i']$
11:     $c \leftarrow \min\{1, c + \mathbf{1}[i \neq i']\}, \quad i \leftarrow i'$
12: end for
13: $\hat{a}_{N-1} \leftarrow \hat{a}_{N-2}$ return $\{\hat{a}_t\}_{0}^{N-1}$

---

For each sampled horizon represented by market observations $\mathbf{s}_{\mathrm{demo}} = (s^{\mathrm{demo}}_{0:h-1})$, the DP planner outputs a demonstration action sequence $\mathbf{a}_{\mathrm{demo}} = (a^{\mathrm{demo}}_{0:h-1})$, where $a^{\mathrm{demo}} \in \{0,1,2\}$ denotes short, flat, or long positions respectively. By construction, exactly one timestep $t \in [0, h-1]$ satisfies $a_t^{\mathrm{demo}} - a_{t-1}^{\mathrm{demo}} \neq 0$, guaranteeing a single trade per horizon. The resulting reward sequence is $\mathbf{r}_{\mathrm{demo}} = (r^{\mathrm{demo}}_{0:h-1})$ by executing the demonstration actions over the horizon. We collect the resulting demonstration trajectories into tuples $\tau = (\mathbf{s}_{\mathrm{demo}}, \mathbf{a}_{\mathrm{demo}}, \mathbf{r}_{\mathrm{demo}})$ and compile them into a dataset $\mathcal{D} = \{\tau_i\}_{i=0}^{n-1}$, which forms the training base for archetype extraction.

We now seek to distill the demonstration trajectories into epresentative trading archetypes. Inspired by skill-based RL approaches in robotics (Pertsch, Lee, and Lim 2021), we adopt an encoder-decoder framework: the encoder ingests each demonstration chunk and projects it to a continuous latent vector, while the decoder reconstructs the original action sequences given the state inputs and the learned representation. However, a purely continuous representation can be challenging for subsequent RL-based trading, primarily for two reasons: 
1) it expands the search space, making it harder for the RL selector in the second phase to effectively explore and select an optimal archetype from a continuous manifold; 
2) it complicates strategy clustering, resulting in fragmented archetypes and inconsistent policy behavior.
To address these limitations, we incorporate a VQ module (Van Den Oord, Vinyals et al. 2017) to learn more meaningful and generalizable archetype embeddings. By discretizing the encoder’s continuous latent outputs into a finite codebook, each demonstration chunk is abstracted to one of a limited set of discrete codes, forming a concise and highly reusable set of archetypes.

In practice, each demonstration trajectory is passed through an LSTM-based encoder,

$$
q_{\theta_e}\!\left(z_e \mid \mathbf{s}_{\mathrm{demo}}, \mathbf{a}_{\mathrm{demo}}, \mathbf{r}_{\mathrm{demo}}\right)
\tag{1}
$$

which produces a continuous embedding $z_e$. Next, this embedding is quantized by selecting the nearest entry from a learnable codebook $\epsilon = \{e_0, \ldots, e_{K-1}\}$, i.e.,

$$
k=\arg\min_{0<=i<=K-1}\left\|z_e-e_i\right\|^2,\qquad z_q=e_k
\tag{2}
$$

yielding a discrete latent representation $z_q$. Finally, a decoder

$$
p_{\theta_d}\!\left(\hat{\mathbf{a}}_{\mathrm{demo}} \mid \mathbf{s}_{\mathrm{demo}}, z_q\right)
\tag{3}
$$

reconstructs the original actions $\mathbf{a}_{\mathrm{demo}}$ from states $\mathbf{s}_{\mathrm{demo}}$ and the quantized embedding $z_q$. We train this model by minimizing the combined loss function:

$$
L = L_{\mathrm{rec}} + \left\|\operatorname{sg}[z_e]-z_q\right\|^2 + \beta_0 \left\|z_e-\operatorname{sg}[z_q]\right\|^2
\tag{4}
$$

where $L_{\mathrm{rec}}$measures the reconstruction loss of demonstration actions, the rest terms enforce commitment to the chosen code and keep the codebook entries close to the encoder output, and sg[·] denotes the stop-gradient operation. By optimizing this loss, the codebook vectors {ei} learn to effectively capture reusable trading patterns, i.e., the archetypes,that effectively summarize the demonstration trajectories. In later phases, selecting an archetype based on current market observations and decoding its latent code into a concrete action sequence allows us to deploy these archetypes as coherent, data-driven trading strategies.


### Archetype Selection
To effectively apply the learned strategic archetypes to trading, we train an RL-based archetype selector to choose the optimal archetype based on the current market observations and execute the micro-actions reconstructed by the previously trained decoder. Specifically, we lift the basic MDP defined in the previous section to a horizon-level

MDP $\mathcal{M}_{\mathrm{sel}} = \langle S_{\mathrm{sel}}, A_{\mathrm{sel}}, R_{\mathrm{sel}}, \gamma \rangle$. For a fixed horizon $H = [t, t + h - 1]$, state $s^{\mathrm{sel}}$ is defined as the state vector $s_t$ defined in the previous section, captured at the first bar of the horizon. Action $a^{\mathrm{sel}} \in \{0,1,\ldots,K-1\}$ denotes the selected archetype from previously learned archetype set $\epsilon = \{e_{0:k-1}\}$ via the selection policy $\pi^{\mathrm{sel}}_{\phi}(a^{\mathrm{sel}} \mid s^{\mathrm{sel}})$. Once an archetype is chosen, its archetype code $e_{a^{\mathrm{sel}}}$ is fed into the frozen decoder $p_{\theta_d}(a^{\mathrm{base}} \mid s, e_{a^{\mathrm{sel}}})$, which emits step-wise micro actions $(a^{\mathrm{base}}_{t:t+h-1})$ based on the upcoming states $(s_{t:t+h-1})$ for the next $h$ steps. We use the sum of step-wise reward over the whole archetype horizon $H$ as the reward function for archetype selection, which is calculated as $r_t^{\mathrm{sel}} = \sum_{\tau=t}^{t+h-1} r_{\tau}^{\mathrm{step}}$. The policy $\pi_{\phi}$ is optimized to maximize the expected return while also encouraging consistency with demonstration trajectories. Concretely, we define the following objective:
$$
J = \mathbb{E}_{\pi^{\mathrm{sel}}_{\phi}} \left[ \sum_{t=0}^{\infty} \left( \gamma^t r_t^{\mathrm{sel}} - \alpha \, KL\!\left(\hat{a}_t^{\mathrm{sel}} \,\|\, \pi^{\mathrm{sel}}_{\phi}\!\left(a_t^{\mathrm{sel}} \mid s_t^{\mathrm{sel}}\right)\right) \right) \right]
\tag{5}
$$

where $r_t^{\mathrm{sel}}$ is the cumulative return obtained over the horizon, $\hat{a}_t^{\mathrm{sel}}$ is the ground-truth archetype label assigned by the VQ encoder to the demonstration chunk of this horizon, and $\alpha$ is a hyperparameter balancing the environment reward against the KL-divergence penalty. This penalty encourages the selection policy to remain near the demonstrated archetype choices, yet still allows it to adapt as it gains experience in the live environment.

### Archetype Refinement
A horizon-level choice of archetype already delivers a coherent trading plan. However, two key limitations arise when market conditions shift or the chosen archetype is suboptimal. First, a single decision at the beginning of the horizon cannot react to rapid intra-chunk changes, potentially missing short-lived trading opportunities or failing to perform timely stop-loss actions. Second, if the archetype selection policy chooses an ill-suited archetype at the horizon’s start,the agent is mostly limited to suboptimal actions for the entire horizon. Yet, in many cases, on-the-fly performance monitoring can quickly reveal that the chosen archetype is misaligned with ongoing market dynamics. Without an effective adaptation method, the agent endures unnecessary losses or foregoes profits until the horizon ends. To handle these issues, we propose a step-level archetype refine-
ment method by employing a policy adapter, which leverages ongoing market observations and partial archetype performance to fine-tune the archetype’s base actions.

For each horizon $H$, we freeze the horizon-level selection policy $\pi^{\mathrm{sel}}_{\phi}$ and decode the corresponding micro action sequence $\mathbf{a}_{\mathrm{base}} = (a^{\mathrm{base}}_{t:t+h-1})$ by the chosen archetype. The archetype refinement process is formulated as a step-level MDP $\mathcal{M}_{\mathrm{ref}} = \{S_{\mathrm{ref}}, A_{\mathrm{ref}}, R_{\mathrm{ref}}\}$. The archetype adaptation agent observes the real-time state $s^{\mathrm{ref}}_{\tau}$ at time $\tau \in [t, t+h-1]$, which consists of two parts: step-wise market observations $s^{\mathrm{ref1}}_{\tau} = s_{\tau}$ and archetype information $s^{\mathrm{ref2}}_{\tau} = [e_{a_t^{\mathrm{sel}}}, a^{\mathrm{base}}_{\tau}, R^{\mathrm{arche}}_{\tau}, \tau_{\mathrm{remain}}]$, where $e_{a_t^{\mathrm{sel}}}$ is the selected archetype embedding, $a_{\tau}^{\mathrm{base}}$ is the current micro action constructed by the archetype, $R_{\tau}^{\mathrm{arche}} = \sum_{i=t}^{\tau} r_{i}^{\mathrm{step}}$ is the cumulative reward under the archetype’s base policy and $\tau_{\mathrm{remain}} = t+h-\tau$ is the number of steps left in the horizon. The adapter generates a refinement signal $a_{\tau}^{\mathrm{ref}} \in \{-1,0,1\}$ that shifts the archetype’s base action $a_{\tau}^{\mathrm{base}}$ while ensuring the original trades are never overridden:

$$
a_{\tau}^{\mathrm{final}}=
\begin{cases}
a_{\tau}^{\mathrm{base}}, & \text{if } a_{\tau}^{\mathrm{base}} \neq a_{\tau-1}^{\mathrm{base}} \text{ or } a_{\tau}^{\mathrm{ref}} = 0, \\
0, & \text{if } a_{\tau}^{\mathrm{ref}} = -1, \\
2, & \text{if } a_{\tau}^{\mathrm{ref}} = 1,
\end{cases}
\tag{6}
$$

Executing $a_{\tau}^{\mathrm{final}}$ yields the realized step-wise profit $r_{\tau}^{\mathrm{step}}$. The refinement policy $\pi_{\omega}^{\mathrm{ref}}(a_{\tau}^{\mathrm{ref}} \mid s_{\tau}^{\mathrm{ref}})$ is trained to issue an override at most once within each horizon, preventing the adjustment from deviating excessively from the selected archetype’s intended strategy. Formally, $a_{\tau}^{\mathrm{ref}} \neq 0$ for a single $\tau \in [t, t+h-1]$ per horizon and $a_{\tau}^{\mathrm{ref}} = 0$ for other timesteps. To enable the refinement policy to jointly consider real-time market signals and the archetype’s evolving performance, we apply adaptive layer normalization (Peebles and Xie 2023; Zong et al. 2024) to condition market state $s_{\tau}^{\mathrm{ref1}}$ on archetype context $s_{\tau}^{\mathrm{ref2}}$ in practice.

However, because only a single-step adaptation is allowed per horizon, the policy must carefully identify the most impactful moment to intervene rather than squandering its opportunity on minor market fluctuations or prematurely overriding an archetype strategy that may yield greater returns later. To focus the refinement policy on meaningful adjustments, we compute the top-5 hindsight-optimal adaptations via DP and introduce a novel regret-aware reward function that penalizes the agent not only for failing to outperform the base strategy but also for missing out on the best possible adaptation. Concretely, the DP solver returns:
$$
O_{\mathrm{top5}}=\left\{(\tau_{\mathrm{opt}}^{n}, a_{\mathrm{opt}}^{n}, R_{\mathrm{opt}}^{n})\right\}_{n=1}^{5}
\tag{7}
$$

where $\tau_{\mathrm{opt}}^{n}$ is the adaptation timestep, $a_{\mathrm{opt}}^{n} \in \{-1,1\}$ is the adaptation action, and $R_{\mathrm{opt}}^{n}$ is the horizon’s cumulative return of executing the adaptation. Instead of directly using the immediate step-wise reward $r_{\tau}^{\mathrm{step}}$, our regret-augmented reward function for the policy adaptation agent is formulated as follows:

$$
r_{\tau}^{\mathrm{ref}}=
\begin{cases}
(R-R_{\mathrm{base}})+\beta_{1}(R-R_{\mathrm{opt}}^{1}), & \text{if } a_{\tau}^{\mathrm{ref}} \neq 0, \\
0, & \text{otherwise}
\end{cases}
\tag{8}
$$
where $R=\sum_{\tau=0}^{h-1} r_{\tau}^{\mathrm{step}}$ is the horizon’s cumulative return of taking action $a_{\tau}^{\mathrm{ref}}$, $R_{\mathrm{base}}$ is the horizon’s return under the base archetype policy, $R_{\mathrm{opt}}^{1}$ is the horizon’s maximum return from the top DP-identified adaptation and $\beta_{1}$ is the hyper-parameter controlling the tolerance for suboptimality. This reward structure encourages profitable adjustments relative to the base strategy while penalizing the gap from the best possible adaptation. By striking this balance, the policy avoids wasting its single adaptation opportunity on negligible improvements and instead aims to deploy its intervention at the most valuable moment. Because only one adaptation is allowed per horizon, the RL episode terminates as soon as the adapter chooses a non-zero action. Finally, the objective function for training the refinement policy is given by
$$
J' = \mathbb{E}_{\pi_{\omega}^{\mathrm{ref}}} \left[ \sum_{\tau=0}^{h-1} \left( \gamma^{\tau} r_{\tau}^{\mathrm{ref}} - \beta_2 L\!\left(\hat{a}_{\tau}^{\mathrm{ref}}, \pi_{\omega}^{\mathrm{ref}}(a_{\tau}^{\mathrm{ref}} \mid s_{\tau}^{\mathrm{ref}})\right) \right) \right]
\tag{9}
$$

where $\hat{a}_{\tau}^{\mathrm{ref}}$ denotes the optimal adaptation action ($\hat{a}_{\tau}^{\mathrm{ref}} = a_{\mathrm{opt}}^{n}$ if $\tau = \tau_{\mathrm{opt}}^{n}$, and $0$ otherwise), and $L$ is the cross-entropy loss guiding the refinement policy toward optimal behavior. By optimizing this objective, we encourage the policy adapter to perform high-impact adjustments to the base archetype actions, ultimately yielding a more profitable and robust trading strategy.
