# Mathematics and PyTorch map

Let `B` be batch size, `T` time samples and `i=1,...,5` appliance identity. Power is stored in watts divided by the shared scale `S=4000`.

## Stage 1 — train CNN weights

`models/cnn.py` computes

\[
z_i=E_\psi(p_i),\qquad \hat p_i=G_\theta(z_i,i),\qquad
L_{rec}=\operatorname{mean}_{b,i,t}(\hat p_{b,i,t}-p_{b,i,t})^2.
\]

In `stage1_forward/train.py`, `model(target)` is the forward pass, `loss.backward()` computes gradients for encoder and decoder parameters \((\psi,\theta)\), and `optimizer.step()` updates those parameters. `stage1_forward/losses.py` also records the optional ON/OFF-balanced experimental loss.

The archived F1 experiment additionally unrolled three latent updates during training and backpropagated through them with `create_graph=True`. It is an historical experimental loss, not the default Stage-1 objective.

## Stage 2 — optimise latent codes, not CNN weights

The decoder is frozen. `stage2_inverse/inverse.py` makes `codes.requires_grad_()` and creates `Adam([codes], lr=...)`. It decodes

\[
s(z)=\sum_iG_{\theta^*}(z_i,i),\qquad
b^*(z)=\max(0,\operatorname{mean}_t[y(t)-s(z,t)]),
\]
\[
L_{meter}(z)=\operatorname{mean}_t[s(z,t)+b^*(z)-y(t)]^2.
\]

`loss.backward()` is still required because autograd differentiates through the frozen decoder to the codes. Frozen parameters receive no optimizer update; `optimizer.step()` changes only `codes`. The solver retains the best candidate per window and may add the existing optional diagonal latent prior.
