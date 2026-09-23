# Stage 3 report — dual adaptive temperature attention

A. Scope/status: PASS. Added base, latent-K/point-Q, and point-K/point-Q modes across six attention variants.

B. Files: `cdlno/linearno_loop/v4/{attention,temperature}.py` and attention tests.

C. Formula mapping: routing feature is `[B,Hd,N,d_h]`; Q predictor gives `[B,Hd,N,1]`; latent-K uses mean over N and gives `[B,Hd,1,M]`; point-K gives `[B,Hd,N,1]`. `tau=tau_base*exp(log(2)*tanh(delta))`, K softmax is over N and Q softmax over M, followed by `K^T V` and `QZ`. Predictors are per operator, head-shared, and terminal layers are zero initialized.

D. Tests: six variants, zero-init parity, nonzero predictor gradients, temperature ordering, softmax axes, ShapeNet M independence, AMP dtypes, dropout RNG and strict state reload passed; one optional CUDA/torch-cluster boundary is skipped where unavailable.

E. Frozen region: original attention classes are delegated/reused on feature-off paths; no old attention math was changed for the new tests.

F. Review points: temperature is applied to base logits before softmax, dynamic tau is not reclamped, and no N×N/M×M tensor is created.

本阶段结束，未执行下一阶段。
