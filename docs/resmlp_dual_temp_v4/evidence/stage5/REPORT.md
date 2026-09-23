# Stage 5 report — six standard PDE tasks

A. Scope/status: PASS on synthetic native closures. Airfoil, Darcy, Elasticity, Pipe, Navier–Stokes and Plasticity use their native entry AST/control flow with v4 dispatch.

B. Files: standard entry/dispatch and `tran_evaluate/linearno_loop_v4/{launch,_task,airfoil,darcy,elasticity,pipe,ns,plasticity}.sh`.

C. Protocol mapping: original task H/head/M/FFN ratio, normalizers, objectives, batch/optimizer/scheduler, NS ten-step teacher-forced/prediction feedback and Plasticity twenty time-conditioned updates are retained. v4 adds only architecture selection and isolated output/checkpoint ownership.

D. Evidence: stable native matrix is now complete for all six tasks × two dynamic modes: interrupt/resume/eval and uninterrupted latent-K exact-weight comparison, 42 rows with exit 0 in `stage7/native-matrix.json` (plus industrial rows). Full v4 suite and pure LinearNO static/temporal integration passed. No real data or long training.

E. Frozen region: original task data and metric code remain unchanged; synthetic inputs are explicitly labeled synthetic.

F. Review points: verify Plasticity’s twenty updates and NS’s ten calls in native reports, and confirm saved-config conflicts fail before tensor loading.

本阶段结束，未执行下一阶段。
