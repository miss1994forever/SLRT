# Project execution constraints

## GPU safety

- Never run CUDA workloads on the physical GPU at PCI bus ID `0000:25:00.0`
  (UUID `GPU-06afe121-c4ce-b981-bb86-399e4a85ae83`).
- Quarantine any additional GPU that is absent from `nvidia-smi`, reports PCI
  revision `ff`, or fails the CUDA preflight. As of 2026-10-08 this also applies
  to PCI `0000:c1:00.0` (UUID `GPU-1c25de34-7fce-3f81-158c-c4d82d3ac729`)
  until it passes a fresh health check.
- Before every GPU job, query the current logical-index to PCI-bus-ID mapping with
  `nvidia-smi`. Abort the job if the mapping cannot be verified.
- Do not identify the excluded device only as "GPU 1": logical indices can change
  after driver recovery or when `CUDA_VISIBLE_DEVICES` is used.
- Prefer selecting verified healthy devices by UUID. A successful `nvidia-smi`
  query is insufficient by itself; verify `cuInit` or `torch.cuda.is_available()`
  before launching training.

## Phoenix-2014T evaluation policy

- Do not read, evaluate, or tune on the Phoenix-2014T test split unless the user
  explicitly authorizes test access in the current conversation.
- Use source-disjoint train calibration for candidate selection and reserve dev
  for the preregistered confirmation step.
