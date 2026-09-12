# Phoenix-2014T pre-repair result index (`81629b2f`)

> **Historical results only — do not present these paths as results from the repaired dataset.**

This directory is a lightweight, non-destructive index. It does not contain copied experiment outputs. The relative symbolic links under `links/` point to the original locations so existing reproduction paths remain valid.

## Data identity

- Pre-repair video SHA-256: `81629b2f3879a189613d87dafcbe04fa5053315ea9f9cfdb2f855d7a35007e30`
- Repaired video SHA-256: `49faacc304666a75cb51e3e2d335dfbead8d08e8dd5ff834c66c690e1175d457`
- The pre-repair archive lacked 61 metadata-declared frames: 54 train frames and 7 dev frames. Test had no missing frames.
- Recovery completed on 2026-09-08. See `code_agent_logs/2026-09-08/phoenix_video_frame_recovery.md`.

## Classification rule

`confirmed_pre_repair` requires one of the following:

1. a protocol manifest recording the full pre-repair SHA-256; or
2. a Phoenix run completed before the atomic recovery, corroborated by its result/log timestamps and the recovery audit stating that existing results were produced with the old standard archive.

Directory names alone were not used as proof. Items without adequate evidence are listed as `unverified`, not linked as confirmed results.

## Inventory summary

| Classification | Directories | Logical size | Notes |
|---|---:|---:|---|
| Confirmed pre-repair, completed | 13 | 2,791,228,054 B (2.60 GiB) | Linked under `links/results/` |
| Confirmed pre-repair, empty/interrupted | 4 | 0 B | Linked and explicitly marked in the inventory |
| Confirmed pre-repair root logs | 16 files | 35,581 B | Linked under `links/logs/` |
| Confirmed repaired results | 3 | 914,655,251 B (872.28 MiB) | Listed only; deliberately not linked here |
| Unverified/non-result artifacts | 3 | 456,070,204 B | Checkpoint, source snapshot, and service log |

The 41,720,845,149-byte pre-repair ZIP is linked at `links/data/PHOENIX2014T_videos.incomplete_81629b2f.zip`. A symbolic link consumes negligible space and does not duplicate the ZIP.

Machine-readable records are in `inventory.json` and `inventory.csv`. `unverified.csv` separates items that must not be treated as confirmed pre-repair results.

## Interpretation

The repaired dev correctness matrix reproduced the same aggregate WER and clip counts, but only because the affected sample still decoded to an empty hypothesis. Its logits changed, proving that restored frames were read. Old and repaired outputs remain distinct provenance groups despite equal aggregate metrics.

Do not delete or move targets through this index. If targets are relocated later, update these relative links and the inventory together.
