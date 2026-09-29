RESULT done

CHANGED: Released $DATA_DIR/releases/S172_vis.zip on exact S171. SHA256 2969891d7472a8fb8bcc6195fee97bf4f55185163149bf71869b5c46fdced549; 116 members, 1,788,774,467 bytes. The frozen DINOv2-L arm uses 2.5 Hz, batch 16 and FP16 autocast. Only collision and min(S171 entry, collision) may change. A 600-second visual / 45-minute process guard preserves S171 for remaining clips and logs coverage.

VERIFIED: Cache-based original five-fold OOF 482/750 versus S160 proxy 458/750; 4 Hz scored 485, 2 Hz 474. On the fixed 150-clip fold 0, exported FP16, cached FP16 and separate FP32 control each scored 103/150 with identical picks. Mocked-clock guard, offline public/long CUDA QA, fallback, field isolation, Stage3, validator and archive hashes passed. Peak QA RSS 3.30 GiB; CUDA reserved 1.43 GiB. Fresh shared-RTX local projection: 456.828 seconds (7.614 added minutes) for 137 long clips.

RISKS: Exported descriptors differ numerically from cache; full 482/750 is a proxy. OOF selection reused clips. L40S runtime and official S172 gain remain unmeasured; S171 remains official champion.

