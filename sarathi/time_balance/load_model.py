from sarathi.time_balance.config import MODEL_CACHE_PATH
from sarathi.time_balance.predict_time import TimePredictor

# (decode_tokens, sum_decode_context_len, batch_request_count, prefill_tokens, prefill_processed_tokens, gpu_mem_used_mb, gpu_mem_free_mb=gmft, cuda_allocated_mb, cuda_reserved_mb)
CASES = [
    (9,2362,10,247,0,21588.4375,2533.75,16867.11865234375,16962.0),
    (7,2417,11,249,77,21457.0625,2665.125,16867.11865234375,16962.0),
]


def main() -> None:
    predictor = TimePredictor.load(MODEL_CACHE_PATH)

    t_ms = [
        predictor.predict(
            decode_tokens=dt,
            sum_decode_context_len=dht,
            batch_request_count=brc,
            prefill_tokens=pt,
            prefill_processed_tokens=ppt,
            gpu_mem_used_mb=gmpt,
            gpu_mem_free_mb=gmft,
            cuda_allocated_mb=cam,
            cuda_reserved_mb=crm
        ).item()
        for (dt, dht, brc, pt, ppt, gmpt, gmft, cam, crm) in CASES
    ]

    print(t_ms)


if __name__ == "__main__":
    main()


# 9,2362,10,247,0,21588.4375,2533.75,16867.11865234375,16962.0,70.28121948242188
# 7,2417,11,249,77,21457.0625,2665.125,16867.11865234375,16962.0,72.86681365966797