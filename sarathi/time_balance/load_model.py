from sarathi.time_balance.config import MODEL_CACHE_PATH
from sarathi.time_balance.predict_time import TimePredictor

# (decode_tokens, decode_history_tokens, batch_request_count, prefill_tokens, prefill_processed_tokens)
CASES = [
    (28,11042,29,85,228),
    (12,8673,13,244,62),
    (11,6741,13,245,334),
    (20,11198,21,236,0),
    (20,11218,21,60,236),
]


def main() -> None:
    predictor = TimePredictor.load(MODEL_CACHE_PATH)

    t_ms = [
        predictor.predict(
            decode_tokens=dt,
            decode_history_tokens=dht,
            batch_request_count=brc,
            prefill_tokens=pt,
            prefill_processed_tokens=ppt,
        ).item()
        for (dt, dht, brc, pt, ppt) in CASES
    ]

    print(t_ms)


if __name__ == "__main__":
    main()


# 28,11042,29,85,228,45.899776458740234
# 12,8673,13,244,62,79.68870544433594
# 11,6741,13,245,334,81.3465576171875
# 20,11198,21,236,0,79.96438598632812
# 20,11218,21,60,236,45.36832046508789