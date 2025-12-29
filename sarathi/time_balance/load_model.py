from sarathi.time_balance.config import MODEL_CACHE_PATH
from sarathi.time_balance.predict_time import TimePredictor

# (decode_tokens, decode_history_tokens, batch_request_count, prefill_tokens, prefill_processed_tokens)
CASES = [
    (26, 11039, 27, 230, 29),
    (30, 16472, 32, 126, 0),
    (30, 10715, 32, 34, 0),
    (31, 4575, 32, 225, 0),
    (31, 12406, 32, 73, 0),
    (31, 17359, 32, 39, 0),
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


# 26,11039,27,230,29,75.14521789550781
# 30,16472,32,126,0,63.63542556762695
# 30,10715,32,34,0,30.249919891357422
# 31,4575,32,225,0,71.9728012084961
# 31,12406,32,73,0,37.017601013183594
# 31,17359,32,39,0,40.05171203613281