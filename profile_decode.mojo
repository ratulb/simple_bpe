"""Profile decode steps."""
from tokenizer import BPETokenizer
from std.pathlib import Path
from std.time import perf_counter_ns
from std.memory import alloc, memcpy


def main() raises:
    var text = Path("benchmarks/corpus.txt").read_text()
    var corpus = List[String]()
    corpus.append(String(text))
    var tok = BPETokenizer()
    print("Training...")
    tok.train(corpus, 500)
    var ids = tok.encode(text)

    var total_len: Int = 0

    # Warmup
    for _ in range(5):
        total_len += tok.decode(Span[Int](ids)).byte_length()

    # Pass 1: sum lengths
    var t0 = perf_counter_ns()
    for _ in range(20):
        var total: Int = 0
        for id in ids:
            total += tok.storage.token_length(id)
    var t1 = perf_counter_ns()
    print("pass 1 (sum):      " + String(Float64(t1 - t0) / 1e6 / 20) + " ms")

    # Pass 2: memcpy chain (alloc + memcpy per token) + from_utf8_lossy
    var t2 = perf_counter_ns()
    for _ in range(20):
        var total2: Int = 0
        for id2 in ids:
            total2 += tok.storage.token_length(id2)
        var buf2 = alloc[UInt8](total2)
        var ptr2 = tok.storage.buffer_ptr()
        var off2: Int = 0
        for id2 in ids:
            var n2 = tok.storage.token_length(id2)
            if n2 > 0:
                memcpy(dest=buf2 + off2, src=ptr2 + tok.storage.token_offset(id2), count=n2)
                off2 += n2
        total_len += String(from_utf8_lossy=Span[UInt8](ptr=buf2, length=total2)).byte_length()
        buf2.free()
    var t3 = perf_counter_ns()
    print("pass 2 (memcpy):   " + String(Float64(t3 - t2) / 1e6 / 20) + " ms")

    # Full decode
    var t8 = perf_counter_ns()
    for _ in range(20):
        total_len += tok.decode(Span[Int](ids)).byte_length()
    var t9 = perf_counter_ns()
    print("full decode:       " + String(Float64(t9 - t8) / 1e6 / 20) + " ms")

    print("Result len sum:", total_len)
